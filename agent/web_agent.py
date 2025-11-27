import re
import time
import json
from pathlib import Path
from urllib.parse import quote_plus
try:
    from playwright.sync_api import sync_playwright, Page
except ImportError:  # Allow running tests without playwright installed
    sync_playwright = None
    class Page:  # minimal placeholder for type hints
        pass
from agent.actions import action_goto, action_click, action_scroll, VALID_ACTIONS

class WebAgent:
    """
    WebAgent with goal validation and auto search fallback for tasks without explicit URL.
    - Auto builds search query from natural language task.
    - Validates completion (only accepts DONE/completed if goal info detected).
    - Fallback to alternative search engine if first page blank.
    """

    def __init__(self, vllm, action_repeat_limit: int = 3):
        self.vllm = vllm
        self.output_dir = Path("web_agent_output")
        self.output_dir.mkdir(exist_ok=True)
        self.history = []
        self.last_actions = []
        self.action_repeat_limit = action_repeat_limit

    def run(self, user_goal: str, max_steps: int = 6):
        print(f"[Agent] Task start: {user_goal}")
        if sync_playwright is None:
            print("[Agent] Playwright unavailable. Activating HTTP fallback agent.")
            return self._fallback(user_goal)
        try:
            p = sync_playwright().start()
            browser = p.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu"
            ])
        except Exception as e:
            print(f"[Agent] Browser launch failed: {e}. Using fallback.")
            try:
                p.stop()
            except Exception:
                pass
            return self._fallback(user_goal)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()

            start_url = self._extract_first_url(user_goal)
            if not start_url:
                # Build search query if no explicit URL
                query = self._build_search_query(user_goal)
                start_url = f"https://www.google.com/search?q={quote_plus(query)}"
            self._safe_goto(page, start_url)

            # Blank page fallback
            if self._page_is_blank(page):
                print("[Agent] First page appears blank. Fallback to DuckDuckGo.")
                query = self._build_search_query(user_goal)
                self._safe_goto(page, f"https://duckduckgo.com/?q={quote_plus(query)}")

            for step in range(max_steps):
                print(f"\n{'-'*60}\n[Step {step+1}/{max_steps}]")
                screenshot_path = self.capture_screenshot(page, step)
                dom_summary = self.get_dom_summary(page)
                current_url = page.url

                plan = self._analyze_with_model(
                    screenshot_path, dom_summary, user_goal, current_url
                )
                self._record_step(step, screenshot_path, dom_summary, plan, current_url)

                print(f"[Plan] action={plan.get('action')} parameter={plan.get('parameter')} completed={plan.get('completed')}")

                if plan.get("completed") or plan.get("action", "").upper() == "DONE":
                    if self._goal_completed(user_goal, dom_summary):
                        print("[Validate] Goal confirmed. Stop.")
                        break
                    else:
                        print("[Validate] Goal info not found. Continue exploring.")
                        plan["completed"] = False
                        # Nudge exploration if premature DONE
                        if plan.get("action", "").upper() == "DONE":
                            plan["action"] = "SCROLL"

                if self._is_loop(plan.get("action", ""), plan.get("parameter", "")):
                    print("[Loop] Repeated action detected. Stop.")
                    break

                if not self.execute_action(page, plan):
                    print("[Agent] Action failed or cannot continue. Stop.")
                    break

                # If still blank after actions, attempt second fallback search site
                if step == 1 and self._page_is_blank(page):
                    print("[Agent] Page still blank after interactions. Second fallback to Bing.")
                    query = self._build_search_query(user_goal)
                    self._safe_goto(page, f"https://www.bing.com/search?q={quote_plus(query)}")

        self._final_screenshot(page)
        context.close()
        browser.close()
        try:
            p.stop()
        except Exception:
            pass
        self._save_log()
        print(f"[Agent] Finished: {self.output_dir.resolve()}")

    def run_and_summarize(self, task):
        """
        Run web agent on a task and produce a summary dict for evaluation.
        Ensures a stable output format for evaluation.py.
        """
        self.history = []
        self.task = task

        print(f"[Agent] Task start: {task}")
        result = self.run(task)   # 你原来的 run()

        # ---- 处理 result 为空的情况 ----
        if result is None:
            result = {"completed": False}

        completed = bool(result.get("completed", False))
        steps = len(self.history)

        # ---- 修复：action 是字符串，不是 dict ----
        invalid_actions = 0
        for h in self.history:
            action = h.get("action", "")
            res = h.get("result", {})

            # action 是字符串，判断是否包含 DONE
            if isinstance(action, str) and action.strip().upper().startswith("DONE"):
                # 只要 model 认为 DONE 但没有完成 → 计为无效动作
                if not (isinstance(res, dict) and res.get("completed", False)):
                    invalid_actions += 1

        return {
            "completed": completed,
            "steps": steps,
            "invalid_actions": invalid_actions
        }

    # ---------------- Utilities ----------------

    def _build_search_query(self, goal: str) -> str:
        """
        Convert natural language to concise search query.
        Removes stop words; keeps nouns/adjectives (simple heuristic).
        """
        stop = {"the", "of", "and", "on", "find", "get", "go", "to", "latest", "what", "for"}
        tokens = re.findall(r"[a-zA-Z0-9]+", goal.lower())
        filtered = [t for t in tokens if t not in stop]
        # Special normalization for date: keep year and month/day terms
        return " ".join(filtered)[:120] or goal

    def _page_is_blank(self, page: Page) -> bool:
        try:
            html = page.content()
            body_len = len(html)
            text = ""
            try:
                text = page.locator("body").inner_text()
            except:
                pass
            return (body_len < 300) or (len(text.strip()) == 0)
        except:
            return True

    # ---------------- Screenshot & DOM ----------------

    def capture_screenshot(self, page: Page, step: int) -> str:
        path = self.output_dir / f"step_{step}_screenshot.png"
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            page.wait_for_load_state("networkidle", timeout=8000)
        except:
            pass
        try:
            page.evaluate("window.scrollTo(0, 0)")
        except:
            pass
        time.sleep(0.6)
        page.screenshot(path=str(path), full_page=True)
        print(f"[Screenshot] Saved: {path}")
        return str(path)

    def get_dom_summary(self, page: Page) -> str:
        try:
            title = page.title()
        except:
            title = ""
        url = page.url

        def safe_count(sel):
            try:
                return page.locator(sel).count()
            except:
                return 0

        buttons = safe_count("button")
        links = safe_count("a")

        meta_info = ""
        try:
            body_text = page.locator("body").inner_text()[:5000]
            # Temperature extraction heuristic (e.g., 25°C, 77 F, 25° C)
            temps = re.findall(r"\b-?\d{1,2}\s?°\s?[CFcf]?\b", body_text)
            if temps:
                meta_info += f"\n- Temps found: {', '.join(list(dict.fromkeys(temps))[:5])}"
            if "hong kong" in body_text.lower():
                meta_info += "\n- 'Hong Kong' keyword present."
        except Exception as e:
            meta_info += f"\n- Text extraction failed: {e}"

        summary = f"""Page Info:
- Title: {title}
- URL: {url}
- Buttons: {buttons}
- Links: {links}{meta_info}"""
        return summary.strip()

    # ---------------- Model Interaction ----------------

    def _analyze_with_model(self, screenshot_path, dom_summary, user_goal, current_url):
        try:
            result = self.vllm.analyze(
                screenshot_path=screenshot_path,
                dom_summary=dom_summary,
                user_goal=user_goal,
                current_url=current_url
            )
            for k in ["thought", "action", "parameter", "completed"]:
                if k not in result:
                    result[k] = "" if k != "completed" else False
            valid = {"GOTO", "CLICK", "SCROLL", "DONE"}
            act = str(result["action"]).upper().strip()
            if "/" in act or act not in valid:
                act_clean = act.split("/")[0].split()[0]
                result["action"] = act_clean if act_clean in valid else "SCROLL"
            return result
        except Exception as e:
            print(f"[Analyze ERROR] {e}")
            return {"thought": "model error", "action": "DONE", "parameter": "", "completed": True}

    # ---------------- Action Execution ----------------

    def execute_action(self, page: Page, plan: dict) -> bool:
        action = plan.get("action", "").upper().strip()
        parameter = (plan.get("parameter") or "").strip()
        print(f"[Execute] action={action} parameter={parameter}")

        if action == "GOTO":
            return action_goto(self, page, parameter)
        if action == "CLICK":
            return action_click(self, page, parameter)
        if action == "SCROLL":
            return action_scroll(self, page)
        if action == "DONE":
            print("[DONE] Model indicates completion.")
            return False

        print(f"[Execute] Unknown action: {action}")
        return False

    # ---------------- Action Implementations ----------------

    # Legacy wrapper names removed; logic now in agent.actions
    def _action_goto(self, page: Page, parameter: str) -> bool:  # backward compatible for tests
        return action_goto(self, page, parameter)

    def _action_click(self, page: Page, parameter: str) -> bool:  # backward compatible for tests
        return action_click(self, page, parameter)

    def _action_scroll(self, page: Page) -> bool:  # backward compatible for tests
        return action_scroll(self, page)

    def _safe_goto(self, page: Page, url: str) -> bool:
        print(f"[GOTO] Navigate: {url}")
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=35000)
            if resp:
                print(f"[GOTO] Status: {resp.status}")
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except:
                pass
            time.sleep(1.8)
            return True
        except Exception as e:
            print(f"[GOTO] Failed: {e}")
            return False

    # ---------------- Completion Check ----------------

    def _goal_completed(self, goal: str, dom_summary: str) -> bool:
        g = goal.lower()
        dom = dom_summary.lower()
        if "hong kong" in g and "temperature" in g:
            has_city = "hong kong" in dom
            # Accept patterns like 25°C / 25 °C / 25° C / 77°F
            has_temp = bool(re.search(r"\b-?\d{1,2}\s?°\s?[cf]?\b", dom))
            return has_city and has_temp
        if "python" in g and "version" in g:
            return bool(re.search(r"python\s+\d+\.\d+\.\d+", dom))
        return False

    # ---------------- Loop & History ----------------

    def _is_loop(self, action: str, parameter: str) -> bool:
        sig = f"{action}:{parameter}"
        self.last_actions.append(sig)
        if len(self.last_actions) > self.action_repeat_limit:
            self.last_actions = self.last_actions[-self.action_repeat_limit:]
        return len(self.last_actions) == self.action_repeat_limit and len(set(self.last_actions)) == 1

    def _record_step(self, step, screenshot, dom_summary, plan, url):
        self.history.append({
            "step": step,
            "screenshot": screenshot,
            "dom_summary": dom_summary,
            "thought": plan.get("thought", ""),
            "action": plan.get("action", ""),
            "parameter": plan.get("parameter", ""),
            "url": url
        })

    def _final_screenshot(self, page: Page):
        final_path = self.output_dir / "final_screenshot.png"
        try:
            page.wait_for_load_state("networkidle", timeout=6000)
        except:
            pass
        time.sleep(0.8)
        page.screenshot(path=str(final_path), full_page=True)
        print(f"[Final] Saved: {final_path}")

    def _extract_first_url(self, text: str):
        m = re.search(r"https?://[^\s]+", text)
        return m.group(0) if m else None

    def _save_log(self):
        log_path = self.output_dir / "execution_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"[Log] Saved: {log_path}")
        return log_path

    # ---------------- Fallback (no browser) ----------------
    def _fallback(self, task: str):
        try:
            from agent.fallback_agent import RequestsAgent
        except ImportError:
            print("[Fallback] requests/bs4 not available; please install beautifulsoup4.")
            return {"completed": False, "reason": "missing_dependencies"}
        fa = RequestsAgent(model=self.vllm)
        result = fa.run(task)
        self.history.append({"step": 0, "action": "FALLBACK", "result": result})
        self._save_log()
        return result