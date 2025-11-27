import time
try:
    from playwright.sync_api import Page
except ImportError:
    class Page:  # placeholder
        pass

# Action implementations extracted from WebAgent for modular management.
# Each function returns bool indicating success.

VALID_ACTIONS = {"GOTO", "CLICK", "SCROLL", "DONE"}

def action_goto(agent, page: Page, parameter: str) -> bool:
    if not parameter:
        print("[GOTO] Missing URL.")
        return False
    current_url = getattr(page, "url", "")
    url = parameter if parameter.startswith(("http://", "https://")) else f"https://{parameter}"
    if url == current_url:
        print("[GOTO] Same URL; skip.")
        return True
    return agent._safe_goto(page, url)

def action_click(agent, page: Page, parameter: str) -> bool:
    if not parameter:
        print("[CLICK] Missing descriptor.")
        return False
    selectors = [
        f"text={parameter}",
        f"a:has-text('{parameter}')",
        f"button:has-text('{parameter}')",
        f"//*[contains(text(), '{parameter}')]",
        f"[aria-label*='{parameter}' i]",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.scroll_into_view_if_needed()
                time.sleep(0.3)
                loc.click(timeout=6000)
                try:
                    page.wait_for_load_state("networkidle", timeout=9000)
                except:
                    pass
                time.sleep(1.2)
                print(f"[CLICK] Success via {sel}")
                return True
        except Exception:
            continue
    print(f"[CLICK] Not found: {parameter}")
    return False

def action_scroll(agent, page: Page) -> bool:
    try:
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        time.sleep(1)
        print("[SCROLL] Done.")
        return True
    except Exception as e:
        print(f"[SCROLL] Failed: {e}")
        return False
