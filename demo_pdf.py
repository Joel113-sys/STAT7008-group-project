"""
VLLM + Playwright Web Agent Demo - Transformers Version
Modified to load remote models via Hugging Face Transformers (no local GGUF).
"""

import os
import json
import re
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from playwright.sync_api import sync_playwright, Page, Browser
import time

from utils.file_tools import WindowsFileTools

class WebAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """Initialize the web agent with Transformers remote model (no local files)."""
        print(f"[Model] Loading Transformers model: {model_name}")
        try:
            # 加载支持视觉-文本的 Qwen2-VL 指令模型
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.device = self.model.device
            print("[Model] ✓ Transformers model loaded")
        except Exception as e:
            print(f"[Model] ✗ Failed to load Transformers model: {e}")
            raise
        
        # Create output directory
        self.output_dir = Path("web_agent_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Session history
        self.history = []
        
        # Loop detection
        self.last_actions = []
        self.action_limit = 3
        
        self.file_tools = WindowsFileTools()
    
    def analyze_page(self, screenshot_path: str, dom_summary: str, user_goal: str, current_url: str) -> dict:
        """Analyze page using Transformers model and decide next action"""
        print("\nAnalyzing page with Transformers model...")
        
        prompt = f"""You are a web browsing assistant.

User Goal: {user_goal}

Current Page Info:
{dom_summary}

Current URL: {current_url}

IMPORTANT CONTEXT:
- You are already on the page: {current_url}
- If the current page already shows the information needed, use DONE action
- Do NOT navigate to the same URL you're already on
- If you see version information in the page info above, the task may already be complete

Based on the page information, decide the next action.

Available actions:
1. GOTO - Navigate to a DIFFERENT URL (only if you need to go somewhere else)
2. CLICK - Click an element on the current page
3. SCROLL - Scroll down to see more content
4. DONE - Task complete (use this if you found the information)

CRITICAL: Use ONLY standard ASCII punctuation in your JSON response:
- Use double quotes: " (NOT " or ")
- Use commas: , (NOT ，)
- Use colons: : (NOT ：)
- Use periods: . (NOT 。)

Respond with ONLY valid JSON in this exact format:
{{
    "thought": "explain your reasoning",
    "action": "ACTION_NAME",
    "parameter": "parameter value",
    "completed": false
}}

Examples:
{{"thought": "Need to visit Downloads page", "action": "CLICK", "parameter": "Downloads", "completed": false}}
{{"thought": "Found Python 3.14.0 on current page", "action": "DONE", "parameter": "", "completed": true}}

Your JSON response:"""
        
        # Generate response using Transformers model
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
            chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[chat_text], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=256, temperature=0.3)
            gen_ids = out[:, inputs.input_ids.shape[1]:]
            output_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            print(f"Raw model response:\n{output_text}\n")
        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            output_text = '{"thought": "Error processing request", "action": "DONE", "parameter": "", "completed": true}'
        
        # 其余解析逻辑保持不变...
        output_text = self.normalize_to_ascii(output_text)
        print(f"After normalization:\n{output_text}\n")
        
        # Try to parse JSON
        try:
            start_idx = output_text.find("{")
            end_idx = output_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = output_text[start_idx:end_idx]
                json_str = re.sub(r'\s+', ' ', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                print(f"Cleaned JSON string:\n{json_str}\n")
                
                try:
                    action_plan = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parse failed: {e}")
                    print(f"Failed JSON: {json_str}")
                    raise
                
                if "action" in action_plan:
                    action_str = str(action_plan["action"]).strip()
                    if " " in action_str:
                        parts = action_str.split(None, 1)
                        action_plan["action"] = parts[0].upper()
                        if len(parts) > 1 and "parameter" not in action_plan:
                            param_str = parts[1].strip().strip("'\"")
                            action_plan["parameter"] = param_str
                            print(f"Extracted parameter from action: {param_str}")
                    else:
                        action_plan["action"] = action_str.upper()
                
                if "parameter" not in action_plan:
                    action_plan["parameter"] = ""
                
                if self.detect_loop(action_plan["action"], action_plan["parameter"]):
                    print("🔄 Forcing DONE due to loop detection")
                    return {
                        "thought": "Detected repeated actions, marking task as complete",
                        "action": "DONE",
                        "parameter": "",
                        "completed": True
                    }
                
                print(f"✓ Parsed action plan:\n{json.dumps(action_plan, indent=2, ensure_ascii=False)}\n")
                return action_plan
            else:
                raise ValueError("No JSON block found in response")
                
        except Exception as e:
            print(f"❌ JSON parsing completely failed: {e}")
            print(f"Attempting intelligent fallback parsing...\n")
            action_plan = self.fallback_parse(output_text, user_goal)
            
            if self.detect_loop(action_plan["action"], action_plan["parameter"]):
                return {
                    "thought": "Loop detected in fallback",
                    "action": "DONE",
                    "parameter": "",
                    "completed": True
                }
            
            return action_plan

    # 其余方法保持不变...
    def capture_screenshot(self, page, step: int) -> str:
        """Capture and save screenshot"""
        screenshot_path = self.output_dir / f"step_{step}_screenshot.png"
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
            time.sleep(2)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)
            page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            page.screenshot(path=str(screenshot_path), full_page=False)
        return str(screenshot_path)
    
    def get_dom_summary(self, page: Page) -> str:
        """Get simplified DOM summary"""
        try:
            title = page.title()
            url = page.url
            buttons = page.locator("button").count()
            links = page.locator("a").count()
            inputs = page.locator("input").count()
            
            visible_text = ""
            try:
                body_text = page.locator("body").inner_text()
                version_pattern = r'Python\s+\d+\.\d+\.\d+'
                versions = re.findall(version_pattern, body_text)
                if versions:
                    unique_versions = list(set(versions))
                    visible_text = f"\n- Found versions on page: {', '.join(unique_versions[:5])}"
                
                if "Latest:" in body_text or "latest" in body_text.lower():
                    latest_match = re.search(r'Latest:?\s*(Python\s+\d+\.\d+\.\d+)', body_text, re.IGNORECASE)
                    if latest_match:
                        visible_text += f"\n- Latest version shown: {latest_match.group(1)}"
                
                if "download" in body_text.lower() and "release" in body_text.lower():
                    visible_text += "\n- This appears to be a downloads/releases page"
            except Exception as e:
                print(f"Could not extract page text: {e}")
            
            summary = f"""Page Information:
- Title: {title}
- URL: {url}
- Buttons: {buttons}
- Links: {links}
- Input fields: {inputs}{visible_text}"""
            
            return summary.strip()
        except Exception as e:
            return f"Error getting DOM: {str(e)}"
    
    def normalize_to_ascii(self, text: str) -> str:
        """Convert all Chinese/special punctuation to standard ASCII"""
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'", '´': "'", '`': "'",
            '，': ',', '。': '.', '、': ',', '：': ':', '；': ';',
            '（': '(', '）': ')', '【': '[', '】': ']', '「': '"', '」': '"', '『': '"', '』': '"',
            '！': '!', '？': '?', '—': '-', '–': '-', '…': '...', '·': '.',
            '\u3000': ' ', '\xa0': ' ',
        }
        for chinese, english in replacements.items():
            text = text.replace(chinese, english)
        return text
    
    def detect_loop(self, action: str, parameter: str) -> bool:
        """Detect if agent is stuck in a loop"""
        current_action = f"{action}:{parameter}"
        self.last_actions.append(current_action)
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)
        if len(self.last_actions) >= self.action_limit:
            recent = self.last_actions[-self.action_limit:]
            if len(set(recent)) == 1:
                print(f"⚠️ Loop detected! Same action repeated {self.action_limit} times: {current_action}")
                return True
        return False

    def fallback_parse(self, text: str, user_goal: str) -> dict:
        """Fallback parser when JSON parsing fails"""
        text_lower = text.lower()
        
        if "goto" in text_lower or "navigate" in text_lower:
            url_pattern = r'https?://[^\s\'"<>)}\]]+'
            urls = re.findall(url_pattern, text)
            if urls:
                print(f"✓ Fallback: Found GOTO action with URL: {urls[0]}")
                return {
                    "thought": "Navigating to website",
                    "action": "GOTO",
                    "parameter": urls[0],
                    "completed": False
                }
        
        if "click" in text_lower:
            patterns = [
                r'click["\s]+([A-Z][a-zA-Z\s]+)["\s,}]',
                r'parameter["\s:]+([A-Z][a-zA-Z\s]+)["\s,}]',
                r'"([A-Z][a-zA-Z\s]{2,15})"',
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    target = match.group(1).strip()
                    if target.lower() not in ['goto', 'click', 'type', 'scroll', 'done', 'action', 'parameter']:
                        print(f"✓ Fallback: Found CLICK action with target: {target}")
                        return {
                            "thought": "Clicking on element",
                            "action": "CLICK",
                            "parameter": target,
                            "completed": False
                        }
        
        clickable_keywords = ["downloads", "download", "docs", "documentation", "about", "community"]
        for keyword in clickable_keywords:
            if keyword in user_goal.lower() and keyword.capitalize() in text:
                print(f"✓ Fallback: Inferred CLICK on {keyword.capitalize()} from context")
                return {
                    "thought": f"Clicking on {keyword.capitalize()}",
                    "action": "CLICK",
                    "parameter": keyword.capitalize(),
                    "completed": False
                }
        
        if "done" in text_lower or "complete" in text_lower or "found" in text_lower:
            print(f"✓ Fallback: Task appears complete")
            return {
                "thought": "Task completed",
                "action": "DONE",
                "parameter": "",
                "completed": True
            }
        
        print(f"⚠ Fallback: No clear action found, marking as DONE")
        return {
            "thought": text[:100] if text else "Unable to determine action",
            "action": "DONE",
            "parameter": "",
            "completed": True
        }

    # 文件工具方法保持不变...
    def download_pdf(self, page, link_text: str = "Download"):
        """Automatically detect and download a PDF from the page"""
        try:
            print(f"📥 Attempting to download PDF: {link_text}")
            with page.expect_download(timeout=15000) as download_info:
                page.click(f"text={link_text}")
            download = download_info.value
            filename = download.suggested_filename
            save_path = self.output_dir / filename
            download.save_as(save_path)
            print(f"✅ PDF downloaded successfully: {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"❌ PDF download failed: {e}")
            return None

    def save_text_output(self, text: str, filename: str = "final_interpretation.txt"):
        """Save model output text to a file"""
        try:
            path = self.file_tools.write_text_file(filename, text)
            print(f"📝 Text result saved to: {path}")
        except Exception as e:
            print(f"Failed to write text file: {e}")

    def perform_ocr_on_screenshot(self, screenshot_path: str):
        """Perform OCR on a screenshot"""
        try:
            result = self.file_tools.process_screenshot_ocr(screenshot_path)
            if result.get("success"):
                print(f"🔍 OCR result saved to: {result['result_file']}")
            else:
                print("⚠ Screenshot OCR failed:", result.get("error"))
        except Exception as e:
            print(f"OCR processing error: {e}")
        
    def check_and_process_pdfs(self):
        """Check for any newly downloaded PDFs and extract text with OCR"""
        pdf_files = list(self.output_dir.glob("*.pdf"))
        if not pdf_files:
            return
        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
        print(f"[PDF detected] {latest_pdf.name} - extracting text...")
        result = self.file_tools.pdf_extract_text_with_ocr(str(latest_pdf))
        if result.get("success"):
            print(f"[PDF text saved to] {result['file_path']}")
        else:
            print("[PDF extraction failed]", result.get("error"))

        print(f"[Extracting images from] {latest_pdf.name} ...")
        image_result = self.file_tools.extract_images_from_pdf(str(latest_pdf))
        if image_result.get("success"):
            print(f"Extracted {len(image_result['image_paths'])} images successfully.")
        else:
            print("Image extraction failed:", image_result.get("error"))
    
    def execute_action(self, page, action_plan: dict) -> bool:
        """Execute the planned action"""
        action = action_plan.get("action", "").upper().strip()
        parameter = action_plan.get("parameter", "").strip()
        
        print(f"\n{'='*50}")
        print(f"Executing: {action}")
        if parameter:
            print(f"Parameter: {parameter}")
        print(f"{'='*50}")
        
        try:
            if action == "GOTO":
                if not parameter:
                    print("❌ Error: GOTO requires a URL parameter")
                    return False
                
                if not parameter.startswith(('http://', 'https://')):
                    parameter = 'https://' + parameter
                
                print(f"🌐 Navigating to: {parameter}")
                
                try:
                    response = page.goto(parameter, wait_until="domcontentloaded", timeout=30000)
                    
                    if response and response.ok:
                        print(f"✓ Page loaded (status: {response.status})")
                    else:
                        print(f"⚠ Page loaded with status: {response.status if response else 'unknown'}")
                    
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except:
                        print("⏳ Network still active, continuing...")
                    
                    time.sleep(3)
                    print(f"✓ Navigation complete")
                    return True
                    
                except Exception as e:
                    print(f"❌ Navigation error: {e}")
                    return False
                
            elif action == "CLICK":
                if not parameter:
                    print("❌ Error: CLICK requires an element parameter")
                    return False
                
                try:
                    if "download" in parameter.lower() or parameter.lower().endswith(".pdf"):
                        pdf_path = self.download_pdf(page, link_text=parameter)
                        if pdf_path:
                            self.check_and_process_pdfs()
                            return True
                except Exception as e:
                    print(f"Download heuristic attempt failed: {e}")
                
                print(f"🔍 Looking for element: {parameter}")
                
                selectors = [
                    f"text={parameter}",
                    f"a:has-text('{parameter}')",
                    f"button:has-text('{parameter}')",
                    f"//*[contains(text(), '{parameter}')]",
                    f"[aria-label*='{parameter}' i]",
                    f"a:has-text('{parameter.lower()}')",
                ]
                
                clicked = False
                for selector in selectors:
                    try:
                        element = page.locator(selector).first
                        count = element.count()
                        if count > 0:
                            print(f"  Found with selector: {selector}")
                            element.scroll_into_view_if_needed()
                            time.sleep(0.5)
                            element.click(timeout=5000)
                            
                            try:
                                page.wait_for_load_state("networkidle", timeout=5000)
                            except:
                                pass
                            
                            time.sleep(2)
                            clicked = True
                            print(f"✓ Successfully clicked: {parameter}")
                            break
                    except Exception as e:
                        continue
                
                if not clicked:
                    print(f"❌ Could not find clickable element: {parameter}")
                    return False
                
                return True
                
            elif action == "TYPE":
                if not parameter:
                    print("❌ Error: TYPE requires text parameter")
                    return False
                
                try:
                    page.keyboard.type(parameter)
                    time.sleep(1)
                    print(f"✓ Typed: {parameter}")
                    return True
                except Exception as e:
                    print(f"❌ Error typing: {e}")
                    return False
                
            elif action == "SCROLL":
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                time.sleep(1)
                print("✓ Scrolled down")
                return True
                
            elif action == "DONE":
                print("✓ Task marked as complete")
                return False
                
            else:
                print(f"❌ Unknown action: {action}")
                return False
                
        except Exception as e:
            print(f"❌ Error executing action: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_task(self, user_goal: str, max_steps: int = 5):
        """Run a complete task"""
        print(f"\n{'='*60}")
        print(f"🎯 Task: {user_goal}")
        print(f"{'='*60}\n")
        
        with sync_playwright() as p:
            print("🚀 Launching browser...")
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )
            
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = context.new_page()
            
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, user_goal)
            
            if urls:
                start_url = urls[0]
                print(f"🌐 Starting at: {start_url}\n")
                try:
                    page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_load_state("networkidle", timeout=10000)
                    time.sleep(3)
                except Exception as e:
                    print(f"⚠ Initial navigation failed: {e}")
                    page.goto("about:blank")
            else:
                page.goto("about:blank")
            
            step = 0
            self.last_actions = []
            
            while step < max_steps:
                print(f"\n{'─'*60}")
                print(f"📍 Step {step + 1}/{max_steps}")
                print(f"{'─'*60}")
                
                screenshot_path = self.capture_screenshot(page, step)

                try:
                    self.perform_ocr_on_screenshot(screenshot_path)
                except Exception as e:
                    print(f"OCR on screenshot failed: {e}")

                dom_summary = self.get_dom_summary(page)
                current_url = page.url
                
                print(f"📄 Current URL: {current_url}")
                
                action_plan = self.analyze_page(screenshot_path, dom_summary, user_goal, current_url)
                
                self.history.append({
                    "step": step,
                    "screenshot": screenshot_path,
                    "dom_summary": dom_summary,
                    "thought": action_plan.get("thought", ""),
                    "action": action_plan.get("action", ""),
                    "parameter": action_plan.get("parameter", ""),
                    "url": current_url
                })
                
                if action_plan.get("completed", False) or action_plan.get("action", "").upper() == "DONE":
                    print("\n✅ Task completed!")
                    break
                
                should_continue = self.execute_action(page, action_plan)

                try:
                    self.check_and_process_pdfs()
                except Exception as e:
                    print(f"PDF post-processing failed: {e}")

                if not should_continue:
                    print("\n⏹ Stopping execution")
                    break
                
                step += 1
            
            print("\n📸 Taking final screenshot...")
            print(f"📄 Final URL: {page.url}")
            
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass
            time.sleep(2)
            
            final_screenshot = self.output_dir / "final_screenshot.png"
            try:
                page.screenshot(path=str(final_screenshot), full_page=True)
                print(f"✓ Final screenshot saved: {final_screenshot}")
            except Exception as e:
                print(f"Failed to take final screenshot: {e}")

            try:
                self.save_text_output("Task completed. Execution artifacts are stored in the output folder.")
            except Exception as e:
                print(f"Failed to save final text output: {e}")
            
            time.sleep(1)
            context.close()
            browser.close()
            print("🔒 Browser closed")
        
        self.save_log()
        print(f"\n✅ Complete! Results in: {self.output_dir.absolute()}")
    
    def save_log(self):
        """Save execution history to file"""
        log_path = self.output_dir / "execution_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"📝 Log saved: {log_path}")


if __name__ == "__main__":
    agent = WebAgent(model_name="Qwen/Qwen2-VL-2B-Instruct")
    task = "Go to https://arxiv.org/pdf/2505.09388 and find information about Table1"
    agent.run_task(task, max_steps=3)