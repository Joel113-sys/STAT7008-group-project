"""
VLLM + Playwright Web Agent Demo - Improved Version
A simplified demonstration of autonomous web browsing with vision-language model

Improvements:
- Loop detection to prevent infinite repeated actions
- Better DOM summary with version extraction
- State-aware prompts
- Enhanced screenshot capture

Requirements:
pip install playwright transformers torch pillow requests
playwright install chromium

Hardware: NVIDIA 4080Ti (tested)
Model: Qwen2-VL-2B-Instruct (lightweight, ~5GB VRAM)
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from playwright.sync_api import sync_playwright, Page, Browser
import time


class WebAgent:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        """Initialize the web agent with a lightweight VLLM"""
        print(f"Loading model: {model_name}")
        
        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Create output directory
        self.output_dir = Path("web_agent_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Session history
        self.history = []
        
        # Loop detection
        self.last_actions = []
        self.action_limit = 3
        
        # 🆕 NEW: Task step tracking
        self.task_steps = []  # List of step descriptions
        self.current_step_index = 0  # Which step are we on?
        
        print("Model loaded successfully!")
    
    def capture_screenshot(self, page, step: int) -> str:
        """Capture and save screenshot - improved to capture more content"""
        screenshot_path = self.output_dir / f"step_{step}_screenshot.png"
        
        try:
            # Wait for page to be fully loaded
            page.wait_for_load_state("networkidle", timeout=10000)
            time.sleep(2)
            
            # Scroll to top first
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)
            
            # Take screenshot
            page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            # Try to take screenshot anyway
            page.screenshot(path=str(screenshot_path), full_page=False)
        
        return str(screenshot_path)
    
    def get_dom_summary(self, page: Page) -> str:
        """Get simplified DOM summary - improved with version detection"""
        try:
            title = page.title()
            url = page.url
            
            # Get main interactive elements
            buttons = page.locator("button").count()
            links = page.locator("a").count()
            inputs = page.locator("input").count()
            
            # NEW: Extract visible text to detect version information
            visible_text = ""
            try:
                # Extract page text
                body_text = page.locator("body").inner_text()
                
                # Find version patterns
                version_pattern = r'Python\s+\d+\.\d+\.\d+'
                versions = re.findall(version_pattern, body_text)
                
                if versions:
                    unique_versions = list(set(versions))
                    visible_text = f"\n- Found versions on page: {', '.join(unique_versions[:5])}"
                
                # Find "Latest:" related content
                if "Latest:" in body_text or "latest" in body_text.lower():
                    latest_match = re.search(r'Latest:?\s*(Python\s+\d+\.\d+\.\d+)', body_text, re.IGNORECASE)
                    if latest_match:
                        visible_text += f"\n- Latest version shown: {latest_match.group(1)}"
                
                # Check for download page indicators
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
        # Mapping of Chinese punctuation to English equivalents
        replacements = {
            # Quotes
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '´': "'",
            '`': "'",
            
            # Commas and periods
            '，': ',',
            '。': '.',
            '、': ',',
            
            # Colons and semicolons
            '：': ':',
            '；': ';',
            
            # Brackets
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '「': '"',
            '」': '"',
            '『': '"',
            '』': '"',
            
            # Other punctuation
            '！': '!',
            '？': '?',
            '—': '-',
            '–': '-',
            '…': '...',
            '·': '.',
            
            # Spaces
            '\u3000': ' ',  # Full-width space
            '\xa0': ' ',    # Non-breaking space
        }
        
        for chinese, english in replacements.items():
            text = text.replace(chinese, english)
        
        return text
    
    def detect_loop(self, action: str, parameter: str) -> bool:
        """Detect if agent is stuck in a loop"""
        current_action = f"{action}:{parameter}"
        
        # Record current action
        self.last_actions.append(current_action)
        
        # Keep only last 5 actions
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)
        
        # Check if last N actions are identical
        if len(self.last_actions) >= self.action_limit:
            recent = self.last_actions[-self.action_limit:]
            if len(set(recent)) == 1:  # All actions are the same
                print(f"⚠️ Loop detected! Same action repeated {self.action_limit} times: {current_action}")
                return True
        
        return False
        
    def get_interactive_elements(self, page: Page) -> dict:
        """Extract all clickable elements with their text and selectors"""
        try:
            elements = {
                "links": [],
                "buttons": [],
                "inputs": []
            }
            
            # Extract links
            links = page.locator("a[href]").all()
            seen_texts = set()  # 🆕 去重
            
            for i, link in enumerate(links):
                try:
                    text = link.inner_text().strip()
                    href = link.get_attribute("href")
                    
                    # 🆕 过滤条件改进
                    if (text and 
                        len(text) > 0 and 
                        len(text) < 100 and 
                        text not in seen_texts and  # 去重
                        href):  # 必须有 href
                        
                        elements["links"].append({
                            "index": len(elements["links"]),  # 🆕 使用实际索引
                            "text": text,
                            "href": href
                        })
                        seen_texts.add(text)
                        
                        # 🆕 限制数量，避免太多
                        if len(elements["links"]) >= 30:
                            break
                except:
                    continue
            
            # Extract buttons
            buttons = page.locator("button").all()
            for i, button in enumerate(buttons[:10]):
                try:
                    text = button.inner_text().strip()
                    if text:
                        elements["buttons"].append({
                            "index": i,
                            "text": text
                        })
                except:
                    continue
            
            return elements
        except Exception as e:
            print(f"Error extracting elements: {e}")
            return {"links": [], "buttons": [], "inputs": []}

    def format_elements_for_prompt(self, elements: dict) -> str:
        """Format extracted elements for the prompt"""
        output = "\nClickable elements on this page:\n"
        
        if elements["links"]:
            output += "\nLinks:\n"
            for link in elements["links"][:15]:  # Show first 15
                output += f"  [{link['index']}] {link['text']}\n"
        
        if elements["buttons"]:
            output += "\nButtons:\n"
            for btn in elements["buttons"][:10]:
                output += f"  [{btn['index']}] {btn['text']}\n"
        
        if not elements["links"] and not elements["buttons"]:
            output += "  (No clickable elements found)\n"
        
        return output

    def parse_task_steps(self, user_goal: str):
        """🆕 Extract numbered steps from task description"""
        step_pattern = r'Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|$)'
        matches = re.findall(step_pattern, user_goal, re.DOTALL | re.IGNORECASE)
        
        self.task_steps = [step.strip() for step in matches]
        self.current_step_index = 0
        
        if self.task_steps:
            print(f"\n📋 Parsed {len(self.task_steps)} task steps:")
            for i, step in enumerate(self.task_steps):
                print(f"   {i+1}. {step}")
            print()
        
        return self.task_steps

    def get_current_step_instruction(self) -> str:
        """🆕 Get the current step instruction"""
        if not self.task_steps:
            return ""
        
        if self.current_step_index < len(self.task_steps):
            return self.task_steps[self.current_step_index]
        
        return "All steps completed"
    
    def advance_to_next_step(self):
        """🆕 Move to next step after successful action"""
        if self.current_step_index < len(self.task_steps):
            self.current_step_index += 1
            print(f"\n✅ Completed step {self.current_step_index}/{len(self.task_steps)}")
            if self.current_step_index < len(self.task_steps):
                print(f"📌 Next step: {self.task_steps[self.current_step_index]}\n")

    def analyze_page(self, screenshot_path: str, dom_summary: str, user_goal: str, current_url: str, elements: dict) -> dict:
        """Analyze page - improved with step-by-step guidance"""
        print("\nAnalyzing page with VLLM...")
        
        # Format elements for prompt
        elements_text = self.format_elements_for_prompt(elements)
        
        # 🆕 Get current step instruction
        current_step = self.get_current_step_instruction()
        step_info = ""
        if current_step:
            step_info = f"""
            CURRENT STEP ({self.current_step_index + 1}/{len(self.task_steps)}):
            {current_step}

            YOU MUST COMPLETE THIS STEP BEFORE MOVING TO THE NEXT ONE.
            """
                    
            prompt = f"""You are a web browsing assistant following a step-by-step task.

            User Goal: {user_goal}

            {step_info}

            Current Page Info:
            {dom_summary}

            Current URL: {current_url}

            {elements_text}

            ⚠️ CRITICAL INSTRUCTIONS:
            1. Focus ONLY on the current step: "{current_step}"
            2. Find the EXACT element that matches the text in the current step
            3. The element text must EXACTLY match what's in the step (e.g., if step says "Community", find "Community" link)
            4. Ignore all other elements, even if they look important
            5. Use the element's INDEX number from the list above

            For CLICK actions:
            - Find the EXACT text match in the elements list
            - Use the corresponding INDEX number
            - Example: Step says "Click Community" → Find "Community" in list → Use its index

            Available actions:
            1. GOTO - Navigate to a URL
            2. CLICK - Click an element (use index number of EXACT text match)
            3. SCROLL - Scroll down
            4. DONE - Current step complete (ONLY after clicking the correct element)

            Set "completed": true ONLY when you have clicked the EXACT element from the current step.

            Respond with ONLY valid JSON:
            {{
                "thought": "Found '{current_step}' at index X, clicking it",
                "action": "CLICK",
                "parameter": "X",
                "completed": false
            }}

            Examples:
            - Step: "Click on Community link"
            → {{"thought": "Current step requires 'Community', found at index 7", "action": "CLICK", "parameter": "7", "completed": false}}
            
            - Step: "Click on Python FAQs link"
            → {{"thought": "Current step requires 'Python FAQs', found at index 15", "action": "CLICK", "parameter": "15", "completed": false}}

            Your JSON response:"""
        
        # Load and encode image
        image = Image.open(screenshot_path)
        
        # Prepare messages for Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Raw model response:\n{output_text}\n")
        
        # Normalize all punctuation to ASCII
        output_text = self.normalize_to_ascii(output_text)
        print(f"After normalization:\n{output_text}\n")
        
        # Try to parse JSON
        try:
            # Extract JSON block
            start_idx = output_text.find("{")
            end_idx = output_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = output_text[start_idx:end_idx]
                
                # Additional cleanup
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                
                print(f"Cleaned JSON string:\n{json_str}\n")
                
                try:
                    action_plan = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parse failed: {e}")
                    print(f"Failed JSON: {json_str}")
                    raise
                
                # Process action field
                if "action" in action_plan:
                    action_str = str(action_plan["action"]).strip()
                    
                    # Handle "GOTO 'url'" format
                    if " " in action_str:
                        parts = action_str.split(None, 1)
                        action_plan["action"] = parts[0].upper()
                        
                        if len(parts) > 1 and "parameter" not in action_plan:
                            param_str = parts[1].strip().strip("'\"")
                            action_plan["parameter"] = param_str
                            print(f"Extracted parameter from action: {param_str}")
                    else:
                        action_plan["action"] = action_str.upper()
                
                # Ensure parameter field exists
                if "parameter" not in action_plan:
                    action_plan["parameter"] = ""
                
                # NEW: Loop detection
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
            
            # Fallback: Extract information using patterns
            action_plan = self.fallback_parse(output_text, user_goal)
            
            # Loop detection for fallback too
            if self.detect_loop(action_plan["action"], action_plan["parameter"]):
                return {
                    "thought": "Loop detected in fallback",
                    "action": "DONE",
                    "parameter": "",
                    "completed": True
                }
            
            return action_plan
    
    def fallback_parse(self, text: str, user_goal: str) -> dict:
        """Fallback parser when JSON parsing fails"""
        text_lower = text.lower()
        
        # Pattern 1: Look for GOTO with URL
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
        
        # Pattern 2: Look for CLICK action
        if "click" in text_lower:
            # Try multiple patterns to extract click target
            patterns = [
                r'click["\s]+([A-Z][a-zA-Z\s]+)["\s,}]',  # "click Downloads"
                r'parameter["\s:]+([A-Z][a-zA-Z\s]+)["\s,}]',  # "parameter": "Downloads"
                r'"([A-Z][a-zA-Z\s]{2,15})"',  # Any capitalized word in quotes
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    target = match.group(1).strip()
                    # Filter out common false positives
                    if target.lower() not in ['goto', 'click', 'type', 'scroll', 'done', 'action', 'parameter']:
                        print(f"✓ Fallback: Found CLICK action with target: {target}")
                        return {
                            "thought": "Clicking on element",
                            "action": "CLICK",
                            "parameter": target,
                            "completed": False
                        }
        
        # Pattern 3: Look for common clickable elements in the user goal
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
        
        # Pattern 4: Check if task seems complete
        if "done" in text_lower or "complete" in text_lower or "found" in text_lower:
            print(f"✓ Fallback: Task appears complete")
            return {
                "thought": "Task completed",
                "action": "DONE",
                "parameter": "",
                "completed": True
            }
        
        # Ultimate fallback: DONE
        print(f"⚠ Fallback: No clear action found, marking as DONE")
        return {
            "thought": text[:100] if text else "Unable to determine action",
            "action": "DONE",
            "parameter": "",
            "completed": True
        }
    
    def execute_action(self, page, action_plan: dict, elements: dict = None) -> bool:
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
                
                # Ensure URL format
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
                
            if action == "CLICK":
                if not parameter:
                    print("❌ Error: CLICK requires a parameter")
                    return False
                
                clicked = False
                
                # Strategy 1: Try clicking by index
                if parameter.isdigit() and elements:
                    index = int(parameter)
                    
                    # Try links first
                    if index < len(elements.get("links", [])):
                        try:
                            link = elements["links"][index]
                            print(f"🔍 Clicking link [{index}]: {link['text']}")
                            
                            # Use href to find element
                            selector = f"a[href='{link['href']}']"
                            element = page.locator(selector).first
                            
                            if element.count() > 0:
                                element.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                element.click(timeout=5000)
                                clicked = True
                                print(f"✓ Clicked link: {link['text']}")
                        except Exception as e:
                            print(f"⚠ Failed to click by index: {e}")
                
                # Strategy 2: Try clicking by text (fallback)
                if not clicked:
                    print(f"🔍 Looking for element with text: {parameter}")
                    
                    # 🆕 先打印页面上所有匹配的元素
                    try:
                        all_matches = page.locator(f"text={parameter}").all()
                        print(f"   Found {len(all_matches)} elements containing '{parameter}'")
                        
                        for i, match in enumerate(all_matches[:5]):
                            try:
                                text = match.inner_text()[:50]
                                print(f"   Match {i}: {text}")
                            except:
                                pass
                    except Exception as e:
                        print(f"   Could not list matches: {e}")
                    
                    selectors = [
                        f"a:has-text('{parameter}'):visible",
                        f"button:has-text('{parameter}'):visible",
                        f"text={parameter}",
                        f"a:text-is('{parameter}')",
                        f"//*[normalize-space(text())='{parameter}']",
                    ]
                    
                    for selector in selectors:
                        try:
                            print(f"   Trying selector: {selector}")  # 🆕 调试输出
                            element = page.locator(selector).first
                            if element.count() > 0:
                                print(f"  ✓ Found with selector: {selector}")
                                element.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                element.click(timeout=5000, force=True)  # 🆕 添加 force=True
                                clicked = True
                                print(f"✓ Clicked: {parameter}")
                                break
                            else:
                                print(f"   ✗ No match for: {selector}")  # 🆕 调试输出
                        except Exception as e:
                            print(f"   ✗ Error with {selector}: {e}")  # 🆕 改进错误信息
                            continue
                
                if not clicked:
                    print(f"❌ Could not find clickable element: {parameter}")
                    print("💡 Available elements (first 10):")
                    if elements:
                        for i, link in enumerate(elements.get("links", [])[:10]):  # 显示更多
                            print(f"   [{i}] {link['text'][:60]}")  # 截断长文本
                    return False
                
                # Wait for page to load
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass
                time.sleep(2)
                
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
    
    def run_task(self, user_goal: str, max_steps: int = 10):
        """Run a complete task"""
        print(f"\n{'='*60}")
        print(f"🎯 Task: {user_goal}")
        print(f"{'='*60}\n")
        
        self.parse_task_steps(user_goal)
        
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
                # 🆕 检查是否所有步骤已完成
                if self.current_step_index >= len(self.task_steps):
                    print("\n🎉 All task steps completed!")
                    break
                
                print(f"\n{'─'*60}")
                print(f"📍 Step {step + 1}/{max_steps}")
                print(f"📌 Current task step: [{self.current_step_index + 1}/{len(self.task_steps)}] {self.get_current_step_instruction()}")
                print(f"{'─'*60}")
                
                screenshot_path = self.capture_screenshot(page, step)
                dom_summary = self.get_dom_summary(page)
                current_url = page.url
                
                elements = self.get_interactive_elements(page)
                
                print(f"📄 Current URL: {current_url}")
                print(f"🔗 Found {len(elements['links'])} links, {len(elements['buttons'])} buttons")
                
                action_plan = self.analyze_page(
                    screenshot_path, 
                    dom_summary, 
                    user_goal, 
                    current_url,
                    elements
                )
                
                self.history.append({
                    "step": step,
                    "task_step": f"{self.current_step_index + 1}/{len(self.task_steps)}: {self.get_current_step_instruction()}",
                    "screenshot": screenshot_path,
                    "dom_summary": dom_summary,
                    "thought": action_plan.get("thought", ""),
                    "action": action_plan.get("action", ""),
                    "parameter": action_plan.get("parameter", ""),
                    "url": current_url
                })
                
                # 🆕 修改：处理 DONE 动作
                if action_plan.get("action", "").upper() == "DONE":
                    print("\n✅ DONE action received")
                    self.advance_to_next_step()
                    step += 1
                    continue
                
                # 🆕 修改：执行动作
                action_success = self.execute_action(page, action_plan, elements)
                
                if not action_success:
                    print("\n❌ Action execution failed")
                    # 🆕 即使失败也尝试继续（可选：也可以break）
                    step += 1
                    continue
                
                # 🆕 关键修改：成功执行动作后，立即前进到下一步
                print(f"\n✅ Action executed successfully")
                self.advance_to_next_step()
                
                step += 1
            
            # 最终截图
            print("\n📸 Taking final screenshot...")
            print(f"📄 Final URL: {page.url}")
            
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass
            time.sleep(2)
            
            final_screenshot = self.output_dir / "final_screenshot.png"
            page.screenshot(path=str(final_screenshot), full_page=True)
            print(f"✓ Final screenshot saved: {final_screenshot}")
            
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


def process_vision_info(messages):
    """Helper function to extract images from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
    
    return image_inputs if image_inputs else None, video_inputs if video_inputs else None


if __name__ == "__main__":
    agent = WebAgent()
    task = """Go to https://www.python.org

    Step 1: Click on "Community" link 
    Step 2: Click on "Python FAQs" link

    """
    agent.run_task(task, max_steps=5)


    # task = """Go to https://arxiv.org/

    # Step 1: Click on "Astrophysics" link 
    # Step 2: Click on "current month's" link
    # Step 3: Click on the first listed link

    # IMPORTANT: Ignore navigation menu items
    # """
# task = "Go to https://www.python.org and click through any Python version hyperlink, and then click another hyperlink in the poped out page."
#  https://arxiv.org/
# task = "Go to https://www.python.org and find information about the latest Python version"