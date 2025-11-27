import re
import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from utils.vision import process_vision_info
from models.base_model import BaseVLLM


class QwenVLLM(BaseVLLM):
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        print(f"[Model] 加载 Qwen 模型: {model_name}")
        # 使用 dtype 而非已弃用 torch_dtype
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device
        print("[Model] Qwen 加载完毕")

    def analyze(self, screenshot_path=None, dom_summary="", user_goal="", current_url=""):
        """
        支持有/无截图两种模式：
        - 有 screenshot_path：图像+文本多模态
        - 无 screenshot_path：纯文本
        返回结构化 JSON，异常时回退 DONE。
        """

        prompt = f"""
你是一个浏览器自动化 agent。

用户目标：{user_goal}

当前页面信息：
{dom_summary}

当前 URL：{current_url}

请根据（可能存在的）截图和信息决定下一步动作。只输出 JSON。

JSON 格式如下：
{{
    "thought": "",
    "action": "GOTO / CLICK / SCROLL / DONE",
    "parameter": "",
    "completed": false
}}
""".strip()

        # 构造消息（是否包含图片）
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        image_obj = None
        if screenshot_path:
            try:
                image_obj = Image.open(screenshot_path)
                messages[0]["content"].append({"type": "image", "image": image_obj})
            except Exception as e:
                print(f"[WARN] 打开截图失败，使用纯文本模式: {e}")

        messages[0]["content"].append({"type": "text", "text": prompt})

        # 纯文本快速路径（无图片且处理器支持）
        if image_obj is None:
            # 直接使用聊天模板生成文本输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            # 多模态路径
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # 避免空视频列表触发 IndexError：仅在非空时传入
            processor_kwargs = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt"
            }
            if video_inputs:  # 若未来支持视频
                processor_kwargs["videos"] = video_inputs

            inputs = self.processor(**processor_kwargs).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2
            )

        # 仅取新生成 token
        gen_ids = out[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # 提取第一个 JSON 块
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            print(f"[WARN] 未找到 JSON，原始输出: {output_text}")
            return {"thought": "fail", "action": "DONE", "parameter": "", "completed": True}

        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
            # 补全缺失字段
            for k in ["thought", "action", "parameter", "completed"]:
                if k not in parsed:
                    parsed[k] = "" if k != "completed" else False
            return parsed
        except Exception as e:
            print(f"[WARN] JSON 解析失败: {e} | 原始: {json_str}")
            return {"thought": "fail", "action": "DONE", "parameter": "", "completed": True}