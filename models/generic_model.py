import re
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from utils.vision import process_vision_info
from models.base_model import BaseVLLM


class GenericVLLM(BaseVLLM):
    """
    通用模型封装：
    - 支持文本模型（无视觉）
    - 支持带图像输入的多模态模型
    - 与 QwenVLLM 保持相同的接口 analyze()
    """

    def __init__(
        self,
        model_name: str,
        model_cls=AutoModelForCausalLM,
        processor_cls=AutoProcessor,
        use_vision: bool = False,
    ):
        print(f"[Model] 加载模型: {model_name}")

        self.use_vision = use_vision

        # 加载模型
        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 文本 tokenizer / 多模态 processor 统一处理
        self.processor = processor_cls.from_pretrained(model_name)

        self.device = self.model.device
        print("[Model] 加载完毕")

    def analyze(self, screenshot_path=None, dom_summary="", user_goal="", current_url=""):

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

        # ---- 构造消息 ----
        messages = [{"role": "user", "content": []}]
        image_obj = None

        # ===== 图像路径处理（如果模型支持视觉） =====
        if screenshot_path and self.use_vision:
            try:
                image_obj = Image.open(screenshot_path)
                messages[0]["content"].append({"type": "image", "image": image_obj})
            except Exception as e:
                print(f"[WARN] 图像加载失败: {e}")

        messages[0]["content"].append({"type": "text", "text": prompt})

        # ---- 输入编码 ----
        if not self.use_vision or image_obj is None:
            # 文本-only
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        else:
            # 多模态
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            processor_kwargs = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt"
            }
            if video_inputs:
                processor_kwargs["videos"] = video_inputs

            inputs = self.processor(**processor_kwargs).to(self.device)

        # ---- 生成 ----
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2
            )

        # 取新生成部分
        gen_ids = out[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # ---- JSON 提取 ----
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            print(f"[WARN] 未找到 JSON，原始输出: {output_text}")
            return {"thought": "fail", "action": "DONE", "parameter": "", "completed": True}

        json_str = json_match.group(0)

        try:
            parsed = json.loads(json_str)
            for k in ["thought", "action", "parameter", "completed"]:
                if k not in parsed:
                    parsed[k] = "" if k != "completed" else False
            return parsed
        except Exception as e:
            print(f"[WARN] JSON 解析失败: {e}")
            return {"thought": "fail", "action": "DONE", "parameter": "", "completed": True}
