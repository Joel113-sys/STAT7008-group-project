from transformers import LlavaForConditionalGeneration, AutoProcessor
from models.generic_model import GenericVLLM


class LLaVAVLLM(GenericVLLM):
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        super().__init__(
            model_name=model_name,
            model_cls=LlavaForConditionalGeneration,
            processor_cls=AutoProcessor,
            use_vision=True
        )
