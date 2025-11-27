from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class DeepSeekVLLM(GenericVLLM):
    def __init__(self, model_name="deepseek-ai/deepseek-v2-chat"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False
        )
