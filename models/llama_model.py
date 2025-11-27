from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class LLaMAVLLM(GenericVLLM):
    def __init__(self, model_name="meta-llama/Llama-3-8B-Instruct"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False    # llama 是纯文本
        )
