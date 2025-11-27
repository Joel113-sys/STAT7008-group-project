from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class ChatGLMVLLM(GenericVLLM):
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False
        )
