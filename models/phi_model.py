from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class PhiVLLM(GenericVLLM):
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False
        )
