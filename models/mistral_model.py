from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class MistralVLLM(GenericVLLM):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False
        )
