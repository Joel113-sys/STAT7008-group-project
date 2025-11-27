from transformers import AutoModelForCausalLM, AutoTokenizer
from models.generic_model import GenericVLLM


class MixtralVLLM(GenericVLLM):
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        super().__init__(
            model_name=model_name,
            model_cls=AutoModelForCausalLM,
            processor_cls=AutoTokenizer,
            use_vision=False
        )
