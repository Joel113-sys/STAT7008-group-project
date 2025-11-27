# models/model_registry.py

from models.qwen_model import QwenModel
from models.llama_model import LlamaModel
from models.deepseek_model import DeepseekModel
from models.mistral_model import MistralModel
from models.phi_model import PhiModel


MODEL_REGISTRY = {
    "qwen": QwenModel,
    "llama": LlamaModel,
    "deepseek": DeepseekModel,
    "mistral": MistralModel,
    "phi": PhiModel,
}


def load_vllm_model(name: str):
    """
    Unified model loader.
    Each model class must implement:
        analyze(screenshot_path, dom_summary, user_goal, current_url)
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[name]()
