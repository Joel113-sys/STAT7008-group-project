import argparse
from agent.web_agent import WebAgent
from models.qwen_model import QwenVLLM
from models.llama_model import LLaMAVLLM
from models.llava_model import LLaVAVLLM
from models.mistral_model import MistralVLLM
from models.mixtral_model import MixtralVLLM
from models.deepseek_model import DeepSeekVLLM
from models.chatglm_model import ChatGLMVLLM
from models.phi_model import PhiVLLM


MODEL_MAP = {
    "qwen": QwenVLLM,
    "llama": LLaMAVLLM,
    "llava": LLaVAVLLM,
    "mistral": MistralVLLM,
    "mixtral": MixtralVLLM,
    "deepseek": DeepSeekVLLM,
    "chatglm": ChatGLMVLLM,
    "phi": PhiVLLM,
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen",
                        choices=list(MODEL_MAP.keys()))
    parser.add_argument("--task", type=str,
                        default="Go to https://www.python.org and find the latest Python version.")
    parser.add_argument("--max_steps", type=int, default=8)
    args = parser.parse_args()

    VLLM = MODEL_MAP[args.model]
    model = VLLM()

    agent = WebAgent(model)
    agent.run(args.task, max_steps=args.max_steps)
