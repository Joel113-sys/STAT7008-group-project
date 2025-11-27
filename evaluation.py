import argparse
import time
import csv
import torch

from agent.web_agent import WebAgent
from models.qwen_model import QwenVLLM
from models.llava_model import LLaVAVLLM
from models.mistral_model import MistralVLLM
from models.mixtral_model import MixtralVLLM
from models.deepseek_model import DeepSeekVLLM
from models.chatglm_model import ChatGLMVLLM
from models.phi_model import PhiVLLM


# ---- 模型映射表 ----
MODEL_MAP = {
    "qwen": QwenVLLM,
    "llava": LLaVAVLLM,
    "mistral": MistralVLLM,
    # "mixtral": MixtralVLLM,
    # "deepseek": DeepSeekVLLM,
    # "chatglm": ChatGLMVLLM,
    # "phi": PhiVLLM,
}


def evaluate_model(model_name, task):
    print(f"\n==============================")
    print(f" Evaluating model: {model_name}")
    print(f"==============================")

    # ---- 加载模型 ----
    model_class = MODEL_MAP[model_name]
    load_start = time.time()
    model = model_class()  # 初始化模型
    load_time = time.time() - load_start

    # ---- 创建 WebAgent ----
    agent = WebAgent(model)

    # 清空显存统计
    torch.cuda.reset_peak_memory_stats()

    # ---- 执行任务 ----
    start = time.time()
    summary = agent.run_and_summarize(task)
    elapsed = time.time() - start

    # ---- summary 必须包含这三个字段 ----
    completed = summary.get("completed", False)
    steps = summary.get("steps", 0)
    invalid_actions = summary.get("invalid_actions", 0)

    # ---- 显存 ----
    max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        "model": model_name,
        "success": int(completed),
        "steps": steps,
        "invalid_actions": invalid_actions,
        "latency_sec": round(elapsed, 3),
        "load_time_sec": round(load_time, 3),
        "gpu_mem_mb": round(max_mem, 2)
    }


def write_csv(results, output="evaluation_results.csv"):
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "success", "steps", "invalid_actions",
            "latency_sec", "load_time_sec", "gpu_mem_mb"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📁 Results saved to {output}")


def print_table(results):
    print("\n📊 Evaluation Summary")
    print("---------------------------------------------------------------")
    print(f"{'Model':12} {'Success':7} {'Steps':5} {'Invalid':7} {'Latency(s)':12} {'Load(s)':8} {'GPU(MB)':8}")
    print("---------------------------------------------------------------")

    for r in results:
        print(f"{r['model']:12} {r['success']:7} {r['steps']:5} {r['invalid_actions']:7} "
              f"{r['latency_sec']:12} {r['load_time_sec']:8} {r['gpu_mem_mb']:8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_MAP.keys()))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--csv", type=str, default="evaluation_results.csv")
    args = parser.parse_args()

    results = []

    for model in args.models:
        res = evaluate_model(model, args.task)
        results.append(res)

    print_table(results)
    write_csv(results, args.csv)
