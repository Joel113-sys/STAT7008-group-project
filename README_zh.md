## 7008Project Web Agent & Multi-Model Framework

### 概览
7008Project 提供一个基于大语言 / 视觉语言模型的网页自主浏览与任务执行框架，支持：
- 多模型快速切换（Qwen2-VL、LLaVA、LLaMA、Mistral、Mixtral、DeepSeek、ChatGLM、Phi）
- 浏览器自动化（Playwright）+ 截图 + DOM 结构摘要
- 模型决策（GOTO / CLICK / SCROLL / DONE）循环执行
- 失败时的无浏览器回退模式（RequestsAgent）
- 评估脚本统计步骤、无效动作、显存占用与时延
- 单元测试（对核心动作进行 Mock 测试）

### 目录结构
```
7008Project/
  main.py                 # 主入口：解析参数并运行 WebAgent
  evaluation.py           # 执行多模型评估，输出 CSV
  demo.py                 # 原始演示（Qwen 浏览）
  demo(1).py              # 使用 Transformers 加载远程 Qwen2-VL 模型的增强 Demo
  agent/
    web_agent.py          # 主 WebAgent（含浏览器与回退逻辑）
    actions.py            # 独立动作实现：GOTO / CLICK / SCROLL
    fallback_agent.py     # 回退 HTTP 抓取 PDF/信息模式
  models/                 # 统一模型封装（各模型类）
  utils/                  # 工具：OCR、Vision、文件处理
  tasks/tasks.json        # 任务配置示例（如需批量）
  tests/                  # Pytest 单元测试（test_web_agent.py）
  requirements.txt        # 精确版本依赖锁定
  web_agent_output/       # 运行产生的截图与日志
```

### 核心特性
- 动态模型装载：通过 `MODEL_MAP`（main.py / evaluation.py）选择不同模型。
- 视觉支持：Qwen2-VL / LLaVA 类模型可处理图像（截图输入）。
- 自适应策略：模型输出不符合规范时清洗；重复动作检测避免死循环；完成条件验证（例如检测 DOM 中是否出现目标关键词 / 温度 / 版本号）。
- 无浏览器降级：系统库缺失或 Playwright 启动失败时自动启用 `RequestsAgent` 进行静态搜索与 PDF 链接发现。
- 评估指标：成功标记、动作步数、无效动作次数、加载耗时、峰值 GPU 显存。
- 单元测试：对动作层进行 Mock，保证逻辑独立性与可维护性。

---
## 环境准备

### 1. Python 版本
推荐 Python 3.11 或 3.10（当前环境使用 3.11）。

### 2. 创建虚拟环境
```bash
conda create -n proj7008 python=3.11 -y
conda activate proj7008
# 或
python -m venv .venv && source .venv/bin/activate
```

### 3. 安装依赖
`requirements.txt` 已锁定与你当前环境一致版本：
```bash
pip install -r requirements.txt
```
文件核心内容（节选）：
```
torch==2.9.1
torchvision==0.24.1
transformers==4.57.1
accelerate==1.12.0
sentencepiece==0.2.1
huggingface-hub==0.36.0
tqdm==4.67.1
numpy==2.3.5
pillow==12.0.0
requests==2.32.5
playwright==1.56.0
beautifulsoup4==4.12.3
```

### 4. 安装浏览器（可选）
服务器若缺少系统库会导致 Chromium 启动失败（如 `libatk-1.0.so.0`）。你无 sudo 时：
- 直接运行：若失败将自动回退至 RequestsAgent。
- 有权限时：
```bash
python -m playwright install chromium
python -m playwright install firefox   # 可尝试 Firefox 减少依赖
```

### 5. 测试安装
```bash
python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('Tokenizer OK')
PY
```

---
## 运行主程序 `main.py`

### 参数说明
`main.py` 支持：
- `--model`：选择模型（qwen / llama / llava / mistral / mixtral / deepseek / chatglm / phi）
- `--task`：自然语言任务，含或不含 URL。
- `--max_steps`：最大动作执行步数（默认 8）。

### 基本示例
```bash
python main.py --model qwen --task "Go to https://www.python.org and find the latest Python version." --max_steps 8
```

### 无 URL 任务（自动搜索）
```bash
python main.py --task "Find the temperature of Hong Kong on November 21 2025" --max_steps 8
```
Playwright 启动失败时将输出 fallback 提示并使用 HTTP 搜索。

### 切换模型
```bash
python main.py --model mistral --task "Summarize current Python stable version"
python main.py --model llava --task "Describe the homepage layout of https://www.python.org"
```

### 回退模式触发
触发条件：Playwright 未安装或浏览器依赖缺失。日志中会显示：`[Agent] Browser launch failed ... Using fallback.`
回退行为：使用 DuckDuckGo 抓取 HTML，寻找与任务相关的 PDF 链接（示例：Qwen 技术报告）。

---
## 增强 Demo (`demo(1).py`)
使用 HuggingFace 的 `Qwen/Qwen2-VL-2B-Instruct` 远程加载（Transformers）：
```bash
python demo\(1\).py  # 注意括号的 shell 转义
```
功能：更丰富的 OCR、PDF 下载与图片提取（依赖 `utils/file_tools.py`、`ocr_tools.py`）。

---
## 评估脚本 `evaluation.py`

### 功能
对指定模型集合运行同一任务，收集：成功标记、步数、无效动作数、推理时延、加载耗时、峰值 GPU 显存，并写入 `evaluation_results.csv`。

### 参数
- `--models`：要评估的模型列表（默认：qwen llava mistral）。
- `--task`：评估任务（必填）。
- `--csv`：输出 CSV 文件名（默认 evaluation_results.csv）。

### 示例
```bash
python evaluation.py --task "Go to https://www.python.org and find the latest Python version." --models qwen llava mistral --csv eval_run1.csv
```
运行完成后：
```
📊 Evaluation Summary
Model        Success Steps Invalid Latency(s)    Load(s)  GPU(MB)
...
📁 Results saved to eval_run1.csv
```

### 快速批量对比（只指定模型）
```bash
python evaluation.py --task "Find the temperature of Hong Kong on November 21 2025"
```

---
## 单元测试
测试文件：`tests/test_web_agent.py`
```bash
pytest -q
```
覆盖点：
- GOTO 缺参 / 同 URL / 补全 https
- CLICK 找到元素 / 未找到元素
- SCROLL 正常与异常分支

无需真实浏览器：已对 Playwright 进行了 Mock 回退导入。

---
## 回退模式 (RequestsAgent)
文件：`agent/fallback_agent.py`
行为：
- 任务无浏览器时构建增强检索 Query（附加 qwen / pdf 等关键词）
- 使用 DuckDuckGo HTML 解析获取 PDF 链接
- 输出伪分析（Figure 1 占位解释）
扩展建议：
- 引入 `pdfminer.six` / `pypdf` 做 PDF 文本抽取
- 抽取前几页图像并送入视觉模型

---
## 模型介绍（框架支持）

### Qwen2-VL-2B-Instruct
- 类型：多模态（文本 + 图像 + 视频支持，当前使用截图图像分支）。
- 优势：轻量、上下文理解强、视觉描述能力好。
- 使用：`models/qwen_model.py` 中封装了 `analyze` 逻辑（含 JSON 规范化与完成校验）。

### LLaVA 系列
- 视觉语言对话模型，适合页面截图理解与元素定位辅助。
- 若需要可扩展为组件检测：可结合 OCR / 简单图像分割。

### LLaMA / Mistral / Mixtral
- 纯文本开源大模型，适合策略规划与复杂推理。
- 在无视觉需求或回退模式下可降低显存使用。

### DeepSeek / ChatGLM / Phi
- 多样化开源或轻量模型，适合快速指令处理与低资源环境。
- Phi 系列（如 Phi-3）在推理效率上表现较好。

### 选择建议
- 需要截图理解 → Qwen2-VL 或 LLaVA。
- 只做搜索策略 → Mistral / Mixtral。
- 低资源显存限制 → Phi / Qwen2-VL-2B。

---
## 常见问题 (FAQ)
| 问题 | 解决 |
|------|------|
| Chromium 启动报缺库 | 使用回退模式或安装系统依赖 / 换 Firefox |
| 模型第一步就 DONE | 调整 `task` 更明确分解；增加 `max_steps`；检查模型输出清洗逻辑 |
| JSON 解析失败 | 查看 `qwen_model.py` 中 RAW OUTPUT；添加正则兜底清洗 |
| 没有找到目标信息却完成 | 查看 `_goal_completed` 条件，扩展关键字 / 正则 |
| 显存不足 | 使用更小模型（2B / Phi），减少 `max_steps`，降低 `max_new_tokens` |

---
## 扩展路线
- 添加动作：`TYPE`（已有 Demo 中实现，可迁入 actions.py）。
- 引入缓存：模型首次加载后保持实例常驻（例如 FastAPI 服务化）。
- 提升多模态：结合 OCR 文本与截图一起构建更丰富 Prompt。
- PDF 深度分析：将下载的 PDF 前 N 页嵌入向量检索或结构化解析。
- 强化评估：增加成功条件自动验证（抓取目标数值 / 版本号）。

---
## 快速开始总结
```bash
conda activate proj7008
pip install -r requirements.txt
python main.py --model qwen --task "Go to https://www.python.org and find the latest Python version." --max_steps 8
python evaluation.py --task "Find the temperature of Hong Kong on November 21 2025" --models qwen mistral llava
pytest -q
```

---
## License / 使用说明
本项目文件暂未声明具体开源许可证；使用第三方模型需遵守其各自的 License（例如 Apache-2.0 等）。在商用或再分发前请先补充项目 LICENSE。 

---
若需添加新的模型适配或增强 PDF 解析，请告知我下一步需求。祝使用愉快！
