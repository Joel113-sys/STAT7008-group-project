## 7008Project Web Agent & Multi-Model Framework

### Overview
7008Project is a modular framework for autonomous web browsing powered by large language and vision-language models. It supports:
- Multiple interchangeable models (Qwen2-VL, LLaVA, LLaMA, Mistral, Mixtral, DeepSeek, ChatGLM, Phi)
- Browser automation via Playwright: screenshots + DOM summarization
- Iterative model-driven actions (GOTO / CLICK / SCROLL / DONE)
- Automatic fallback (RequestsAgent) when browser launch fails or system libs are missing
- Evaluation of success rate, steps, invalid actions, latency, model load time, peak GPU memory
- Unit tests for action logic (mocked Playwright interactions)

### Directory Structure
```
7008Project/
  main.py               # Entry point: parse args and run WebAgent
  evaluation.py         # Multi-model evaluation, CSV output
  demo.py               # Original simple demo (Qwen)
  demo(1).py            # Enhanced demo using Transformers remote Qwen2-VL model
  agent/
    web_agent.py        # Main WebAgent (browser + fallback logic)
    actions.py          # Separated action implementations: GOTO / CLICK / SCROLL
    fallback_agent.py   # HTTP-only fallback (search + PDF link discovery)
  models/               # Model wrappers / adapters
  utils/                # OCR, vision, file tools
  tasks/tasks.json      # Sample task config (optional)
  tests/                # Pytest unit tests (test_web_agent.py)
  requirements.txt      # Pinned environment dependencies
  web_agent_output/     # Generated screenshots and execution logs
```

### Key Features
- Dynamic model selection via `MODEL_MAP` (in `main.py` / `evaluation.py`).
- Vision support: Qwen2-VL / LLaVA process screenshots.
- Robust output normalization: JSON cleaning, loop detection, completion validation.
- Fallback mode: gracefully degrades to HTTP-only search if Playwright fails.
- Evaluation metrics: success flag, step count, invalid actions, load & run latency, peak GPU memory.
- Testability: action layer split for isolated unit testing without a real browser.

---
## Environment Setup

### 1. Python Version
Recommend Python 3.11 (3.10 also works). Current environment uses 3.11.

### 2. Create Virtual Environment
```bash
conda create -n proj7008 python=3.11 -y
conda activate proj7008
# or
python -m venv .venv && source .venv/bin/activate
```

### 3. Install Dependencies
Pinned versions in `requirements.txt` for reproducibility:
```bash
pip install -r requirements.txt
```
Excerpt:
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

### 4. Install Browsers (Optional)
If Chromium fails due to missing system libraries (e.g. `libatk-1.0.so.0`) and you lack sudo, the agent will auto-fallback. With proper privileges:
```bash
python -m playwright install chromium
python -m playwright install firefox    # Sometimes fewer system deps
```

### 5. Quick Model Pull Test
```bash
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('Tokenizer OK')
PY
```

---
## Running `main.py`

### Arguments
`main.py` exposes:
- `--model`: one of (qwen, llama, llava, mistral, mixtral, deepseek, chatglm, phi)
- `--task`: natural language instruction (URL optional)
- `--max_steps`: maximum action iterations (default 8)

### Basic Example
```bash
python main.py --model qwen --task "Go to https://www.python.org and find the latest Python version." --max_steps 8
```

### Task Without URL (Auto Search)
```bash
python main.py --task "Find the temperature of Hong Kong on November 21 2025" --max_steps 8
```
If Playwright fails, you will see fallback activation logs.

### Switching Models
```bash
python main.py --model mistral --task "Summarize current Python stable version"
python main.py --model llava --task "Describe the homepage layout of https://www.python.org"
```

### Fallback Trigger
Occurs when Playwright is unavailable or browser launch errors. Log sample: `[Agent] Browser launch failed ... Using fallback.`
Fallback behavior: DuckDuckGo HTML search → PDF link discovery (e.g., Qwen technical report).

---
## Enhanced Demo (`demo(1).py`)
Loads remote `Qwen/Qwen2-VL-2B-Instruct` via Transformers:
```bash
python demo\(1\).py
```
Adds: OCR, PDF download heuristics, image extraction.

---
## Evaluation (`evaluation.py`)

### Purpose
Runs a set of models on the same task; records success, steps, invalid actions, latency, load time, GPU peak memory; writes a CSV.

### Arguments
- `--models`: list of models (default: qwen llava mistral)
- `--task`: evaluation task (required)
- `--csv`: output CSV file (default: `evaluation_results.csv`)

### Example
```bash
python evaluation.py --task "Go to https://www.python.org and find the latest Python version." --models qwen llava mistral --csv eval_run1.csv
```
Produces summary table + `eval_run1.csv`.

### Simple Run (Default Models)
```bash
python evaluation.py --task "Find the temperature of Hong Kong on November 21 2025"
```

---
## Unit Tests
Run tests:
```bash
pytest -q
```
Coverage:
- GOTO: missing URL / same URL / auto https
- CLICK: element found vs not found
- SCROLL: success and exception path
Playwright import is mocked → no browser required.

---
## Fallback Mode (RequestsAgent)
File: `agent/fallback_agent.py`
Logic:
- Builds enriched search query (adds qwen / pdf keywords)
- Uses DuckDuckGo HTML results
- Extracts first matching PDF link
- Provides placeholder Figure 1 analysis
Suggested enhancements:
- Integrate `pdfminer.six` / `pypdf` for text extraction
- Feed PDF images + OCR into vision model for richer interpretation

---
## Supported Models (Framework Adapters)

### Qwen2-VL-2B-Instruct
Multimodal (text + image + video). Strong page screenshot understanding. JSON normalization + completion checks in `models/qwen_model.py`.

### LLaVA
Vision-language dialogue model. Good for screenshot interpretation and element guidance.

### LLaMA / Mistral / Mixtral
High-quality open weight text models. Efficient for planning / reasoning when vision not required.

### DeepSeek / ChatGLM / Phi
Diverse lightweight or instruction-tuned models. Phi excels in efficiency for smaller GPU memory footprints.

### Selection Guidance
- Need vision comprehension → Qwen2-VL or LLaVA.
- Pure search / reasoning → Mistral / Mixtral.
- Low resource → Phi or Qwen2-VL-2B.

---
## FAQ
| Issue | Resolution |
|-------|------------|
| Chromium missing libs | Use fallback or install system dependencies / switch to Firefox |
| Model outputs DONE too early | Refine `--task`, increase `--max_steps`, inspect normalization in model wrapper |
| JSON parse errors | Check RAW OUTPUT logs; adjust regex cleaning in `qwen_model.py` |
| Completion incorrect | Extend `_goal_completed` conditions (add keywords/regex) |
| Out of GPU memory | Use smaller models (2B / Phi), lower `max_steps`, reduce generation length |

---
## Roadmap / Extensions
- Add `TYPE` action (present in demo; move to `actions.py`).
- Serve as API (FastAPI / lightweight server) with persistent model instances.
- Combine OCR text + screenshot into richer multimodal prompts.
- Deep PDF inspection (vector search / structural parsing / figure caption extraction).
- Stronger evaluation success criteria (numeric verification, pattern matching).

---
## Quick Start
```bash
conda activate proj7008
pip install -r requirements.txt
python main.py --model qwen --task "Go to https://www.python.org and find the latest Python version." --max_steps 8
python evaluation.py --task "Find the temperature of Hong Kong on November 21 2025" --models qwen mistral llava
pytest -q
```

---
## License Notice
No explicit license currently included. Ensure compliance with each third-party model's license (e.g., Apache-2.0). Add a LICENSE file before redistribution or commercial use.

---
If you need an English + Chinese dual README or want to enhance PDF analysis, let me know the next step. Enjoy!
