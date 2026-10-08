---
license: mit
language:
- en
tags:
- function-calling
- tiny-model
- edge-ai
- tool-use
- router
- pytorch
- gguf
pipeline_tag: text-generation
library_name: generic
widget:
- text: "weather in tokyo today"
  example_title: "Web Search Intent"
- text: "tell me a joke"
  example_title: "Abstain (none)"
- text: "who is marie curie"
  example_title: "Person Search"
- text: "latest iphone 16 price in india"
  example_title: "Price Search"
datasets:
- Safeeq/FCtiny
---

<div align="center">

# 🔀 TinyLM-FC — Tiny Function-Calling Router
### ~0.47M Parameters | CPU < 4ms | MIT License

[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-black?logo=github)](https://github.com/Safeeq008/tiny-fc-lm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Parameters](https://img.shields.io/badge/Parameters-~0.47M-red.svg)](#architecture)
[![CPU Latency](https://img.shields.io/badge/CPU%20Latency-~3.2ms-purple.svg)](#benchmarks)
[![GGUF](https://img.shields.io/badge/Ollama%2FlllamaCpp-GGUF%20Available-green.svg)](#gguf--ollama)

</div>

A **sub-half-million-parameter decoder-only transformer** trained completely from scratch to act as a lightweight, low-latency **function-calling router**. Given a user utterance, TinyLM decides whether to invoke an external tool (`web_search`) with structured arguments, or abstain (`none`) for chit-chat and greetings.

Built as an educational and empirical case study: *how small can a specialized router be while still maintaining high precision?*

---

## 🚀 Quick Links

| Resource | Link |
| :--- | :--- |
| 📦 **Full Source Code, Training & Eval Pipeline** | [github.com/Safeeq008/tiny-fc-lm](https://github.com/Safeeq008/tiny-fc-lm) |
| 🤗 **Model Weights (this page)** | [huggingface.co/Safeeq/FCtiny](https://huggingface.co/Safeeq/FCtiny) |
| 📄 **Detailed README & Architecture Docs** | [README.md on GitHub](https://github.com/Safeeq008/tiny-fc-lm/blob/main/README.md) |
| 🐳 **Docker Deployment** | [Dockerfile](https://github.com/Safeeq008/tiny-fc-lm/blob/main/Dockerfile) · [DEPLOY.md](https://github.com/Safeeq008/tiny-fc-lm/blob/main/DEPLOY.md) |
| 🦙 **Ollama / llama.cpp Guide** | [OLLAMA.md](https://github.com/Safeeq008/tiny-fc-lm/blob/main/OLLAMA.md) |

---

## 📦 Files in This Repository

| File | Size | Description |
| :--- | :--- | :--- |
| `model.safetensors` | ~1.8 MB | PyTorch weights in safe zero-copy format |
| `config.json` | ~1 KB | Architecture hyperparameters |
| `tokenizer.json` | ~121 KB | Custom 2048-vocab ByteLevel BPE tokenizer |
| `tinyfc.gguf` | ~1.0 MB | Quantized model for Ollama & llama.cpp |

---

## ⚡ Quickstart

### Option A — Python (from this Hub)

```bash
git clone https://github.com/Safeeq008/tiny-fc-lm.git
cd tiny-fc-lm
pip install -r requirements.txt
python infer.py
```

### Option B — GGUF via Ollama (zero-install inference)

```bash
# Download the GGUF and Modelfile from GitHub
curl -LO https://github.com/Safeeq008/tiny-fc-lm/raw/main/Modelfile

# Pull the GGUF from Hugging Face and register it
ollama pull hf.co/Safeeq/FCtiny:latest
ollama run hf.co/Safeeq/FCtiny
```

Or using the local GGUF:
```bash
# Download tinyfc.gguf from the Files tab above, then:
ollama create tinyfc -f Modelfile
ollama run tinyfc "weather in new york today"
```

### Option C — FastAPI REST Server (from GitHub)

```bash
git clone https://github.com/Safeeq008/tiny-fc-lm.git
cd tiny-fc-lm && pip install -r requirements.txt
python serve.py 8000
```
- Web UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### Option D — Docker

```bash
docker pull ghcr.io/safeeq008/tiny-fc-lm:latest  # or build locally:
git clone https://github.com/Safeeq008/tiny-fc-lm.git
docker build -t tiny-fc-lm . && docker run -p 8000:8000 tiny-fc-lm
```

---

## 🔍 Python Inference (Manual)

```python
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

# ── Clone the repo for model.py ──
# git clone https://github.com/Safeeq008/tiny-fc-lm.git
import sys; sys.path.insert(0, "tiny-fc-lm")
from model import TinyLM

# 1. Load weights + tokenizer (download from Files tab above)
state_dict = load_file("model.safetensors")
tok = Tokenizer.from_file("tokenizer.json")

# 2. Instantiate model
model = TinyLM(vocab=2048, d=80, n_layers=4, n_heads=4, ffn_mult=4, max_len=80)
model.load_state_dict(state_dict)
model.eval()

# 3. Encode prompt with special tokens (<user>=1, <call>=2)
user_input = "weather in chennai today"
prompt_ids = [1] + tok.encode(user_input).ids + [2]

# 4. Greedily decode
with torch.no_grad():
    ids = torch.tensor([prompt_ids])
    for _ in range(30):
        logits = model(ids)[0]
        next_id = logits[0, -1].argmax().item()
        if next_id == 3:  # <end> token
            break
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=-1)
    output_ids = ids[0, len(prompt_ids):].tolist()
    print(tok.decode(output_ids))
# → web_search|query=chennai weather today|recency=day

# For full constrained decoding + live DuckDuckGo dispatch, see:
# https://github.com/Safeeq008/tiny-fc-lm/blob/main/infer.py
```

---

## 📐 Architecture

| Hyperparameter | Value | Notes |
| :--- | :---: | :--- |
| **Total Parameters** | 471,760 (~0.47M) | Trainable weight count |
| **Layers** | 4 | Transformer decoder blocks |
| **Hidden Dim** | 80 | Embedding & layer dimensionality |
| **Attention Heads** | 4 | Head dim = 20 (even for RoPE) |
| **Positional Encoding** | RoPE | θ = 10000.0 |
| **Normalization** | RMSNorm | ε = 1e-6, pre-norm |
| **FFN** | GELU (4× width) | Hidden dim = 320 |
| **Weight Tying** | ✅ Yes | Input embeddings = output head |
| **Vocabulary Size** | 2,048 | Custom ByteLevel BPE |
| **Context Length** | 80 tokens | Max prompt + generation |

---

## 📊 Benchmarks

| Metric | Val (In-Distribution) | OOD (Held-Out Phrasings) |
| :--- | :---: | :---: |
| **Exact Match Accuracy** | **~99.8%** | **~88.2%** |
| **Routing Decision Accuracy** | **99.9%** | **96.4%** |
| **Routing Precision (Tool)** | **99.9%** | **97.1%** |
| **Routing Recall (Tool)** | **99.9%** | **98.8%** |
| **Query Slot Exact Match** | **99.8%** | **89.5%** |
| **Recency Slot Accuracy** | **99.9%** | **97.2%** |
| **Syntactic Validity Rate** | **100.0%** | **99.8%** |
| **CPU Inference Latency** | **~3.2 ms** | **~3.4 ms** |

> OOD = templates strictly held-out from training (unseen surface forms, same entity pools).

---

## 🔀 Output Protocol

TinyLM outputs a strict, pipe-delimited schema:

```
web_search|query=<search query>|recency=<day|week|any>
none
```

**Examples:**

| User Input | Model Output |
| :--- | :--- |
| `weather in tokyo today` | `web_search\|query=tokyo weather today\|recency=day` |
| `latest AI news` | `web_search\|query=latest ai news\|recency=day` |
| `iphone 16 price in india` | `web_search\|query=iphone 16 price india\|recency=week` |
| `hi how are you` | `none` |
| `tell me a joke` | `none` |
| `what is 2 + 2` | `none` |

---

## 🦙 GGUF & Ollama

A `tinyfc.gguf` artifact (~1.09 MB, fp16) is available in the **Files** tab above. This is a canonical Llama-architecture twin (~512k params, SwiGLU MLP) trained on the same task and tokenizer, exported for llama.cpp-compatible runtimes.

**Quick usage:**
```bash
# Download tinyfc.gguf from the Files tab, then:
ollama create tinyfc -f Modelfile
ollama run tinyfc "latest ethereum price"
# → web_search|query=ethereum price today|recency=day
```

See the full guide in [OLLAMA.md](https://github.com/Safeeq008/tiny-fc-lm/blob/main/OLLAMA.md).

---

## 🏗️ Reproduce from Scratch

Full pipeline available at [github.com/Safeeq008/tiny-fc-lm](https://github.com/Safeeq008/tiny-fc-lm):

```bash
git clone https://github.com/Safeeq008/tiny-fc-lm.git
cd tiny-fc-lm && pip install -r requirements.txt

python gen_data.py      # 1. Generate 80k synthetic training samples
python train_tok.py     # 2. Train custom 2048-vocab BPE tokenizer
python train.py         # 3. Train TinyLM-FC model (~5-10 min, CPU)
python eval.py          # 4. Full benchmark suite (val + OOD)
python infer.py         # 5. Interactive CLI with live DuckDuckGo search
python serve.py 8000    # 6. FastAPI server + web UI

# Optional: GGUF/Ollama path
python train_llama.py   # 7. Train Llama-arch twin
python export_gguf.py   # 8. Convert to GGUF
ollama create tinyfc -f Modelfile && ollama run tinyfc
```

---

## ⚠️ Known Limitations

1. **Single-tool**: Strictly routes `web_search` only — no multi-tool schemas.
2. **English only**: Custom BPE tokenizer trained on English synthetic corpus.
3. **80-token context window**: Inputs > ~60 user tokens are silently truncated.
4. **Synthetic training data**: Real-world query distributions are far noisier; OOD generalization is limited to phrasing variation, not domain shift.
5. **Constrained decoding required**: Raw greedy decoding may produce malformed outputs; use `infer.py`'s constrained decoder for production.

---

## 📄 License

Released under the **MIT License** — free for research, benchmarking, commercial use, and edge deployment.

**Repository:** [github.com/Safeeq008/tiny-fc-lm](https://github.com/Safeeq008/tiny-fc-lm)
