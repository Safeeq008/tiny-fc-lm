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
pipeline_tag: text-generation
widget:
- text: "weather in tokyo today"
- text: "tell me a joke"
- text: "who is marie curie"
---

# Tiny Function-Calling LM (TinyLM-FC ~0.47M parameters)

A sub-half-million parameter decoder-only transformer trained from scratch to act as a **deterministic function-calling router**. Given an incoming user utterance, TinyLM decides whether to invoke an external search tool (`web_search`) with structured parameters or abstain (`none`) for general conversation.

Built as an educational and empirical case study on how small a specialized router model can be while maintaining high precision.

- **Checkpoint & Weights:** [Safeeq/tiny-fc-lm](https://huggingface.co/Safeeq/tiny-fc-lm)
- **Source Code Repository:** [GitHub Repository](https://github.com/Safeeq008/tiny-fc-lm)

---

## Model Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | 471,760 (~0.47M) | Trainable weight count |
| **Layers** | 4 | Transformer decoder blocks |
| **Hidden Dim ($d_{model}$)** | 80 | Embedding and layer dimensionality |
| **Attention Heads** | 4 | Head dimension = 20 (even for RoPE) |
| **Positional Encoding** | RoPE (Rotary) | Base frequency $\theta = 10000.0$ |
| **Normalization** | RMSNorm | $\epsilon = 10^{-6}$ (pre-norm configuration) |
| **Feed-Forward Network** | GELU (4× width) | Hidden dimension = 320 |
| **Weight Tying** | Yes | Input embeddings tied with output linear head |
| **Vocabulary Size** | 2,048 | Custom ByteLevel BPE trained on domain syntax |
| **Context Length** | 80 tokens | Maximum prompt + generation sequence length |

---

## Output Protocol & Grammar

TinyLM outputs a strict, pipe-delimited schema:
```text
web_search|query=<search query>|recency=<day|week|any>
none
```

- `web_search`: Invokes external search.
  - `query`: Formatted search terms extracted and normalized from user intent.
  - `recency`: Temporal constraint bucket (`day`, `week`, or `any`).
- `none`: Abstention signal for greetings, chit-chat, creative prompts, or statements not requiring search.

---

## Evaluation Benchmark & Rigor

The model was evaluated on both in-distribution validation data and out-of-distribution (OOD) phrasing sets containing unseen syntactic templates:

| Metric | Validation Split (In-Distribution) | OOD Split (Held-Out Phrasings) |
| :--- | :---: | :---: |
| **Exact Match Accuracy** | **~99.8%** | **~88.2%** |
| **Routing Decision Accuracy** | **99.9%** | **96.4%** |
| **Routing Precision (Tool)** | **99.9%** | **97.1%** |
| **Routing Recall (Tool)** | **99.9%** | **98.8%** |
| **Query Slot Exact Match** | **99.8%** | **89.5%** |
| **Recency Slot Accuracy** | **99.9%** | **97.2%** |
| **Syntactic Validity Rate** | **100.0%** | **99.8%** |
| **Inference Latency (CPU)** | **~3.2 ms** | **~3.4 ms** |

*Note: In OOD evaluations, templates were strictly held-out from training. The entity vocabulary remained consistent with training pools.*

---

## Honest Limitations & Known Failure Modes

1. **Narrow Task Domain**: This model is strictly a router for `web_search`. It does not generate conversational responses or answers to search queries.
2. **Vocabulary Memorization vs Entity Extraction**: At 471k parameters, the model partially memorizes entity associations rather than performing open-world named entity recognition. Genuinely unseen foreign names or novel technical terms outside the 2,048-token vocabulary may be split sub-optimally or mapped to known training concepts.
3. **English Monolingual**: The custom BPE tokenizer and training corpus are exclusively English.
4. **Context Window Constraint**: Inputs longer than 60 tokens are truncated to conform to `MAX_LEN=80`.
5. **Greedy / Constrained Decoding Dependency**: Best results require the constrained decoding routine implemented in `infer.py` (which forces the first-token tool name and validates parameters).

---

## Quickstart: Python Inference

```python
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from model import TinyLM  # Available in companion GitHub repo

# 1. Load weights and custom tokenizer
state_dict = load_file("model.safetensors")
tok = Tokenizer.from_file("tokenizer.json")

# 2. Instantiate TinyLM
model = TinyLM(vocab=2048, d=80, n_layers=4, n_heads=4, ffn_mult=4, max_len=80)
model.load_state_dict(state_dict)
model.eval()

# 3. Format input sequence
user_input = "weather in chennai today"
prompt_ids = [1] + tok.encode(user_input).ids + [2]  # <user>=1, <call>=2

# 4. Generate prediction
with torch.no_grad():
    logits = model(torch.tensor([prompt_ids]))[0][0, -1]
    # For full constrained decoding and live DuckDuckGo dispatch, see infer.py
```

---

## Ethical Considerations & Environmental Impact

- **Training Footprint**: Trained on CPU in under 10 minutes (~0.002 kWh energy consumed).
- **Deployment Efficiency**: Runs at sub-5ms latency on a single CPU thread with negligible memory footprint (~2MB RAM).

---

## Citation & License

Released under the **MIT License**. Free for research, benchmarking, and edge deployment.
