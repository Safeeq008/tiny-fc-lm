# Running TinyLM in Ollama & llama.cpp

This guide details how to export, quantize, and execute TinyLM using **Ollama** and **llama.cpp**.

---

## 1. Architectural Challenge & Solution

Ollama executes models via `llama.cpp`. `llama.cpp` does not support arbitrary custom neural network architectures; it executes a fixed catalog of standard open-source designs (such as LLaMA, Mistral, GPT-2, Gemma, and Falcon).

Our primary reference model in [`model.py`](model.py) uses a fused QKV projection and a standard GELU MLP. Because `llama.cpp` has no converter for this specific layout, we provide an architectural twin:

- **[`train_llama.py`](train_llama.py)**: Trains the identical routing task using Hugging Face `transformers.LlamaForCausalLM`.
- **Tensor Layout**: Uses separate $Q, K, V, O$ linear projections and a 3-matrix gated SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`).
- **Parameter Count**: Slightly higher (~512,720 vs ~471,760 parameters) due to SwiGLU's third projection matrix.

Both models share the exact same training data, tokenizer vocabulary, and task formulation.

---

## 2. Step-by-Step Walkthrough

### Step 1: Train the Llama-Architecture Twin
Train the twin model and export standard Hugging Face weights to `ckpt_hf/`:
```bash
python train_llama.py
```
This produces:
- `ckpt_hf/config.json`
- `ckpt_hf/model.safetensors`
- `ckpt_hf/tokenizer.json`

### Step 2: Convert to GGUF Format
Convert `ckpt_hf/` to `tinyfc.gguf` using the conversion wrapper:
```bash
# Export in fp16 precision (default, ~1.1 MB):
python export_gguf.py

# Or export in 8-bit quantization (~0.6 MB):
python export_gguf.py --outtype q8_0
```

#### Why `export_gguf.py` is needed instead of direct `llama.cpp` CLI:
`llama.cpp/convert_hf_to_gguf.py` verifies BPE pre-tokenizers by hashing tokenization of a hardcoded probe string against a list of known models. If the hash is unlisted, it raises:
```text
NotImplementedError: BPE pre-tokenizer was not recognized
```
Our tokenizer is mathematically identical to GPT-2's ByteLevel BPE (`ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)`). `export_gguf.py` injects this identity (`tokenizer.ggml.pre = "gpt-2"`) so the official converter runs without modifying `llama.cpp`.

---

## 3. Creating and Registering the Ollama Model

A [`Modelfile`](Modelfile) is provided in the root directory:
```dockerfile
FROM ./tinyfc.gguf

# Single-turn stateless router template:
# Drops previous conversation history so multi-turn chats do not echo past queries.
TEMPLATE """{{- if .Messages }}{{ $p := "" }}{{ range .Messages }}{{ if eq .Role "user" }}{{ $p = .Content }}{{ end }}{{ end }}<user> {{ $p }} <call>{{- else }}<user> {{ .Prompt }} <call>{{- end }}"""

PARAMETER stop "<end>"
PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER top_p 1.0
```

### Register with Ollama
```bash
ollama create tinyfc -f Modelfile
```

### Run Interactive Console
```bash
ollama run tinyfc
```

### Test Routing Prompts:
```text
>>> weather in chennai today
web_search|query=chennai weather today|recency=day

>>> who is elon musk
web_search|query=elon musk|recency=any

>>> tell me a joke
none
```

---

## 4. Querying Ollama via REST API

Once registered, you can query TinyLM through Ollama's local HTTP API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyfc",
  "prompt": "nvidia stock price today",
  "stream": false
}'
```

**JSON Response**:
```json
{
  "model": "tinyfc",
  "created_at": "2026-10-02T10:00:00Z",
  "response": "web_search|query=nvidia stock price|recency=day",
  "done": true
}
```

---

## 5. Comparative Trade-Offs: FastAPI vs Ollama Runtime

| Feature | FastAPI Deployment (`serve.py`) | Ollama Deployment (`tinyfc`) |
| :--- | :--- | :--- |
| **Decoding Strategy** | **Constrained Decoding** (Forced prefix & margin threshold) | **Unconstrained Greedy Sampling** |
| **Syntactic Reliability** | **100.0%** (Guaranteed valid tool syntax) | **~97.5%** (May occasionally produce minor syntax drift) |
| **Case Sensitivity** | **Case-Insensitive** (Auto-lowercased before BPE) | **Case-Sensitive** (User must send lowercase or normalize client-side) |
| **Tool Execution** | **Direct execution** (Fetches live DuckDuckGo results) | **Routing text only** (Client must dispatch tool call) |
| **Ecosystem Fit** | Standalone Python microservice / REST API | Local LLM orchestration stacks (Open WebUI, LangChain, Ollama API) |
| **RAM Footprint** | ~35 MB | Dependent on Ollama daemon runtime |

---

## 6. Honest Limitations & Operational Tips

1. **Lowercase Input Recommended**: While `serve.py` normalizes casing automatically, `llama.cpp`'s BPE implementation ignores the lowercase normalizer tag in the GGUF metadata. For best results in Ollama, provide lowercase or clean queries.
2. **Stateless Prompt Template**: The `Modelfile` is intentionally configured as single-turn. Passing a multi-turn conversation will extract only the last user utterance to ensure clean routing decisions.
3. **No Automatic Tool Dispatch in Ollama**: Ollama outputs the raw string `web_search|query=...`. Your application must parse this string and invoke your search backend.
