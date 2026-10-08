"""Central configuration for the Tiny Function-Calling LM project.

Every tunable constant lives here so that all scripts (gen_data, train, eval,
infer, serve, export) share a single source of truth.  Changing a value here
propagates everywhere — no hunting through scattered magic numbers.

Design rationale
----------------
* **VOCAB_SIZE = 2048** — deliberately tiny; our synthetic corpus has a small
  lexicon, so a larger vocab wastes embedding parameters on tokens that never
  appear.  A production model on real-world text would need 32k–100k+.
* **D_MODEL = 80, N_HEADS = 4** → head_dim = 20.  Must be even for RoPE
  (pairs elements).  80 is small enough to train on a laptop CPU in minutes.
* **MAX_LEN = 80** — sufficient for "<user> ... <call> web_search|query=...|
  recency=..." with room to spare.  Not enough for multi-turn or long docs.
* **NONE_MARGIN = 0.6** — a conservative abstention threshold: the model says
  "none" only when the none-logit exceeds the web_search-logit by 0.6.  Tuned
  on the val set; lowering it makes the model more trigger-happy on searches,
  raising it makes it too cautious.  See eval.py for measuring the trade-off.

Limitations (be honest)
-----------------------
* These hyper-parameters are tuned for the *synthetic* data distribution only.
  Switching to real-world queries will require re-tuning and very likely a
  bigger model.
* MAX_LEN = 80 truncates any prompt longer than ~60 user tokens — long
  requests are silently clipped, which can break intent detection.
* The tool schema is hard-coded to a single tool (`web_search`) with three
  recency slots.  Extending to multiple tools requires changes in gen_data,
  train, infer, and eval.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Console encoding fix (Windows-specific)
# ---------------------------------------------------------------------------
# Windows cmd/PowerShell default to cp1252, which cannot encode ByteLevel
# tokenizer glyphs (U+0120), the '✗' marker in eval.py, or non-ASCII search
# results — all of which crash with UnicodeEncodeError when output is piped.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT = ROOT / "ckpt" / "model.pt"
TOK_PATH = ROOT / "ckpt" / "tokenizer.json"

# ---------------------------------------------------------------------------
# Special tokens  (ids are fixed by insertion order: 0, 1, 2, 3)
# ---------------------------------------------------------------------------
# <pad>  — padding for batched sequences (ignored by the loss function)
# <user> — marks the start of the user's natural-language request
# <call> — separator: everything after this is the model's tool-call output
# <end>  — signals that the model has finished generating its output
SPECIAL = ["<pad>", "<user>", "<call>", "<end>"]
PAD, USER, CALL, END = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Model architecture  (~0.47M params with vocab 2048)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 2048       # BPE vocab trained on the synthetic corpus
D_MODEL    = 80         # embedding / hidden dimension
N_LAYERS   = 4          # number of transformer blocks
N_HEADS    = 4          # attention heads per block  (head_dim = D_MODEL / N_HEADS = 20)
FFN_MULT   = 4          # feed-forward inner dimension = FFN_MULT * D_MODEL = 320
MAX_LEN    = 80         # maximum sequence length in tokens (prompt + completion)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE   = 128      # samples per gradient step
STEPS        = 8000     # total training steps (~8k × 128 = ~1M samples seen)
LR           = 2e-3     # peak learning rate (cosine-decayed after warmup)
WARMUP       = 200      # linear warmup steps (≈2.5% of total)
WEIGHT_DECAY = 0.1      # AdamW L2 regularisation coefficient
EVAL_EVERY   = 500      # run validation + save checkpoint every N steps

# ---------------------------------------------------------------------------
# Tool schema  (the only tool the model knows about)
# ---------------------------------------------------------------------------
# Maps tool_name → {param_name: allowed_values}.  Used by infer.py to
# validate parsed calls and by gen_data.py to construct gold labels.
TOOLS = {"web_search": {"recency": ["day", "week", "any"]}}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Abstention margin: the model predicts "none" (no tool needed) only if the
# first-token logit for "none" exceeds the logit for "web_search" by at least
# this many units.  Higher → fewer false-positive searches but more missed
# true searches.  0.6 was chosen to maximise exact-match on the val set.
NONE_MARGIN = 0.6

# ---------------------------------------------------------------------------
# Ollama / GGUF export paths  (Llama-architecture twin — see train_llama.py)
# ---------------------------------------------------------------------------
CKPT_HF = ROOT / "ckpt_hf"

# ---------------------------------------------------------------------------
# Hugging Face export of the custom TinyLM  (export_hf.py / push_to_hub.py)
# ---------------------------------------------------------------------------
HF_MODEL_DIR = ROOT / "hf_model"
HF_README_TEMPLATE = ROOT / "hf_model_readme_template.md"

# Separate tokenizer copy for the Ollama/GGUF path — keeps ckpt/tokenizer.json
# (used by the FastAPI deployment) completely untouched so that retraining the
# Llama twin never accidentally overwrites the main tokenizer.
TOK_PATH_LLAMA = ROOT / "ckpt" / "tokenizer_llama.json"