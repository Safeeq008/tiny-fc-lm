"""Train a BPE tokenizer on the synthetic corpus.

Why a custom tokenizer?
-----------------------
Pre-trained tokenizers (GPT-2, LLaMA, etc.) have 32k–100k+ tokens — far too
large for a 0.47M-parameter model where the embedding table alone would
dominate the parameter budget.  Training a 2,048-token BPE on the synthetic
data gives us a vocabulary perfectly tuned for the task's lexicon (tool names,
entity names, pipe-delimited call format).

Design choices
--------------
* **ByteLevel pre-tokenizer**: encodes arbitrary UTF-8 as byte tokens, so no
  unknown-token issues and the model can handle any character.
* **Lowercase normaliser**: the model operates in lowercase-only space.
  This halves the effective vocabulary for English and makes the model
  case-insensitive — users can type "Weather" or "weather" and get the same
  result.  Downside: the Ollama/GGUF path doesn't apply this normalisation
  (see OLLAMA.md), so Ollama users must send lowercase input.
* **Two copies of the tokenizer**: ``ckpt/tokenizer.json`` for the main
  pipeline (train.py / infer.py / serve.py) and ``ckpt/tokenizer_llama.json``
  for the Llama-architecture path (train_llama.py / export_gguf.py).  They
  are byte-identical but kept separate so that retraining the Llama twin
  never accidentally overwrites the main tokenizer.

Reproducibility
---------------
The tokenizer is deterministic given the training data.  Since gen_data.py is
also seeded, the full pipeline gen_data → train_tok → train is reproducible.

Usage:
  python gen_data.py      # must run first to create data/train.jsonl
  python train_tok.py     # creates ckpt/tokenizer.json + ckpt/tokenizer_llama.json
"""

import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from config import DATA_DIR, TOK_PATH, TOK_PATH_LLAMA, SPECIAL, VOCAB_SIZE


def texts():
    """Yield all user and call strings from the training data.

    The tokenizer sees both the user input and the gold tool-call strings so
    that it learns good subword splits for both natural language and the
    pipe-delimited call format.
    """
    with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield r["user"]
            yield r["call"]


def main():
    """Train a 2,048-token BPE and save it to disk."""
    # --- configure the tokenizer ---
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts(), trainer)

    # --- save ---
    TOK_PATH.parent.mkdir(exist_ok=True)
    tok.save(str(TOK_PATH))
    # Identical copy for the Ollama/GGUF path — see module docstring.
    tok.save(str(TOK_PATH_LLAMA))

    # --- verify ---
    print(f"Tokenizer trained successfully:")
    print(f"  vocab size : {tok.get_vocab_size()}")
    print(f"  saved to   : {TOK_PATH}")
    print(f"  copy at    : {TOK_PATH_LLAMA}")
    print(f"\nSample encoding:")
    sample = "web_search|query=chennai weather today|recency=day"
    tokens = tok.encode(sample).tokens
    print(f"  '{sample}'")
    print(f"  → {tokens}")


if __name__ == "__main__":
    main()
