"""Train the custom-architecture TinyLM (fused QKV + GELU MLP).

This is the **main training script** for the project.  It produces
``ckpt/model.pt`` (weights + architecture config dict), which is loaded by
``infer.py``, ``serve.py``, ``eval.py``, and ``export_hf.py``.

Training recipe
---------------
* **Optimiser**: AdamW with β=(0.9, 0.95) and weight decay 0.1 — the standard
  recipe for small transformers (following Chinchilla/GPT-3 practice).
* **Schedule**: linear warmup (200 steps) → cosine decay to 10% of peak LR.
  Cosine decay is smoother than step-decay and generally gives better final
  loss for a fixed compute budget.
* **Gradient clipping**: max-norm 1.0 — prevents exploding gradients during
  early training when loss is high.
* **Data loading**: simple in-memory shuffle-and-slice (no DataLoader) — the
  entire 80k-sample dataset fits in RAM, and the model is small enough that
  data loading is never the bottleneck.

Loss function
-------------
Cross-entropy with ``ignore_index=-100`` on the *prompt* portion.  The model
is trained to predict the tool-call tokens only; it receives no gradient
signal from the user's input tokens.  This focuses capacity on the task and
avoids wasting parameters on modelling user-prompt distributions.

Reproducibility
---------------
Both ``torch.manual_seed(0)`` and ``random.seed(0)`` are set, and the data
order is shuffled with the seeded stdlib RNG.  Results should be bitwise
reproducible on the same hardware + PyTorch version.  CUDA non-determinism
(cuDNN, atomics) may cause minor variation on GPU — set
``torch.use_deterministic_algorithms(True)`` if exact reproducibility is needed.

Limitations
-----------
* No learning-rate finder, no hyper-parameter search — the values in config.py
  were hand-tuned on a single run.  They work well for this synthetic task but
  are NOT transferable to other datasets.
* No gradient accumulation — effective batch size equals ``BATCH_SIZE``.
* No mixed-precision (fp16/bf16) — not needed at this scale.
* No early stopping — we always run the full ``STEPS``.

Usage:
  python gen_data.py      # if you haven't already
  python train_tok.py     # if you haven't already
  python train.py         # ~8000 steps, minutes on CPU
"""

import json
import math
import random
import torch
from tokenizers import Tokenizer

from config import (
    BATCH_SIZE, CKPT, D_MODEL, DATA_DIR, EVAL_EVERY, FFN_MULT, LR, MAX_LEN,
    N_HEADS, N_LAYERS, PAD, STEPS, TOK_PATH, USER, CALL, END,
    WARMUP, WEIGHT_DECAY,
)
from model import TinyLM


def encode(tok: Tokenizer, r: dict) -> tuple[list[int], list[int]]:
    """Encode a single (user, call) example into token ids and label ids.

    The sequence is: <user> [user tokens] <call> [call tokens] <end>

    Labels are set to -100 (PyTorch's ignore index) for the prompt portion
    so the loss only trains the model to predict the tool-call tokens.

    Returns:
        (input_ids, label_ids) — both truncated to MAX_LEN.
    """
    p = [USER] + tok.encode(r["user"]).ids + [CALL]
    t = tok.encode(r["call"]).ids + [END]
    return (p + t)[:MAX_LEN], ([-100] * len(p) + t)[:MAX_LEN]


def load(tok: Tokenizer, name: str) -> list[tuple[list[int], list[int]]]:
    """Load a JSONL data file and encode all examples."""
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return [encode(tok, json.loads(l)) for l in f]


def batchify(rows: list[tuple[list[int], list[int]]], device: str):
    """Pad and stack a list of encoded examples into batch tensors.

    Padding uses the PAD token id for inputs and -100 for labels (so padded
    positions contribute zero loss).

    Returns:
        (input_ids, labels) — both shape (batch_size, max_seq_len).
    """
    L = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    for k, (i, l) in enumerate(rows):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
    return ids.to(device), lab.to(device)


@torch.no_grad()
def val_loss(model: TinyLM, val: list, device: str) -> float:
    """Compute average cross-entropy loss on the validation set.

    Processes the data in chunks of 256 to avoid OOM on large val sets.
    Returns the mean loss across all chunks.
    """
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(val), 256):
        x, y = batchify(val[s:s + 256], device)
        tot += model(x, y)[1].item()
        n += 1
    model.train()
    return tot / n


def main():
    """Run the full training loop and save the checkpoint."""
    # --- setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)

    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = dict(
        vocab=tok.get_vocab_size(), d=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, ffn_mult=FFN_MULT, max_len=MAX_LEN,
    )
    model = TinyLM(**cfg).to(device)
    print(f"Model: TinyLM | params: {model.n_params():,} | device: {device}")
    print(f"Config: {cfg}")

    # --- data ---
    train, val = load(tok, "train.jsonl"), load(tok, "val.jsonl")
    print(f"Data: train={len(train):,}  val={len(val):,}")

    # --- optimiser ---
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    def lr_at(s: int) -> float:
        """Cosine learning-rate schedule with linear warmup.

        - Steps 0..WARMUP-1:  linear ramp from ~0 to LR.
        - Steps WARMUP..STEPS: cosine decay from LR to 0.1×LR.
        """
        if s < WARMUP:
            return LR * (s + 1) / WARMUP
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))))

    # --- training loop ---
    order, ptr = list(range(len(train))), 0
    random.shuffle(order)

    for step in range(STEPS):
        # Re-shuffle when we've exhausted the data
        if ptr + BATCH_SIZE > len(order):
            random.shuffle(order)
            ptr = 0

        x, y = batchify([train[i] for i in order[ptr:ptr + BATCH_SIZE]], device)
        ptr += BATCH_SIZE

        # Update learning rate
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # Forward + backward
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Logging
        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr_at(step):.2e}")

        # Periodic evaluation + checkpoint
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            vl = val_loss(model, val, device)
            print(f"   >> val loss {vl:.4f}")
            CKPT.parent.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, CKPT)

    print(f"\nTraining complete. Checkpoint saved to {CKPT}")
    print(f"Final val loss: {val_loss(model, val, device):.4f}")


if __name__ == "__main__":
    main()
