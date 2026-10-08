"""Train a standard Llama-architecture twin of TinyLM using Hugging Face transformers.

Why This Separate Training Script Exists
---------------------------------------
`model.py` implements a minimal custom transformer using fused QKV projections
and a simple two-layer GELU MLP. However, runtime engines like **llama.cpp** and
**Ollama** execute a fixed set of canonical architectures (Llama, GPT-2, Falcon, etc.)
and do not support arbitrary custom architectures.

`transformers.LlamaForCausalLM` uses the exact tensor layout that `llama.cpp`'s
`convert_hf_to_gguf.py` natively parses:
  - Separate `q_proj`, `k_proj`, `v_proj`, and `o_proj` weight tensors.
  - Gated SwiGLU MLP: `gate_proj`, `up_proj`, and `down_proj`.
  - Standard RMSNorm and RoPE.

Because SwiGLU uses three matrices per MLP instead of two, the parameter count
is slightly higher:
  - Custom TinyLM (`model.py`): ~471,760 parameters
  - Llama twin (`train_llama.py`): ~512,720 parameters

Pipeline:
  1. python gen_data.py
  2. python train_tok.py
  3. python train_llama.py   --> saves to ckpt_hf/
  4. python export_gguf.py   --> produces tinyfc.gguf
  5. ollama create tinyfc -f Modelfile
"""

import argparse
import json
import math
import random
import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from config import (
    BATCH_SIZE, CKPT_HF, DATA_DIR, EVAL_EVERY, LR, MAX_LEN, PAD, STEPS,
    TOK_PATH, TOK_PATH_LLAMA, USER, CALL, END, WARMUP, WEIGHT_DECAY,
)


def encode(tok: Tokenizer, r: dict) -> tuple[list[int], list[int]]:
    """Tokenize user request and target tool call into prompt and labels."""
    p = [USER] + tok.encode(r["user"]).ids + [CALL]
    t = tok.encode(r["call"]).ids + [END]
    ids = (p + t)[:MAX_LEN]
    # Loss only applied to the completion portion (tool call)
    labels = ([-100] * len(p) + t)[:MAX_LEN]
    return ids, labels


def load(tok: Tokenizer, name: str) -> list[tuple[list[int], list[int]]]:
    """Load JSONL dataset split from DATA_DIR."""
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return [encode(tok, json.loads(l)) for l in f]


def batchify(rows: list[tuple[list[int], list[int]]], device: str):
    """Collate samples into padded batch tensors with attention mask."""
    L = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), L), dtype=torch.long)
    for k, (i, l) in enumerate(rows):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        attn[k, :len(i)] = 1
    return ids.to(device), lab.to(device), attn.to(device)


@torch.no_grad()
def val_loss(model, val, device: str) -> float:
    """Compute validation loss across batches."""
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(val), 256):
        ids, lab, attn = batchify(val[s:s + 256], device)
        tot += model(input_ids=ids, attention_mask=attn, labels=lab).loss.item()
        n += 1
    model.train()
    return tot / n


def main():
    parser = argparse.ArgumentParser(description="Train Llama-Architecture Twin of TinyLM")
    parser.add_argument("--steps", type=int, default=STEPS, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Peak learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)

    # Use dedicated Llama tokenizer copy if available; fallback to main tokenizer
    tok_path = TOK_PATH_LLAMA if TOK_PATH_LLAMA.exists() else TOK_PATH
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run `python train_tok.py` first.")

    raw_tok = Tokenizer.from_file(str(tok_path))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="<pad>",
        bos_token="<user>",
        eos_token="<end>",
        additional_special_tokens=["<call>"],
    )

    # Architecture configuration aligned with canonical Llama specs
    cfg = LlamaConfig(
        vocab_size=raw_tok.get_vocab_size(),
        hidden_size=80,
        intermediate_size=256,       # SwiGLU hidden projection
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,       # Standard Multi-Head Attention
        max_position_embeddings=MAX_LEN,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        pad_token_id=PAD,
        bos_token_id=USER,
        eos_token_id=END,
    )

    model = LlamaForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training Llama-architecture model | Parameters: {n_params:,} | Device: {device}")

    train, val = load(raw_tok, "train.jsonl"), load(raw_tok, "val.jsonl")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    def lr_at(s: int) -> float:
        if s < WARMUP:
            return args.lr * (s + 1) / WARMUP
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (args.steps - WARMUP))))

    order, ptr = list(range(len(train))), 0
    random.shuffle(order)

    for step in range(args.steps):
        if ptr + args.batch_size > len(order):
            random.shuffle(order)
            ptr = 0

        batch = [train[i] for i in order[ptr:ptr + args.batch_size]]
        ptr += args.batch_size
        ids, lab, attn = batchify(batch, device)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr_at(step):.2e}")

        if step % EVAL_EVERY == 0 or step == args.steps - 1:
            vl = val_loss(model, val, device)
            print(f"   >> val loss: {vl:.4f}")

    CKPT_HF.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(CKPT_HF)
    hf_tok.save_pretrained(CKPT_HF)
    print(f"\nSaved Hugging Face format Llama model to: {CKPT_HF}")


if __name__ == "__main__":
    main()