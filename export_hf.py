"""Export the trained checkpoint to standard Hugging Face format (safetensors + config).

Output artifacts in hf_model/:
  - model.safetensors: Weights stored in zero-copy safe format
  - config.json: Architecture hyperparameters and model metadata
  - tokenizer.json: Fast BPE tokenizer definition (Hugging Face tokenizers compatible)
  - README.md: Model card detailing training, evaluation, and usage instructions

Usage:
  python export_hf.py
  python export_hf.py --output custom_hf_dir
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

from config import CKPT, TOK_PATH, HF_MODEL_DIR, HF_README_TEMPLATE


def export(checkpoint_path: Path = CKPT,
           tokenizer_path: Path = TOK_PATH,
           output_dir: Path = HF_MODEL_DIR,
           readme_template: Path = HF_README_TEMPLATE):
    """Package PyTorch checkpoint into Hugging Face model folder."""
    if not checkpoint_path.exists():
        sys.exit(f"Error: Checkpoint {checkpoint_path} not found. Run `python train.py` first.")
    if not tokenizer_path.exists():
        sys.exit(f"Error: Tokenizer {tokenizer_path} not found. Run `python train_tok.py` first.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu")
    state = ck["model"]
    cfg = ck["cfg"]

    # Ensure tensors are contiguous and clean for safetensors serialisation
    tensors = {k: v.contiguous() for k, v in state.items()}
    safetensors_path = output_dir / "model.safetensors"
    save_file(tensors, str(safetensors_path))

    n_params = sum(t.numel() for t in tensors.values())

    config = {
        "model_type": "custom",
        "architectures": ["TinyLM"],
        "vocab_size": cfg["vocab"],
        "d_model": cfg["d"],
        "hidden_size": cfg["d"],
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["n_heads"],
        "ffn_mult": cfg["ffn_mult"],
        "intermediate_size": cfg["ffn_mult"] * cfg["d"],
        "max_position_embeddings": cfg["max_len"],
        "positional_encoding": "rope",
        "rope_theta": 10000.0,
        "normalization": "rmsnorm",
        "activation_function": "gelu",
        "tie_word_embeddings": True,
        "num_parameters": n_params,
        "task": "function_calling_router",
        "allowed_tools": ["web_search"],
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer
    shutil.copy(tokenizer_path, output_dir / "tokenizer.json")

    # Copy or populate README.md
    if readme_template.exists():
        shutil.copy(readme_template, output_dir / "README.md")
    else:
        # Fallback minimal README if template not present
        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("# TinyLM Function Calling Model\n\nExported weights and config.\n")

    print(f"Successfully exported Hugging Face repository bundle:")
    print(f"  Target directory: {output_dir}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Artifacts created: {sorted(os.listdir(output_dir))}")


def main():
    parser = argparse.ArgumentParser(description="Export TinyLM Checkpoint to Hugging Face Directory")
    parser.add_argument("--output", "-o", type=Path, default=HF_MODEL_DIR,
                        help=f"Output directory (default: {HF_MODEL_DIR})")
    args = parser.parse_args()
    export(output_dir=args.output)


if __name__ == "__main__":
    main()
