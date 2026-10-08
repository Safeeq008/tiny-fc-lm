"""Convert ckpt_hf/ (Llama-architecture checkpoint) to GGUF format for llama.cpp & Ollama.

Technical Background & The Tokenizer Pre-Tokenizer Problem
----------------------------------------------------------
`llama.cpp`'s `convert_hf_to_gguf.py` inspects HF tokenizers to identify the BPE
pre-tokenizer rule (e.g. GPT-2, LLaMA, DeepSeek). It does this by hashing the token
IDs produced on a hardcoded probe string. If the resulting hash is not in its
hard-coded registry of known open-source models, it refuses to convert and terminates:

    NotImplementedError: BPE pre-tokenizer was not recognized

Because our BPE tokenizer was trained from scratch in `train_tok.py`, its vocabulary
indices for the probe string are unique and never match the official pre-recorded hashes.

However, our tokenizer configuration is mathematically and structurally identical to
the GPT-2 byte-level pre-tokenizer:
    `ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)`

This script monkey-patches `TextModel.get_vocab_base_pre` to return `"gpt-2"`, allowing
llama.cpp's converter to successfully export vocabulary tokens, merges, special tokens,
and tensor weights without altering any llama.cpp source files.

Usage:
  python export_gguf.py              # Export in fp16 (default)
  python export_gguf.py --outtype q8_0 # Quantized 8-bit output
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from config import ROOT, CKPT_HF

DEFAULT_GGUF_OUT = ROOT / "tinyfc.gguf"
LLAMA_CPP_DIR = ROOT / "llama.cpp"
GGML_PRE = "gpt-2"


def convert(input_dir: Path = CKPT_HF, output_path: Path = DEFAULT_GGUF_OUT, outtype: str = "f16"):
    converter_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not converter_script.exists():
        sys.exit(
            f"Error: llama.cpp converter script not found at {converter_script}.\n"
            f"Ensure the submodule or directory '{LLAMA_CPP_DIR}' exists."
        )

    if not (input_dir / "config.json").exists():
        sys.exit(
            f"Error: Missing {input_dir / 'config.json'}.\n"
            f"Please run `python train_llama.py` first to produce the HF Llama checkpoint."
        )

    print(f"Loading llama.cpp conversion module from {converter_script}...")
    sys.path.insert(0, str(LLAMA_CPP_DIR))
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", converter_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Monkey-patch pre-tokenizer detection to recognize custom ByteLevel BPE
    from conversion import TextModel
    TextModel.get_vocab_base_pre = lambda self, tokenizer: GGML_PRE

    orig_argv = sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py",
        str(input_dir),
        "--outfile", str(output_path),
        "--outtype", outtype
    ]

    try:
        print(f"Converting {input_dir} -> {output_path} (format: {outtype})...")
        mod.main()
    finally:
        sys.argv = orig_argv

    if output_path.exists():
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSuccessfully generated GGUF artifact:")
        print(f"  Path: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Next step: Create Ollama model via `ollama create tinyfc -f Modelfile`")
    else:
        sys.exit("Error: Conversion script completed but output file was not found.")


def main():
    parser = argparse.ArgumentParser(description="Convert TinyLM Llama Checkpoint to GGUF")
    parser.add_argument("--outtype", type=str, default="f16",
                        choices=["f16", "f32", "q8_0"],
                        help="Data type for GGUF weights (default: f16)")
    parser.add_argument("--outfile", "-o", type=Path, default=DEFAULT_GGUF_OUT,
                        help=f"Target GGUF file path (default: {DEFAULT_GGUF_OUT})")
    parser.add_argument("--input-dir", "-i", type=Path, default=CKPT_HF,
                        help=f"Source Hugging Face directory (default: {CKPT_HF})")
    args = parser.parse_args()

    convert(input_dir=args.input_dir, output_path=args.outfile, outtype=args.outtype)


if __name__ == "__main__":
    main()
