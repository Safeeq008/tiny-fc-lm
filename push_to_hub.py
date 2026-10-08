"""Upload the exported Hugging Face model repository to Hugging Face Hub.

Prerequisites:
  1. pip install huggingface_hub
  2. Authenticate via:
     - `huggingface-cli login`, or
     - Pass `--token <HF_TOKEN>`, or
     - Set the `HF_TOKEN` environment variable.

Usage:
  python push_to_hub.py USERNAME/REPO_NAME
  python push_to_hub.py USERNAME/REPO_NAME --token YOUR_TOKEN
  python push_to_hub.py USERNAME/REPO_NAME --private
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from config import HF_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Upload TinyLM model folder to Hugging Face Hub")
    parser.add_argument("repo_id", type=str,
                        help="Target repository ID on Hugging Face (e.g. 'username/tiny-fc-lm')")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--model-dir", type=Path, default=HF_MODEL_DIR,
                        help=f"Path to exported HF directory (default: {HF_MODEL_DIR})")
    parser.add_argument("--private", action="store_true",
                        help="Create a private model repository")
    parser.add_argument("--include-gguf", action="store_true", default=False,
                        help="Also upload tinyfc.gguf artifact to the Hub repository if present")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if not args.model_dir.is_dir():
        sys.exit(
            f"Error: Model directory '{args.model_dir}' not found.\n"
            "Please run `python export_hf.py` first to generate the model artifacts."
        )

    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for fname in required_files:
        if not (args.model_dir / fname).exists():
            sys.exit(f"Error: Missing required artifact '{fname}' in '{args.model_dir}'.")

    print(f"Connecting to Hugging Face Hub for repo: '{args.repo_id}'...")
    try:
        create_repo(args.repo_id, repo_type="model", private=args.private, token=token, exist_ok=True)
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(args.model_dir),
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message="Update TinyLM-FC weights, config, tokenizer, and model card",
        )
        gguf_file = Path(args.model_dir).parent / "tinyfc.gguf"
        if args.include_gguf and gguf_file.exists():
            print(f"Uploading GGUF model: {gguf_file.name}...")
            api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo="tinyfc.gguf",
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
                commit_message="Add tinyfc.gguf for llama.cpp and Ollama",
            )
        print(f"Successfully published model to: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        sys.exit(f"Upload failed: {e}\nEnsure your token has 'write' permission and the repository name is correct.")


if __name__ == "__main__":
    main()
