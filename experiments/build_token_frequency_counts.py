#!/usr/bin/env python3
"""Build token-frequency counts from a local text/code corpus.

Usage:
    python experiments/build_token_frequency_counts.py \
      --model-name Qwen/Qwen2.5-Coder-7B \
      --input-root /path/to/corpus \
      --output outputs/confounds/qwen2_5_coder_7b_counts.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".sh",
}


def _iter_files(root: Path, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]

    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    ]
    files = sorted(files)
    if max_files > 0:
        files = files[:max_files]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True, help="Output .npy counts path.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=5000,
        help="Maximum files to tokenize (0 = unlimited).",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=200_000,
        help="Clip each file to this many chars to bound runtime.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help=(
            "Optional output length override. If larger than tokenizer vocab, "
            "counts are zero-padded; if smaller, counts are truncated."
        ),
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Output must be a .npy path")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        trust_remote_code=True,
    )
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    output_vocab_size = (
        int(args.target_vocab_size)
        if args.target_vocab_size is not None
        else tokenizer_vocab_size
    )
    if output_vocab_size <= 0:
        raise ValueError(f"Invalid target vocab size: {output_vocab_size}")
    counts = np.zeros(output_vocab_size, dtype=np.int64)

    files = _iter_files(input_root, max_files=int(args.max_files))
    if not files:
        raise RuntimeError(f"No corpus files found under {input_root}")

    total_chars = 0
    total_tokens = 0
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if not text:
            continue

        clip_n = int(args.max_chars_per_file)
        if clip_n > 0 and len(text) > clip_n:
            text = text[:clip_n]

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue

        bincount = np.bincount(token_ids, minlength=tokenizer_vocab_size).astype(np.int64)
        if output_vocab_size == tokenizer_vocab_size:
            counts += bincount
        elif output_vocab_size > tokenizer_vocab_size:
            counts[:tokenizer_vocab_size] += bincount
        else:
            counts += bincount[:output_vocab_size]
        total_chars += len(text)
        total_tokens += int(len(token_ids))

    np.save(output_path, counts)
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "model_name": args.model_name,
        "revision": args.revision,
        "input_root": str(input_root),
        "n_files": len(files),
        "total_chars": int(total_chars),
        "total_tokens": int(total_tokens),
        "tokenizer_vocab_size": int(tokenizer_vocab_size),
        "output_vocab_size": int(output_vocab_size),
        "target_vocab_size_arg": (
            int(args.target_vocab_size)
            if args.target_vocab_size is not None
            else None
        ),
        "output_path": str(output_path),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote counts: {output_path}")
    print(f"Wrote meta:   {meta_path}")
    print(f"Files: {len(files)} | Chars: {total_chars} | Tokens: {total_tokens}")


if __name__ == "__main__":
    main()
