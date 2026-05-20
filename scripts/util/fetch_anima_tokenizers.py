"""Download just the tokenizer files for the two encoders Anima uses.

The Anima converter (`scripts/convert_anima_to_diffusers.py` inside the
pinned diffusers checkout) needs:

  --qwen_tokenizer_path : a directory containing Qwen2/Qwen3 tokenizer
                          files (loaded by AutoTokenizer)
  --t5_tokenizer_path   : a directory containing T5 tokenizer files
                          (loaded by T5TokenizerFast)

It does NOT need the model weights -- those live elsewhere on disk.
This script downloads only the tokenizer files, which is ~5 MB total.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\fetch_anima_tokenizers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


TOKENIZERS_ROOT = Path("D:/models/tokenizers")

# Qwen tokenizer files. AutoTokenizer dispatches via tokenizer_config.json
# to Qwen2Tokenizer (Qwen2/2.5/3 all share the same BPE vocab + class).
QWEN_REPO = "Qwen/Qwen3-0.6B-Base"
QWEN_DEST = TOKENIZERS_ROOT / "qwen3-0.6b-base"
QWEN_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "*.txt",  # README etc, harmless
]

# T5 tokenizer files. T5TokenizerFast needs spiece.model + tokenizer.json.
# The T5 vocab (32128 SentencePiece tokens) is identical across all
# google-t5/t5-* sizes, so the tiny t5-small repo is the cheapest source.
T5_REPO = "google-t5/t5-small"
T5_DEST = TOKENIZERS_ROOT / "t5-small-tokenizer"
T5_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
    "special_tokens_map.json",
]


def _fetch(repo: str, dest: Path, patterns: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {repo}  ->  {dest}")
    snapshot_download(
        repo_id=repo,
        allow_patterns=patterns,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    files = sorted(p.name for p in dest.iterdir() if p.is_file())
    print(f"        files: {files}")


def main() -> int:
    print(f"target root: {TOKENIZERS_ROOT}")
    _fetch(QWEN_REPO, QWEN_DEST, QWEN_PATTERNS)
    _fetch(T5_REPO, T5_DEST, T5_PATTERNS)

    print()
    print("OK. Pass these to convert_anima_to_diffusers.py:")
    print(f'  --qwen_tokenizer_path "{QWEN_DEST}"')
    print(f'  --t5_tokenizer_path   "{T5_DEST}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
