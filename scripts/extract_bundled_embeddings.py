"""Extract bundled embeddings from a OneTrainer LoRA .safetensors file.

When a LoRA is trained with `bundle_additional_embeddings=true` (the default),
its additional embeddings are stored inside the LoRA file under keys of the
form `bundle_emb.{placeholder}.{encoder_key}`. This script reads a LoRA
.safetensors, groups the bundled embedding tensors by placeholder, and writes
one standalone .safetensors per placeholder -- so a bundled embedding can be
reused independently of the adapter it shipped inside.

Usage:
    python scripts/extract_bundled_embeddings.py <lora.safetensors> [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Put the repo root on sys.path so `modules` resolves when this script is run
# directly from scripts/ (sys.path[0] is the script dir, not the repo root).
# path_util imports only json/os -- no torch -- so this stays cheap. Mirrors
# scripts/generate_debug_report.py's fallback import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.util.path_util import safe_filename  # noqa: E402

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

BUNDLE_PREFIX = "bundle_emb."


def extract(lora_path: Path, out_dir: Path) -> list[Path]:
    grouped: dict[str, dict] = defaultdict(dict)

    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safetensors handle, not a dict
            if not key.startswith(BUNDLE_PREFIX):
                continue
            remainder = key[len(BUNDLE_PREFIX):]
            # rpartition: the encoder key (qwen/qwen_out/t5/...) never contains a
            # dot, but a placeholder can (e.g. "v1.0"), so split from the RIGHT to
            # keep the placeholder whole -- a left partition() mis-assigns it as
            # "v1" + "0.qwen".
            placeholder, _, encoder_key = remainder.rpartition(".")
            if not encoder_key:
                print(f"  skipping malformed key: {key}")
                continue
            grouped[placeholder][encoder_key] = f.get_tensor(key)

    if not grouped:
        print(f"No bundled embeddings found in {lora_path.name}")
        print("(no keys starting with 'bundle_emb.')")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for placeholder, tensors in grouped.items():
        # OneTrainer's own sanitizer (the same call every *EmbeddingSaver.save_multiple
        # makes on a placeholder): strips path separators AND the characters illegal on
        # Windows (`<>:"|?*`), so a bracketed placeholder like "<token>" -- the default
        # placeholder value -- yields a writeable, loadable "token.safetensors" instead
        # of an invalid "<token>.safetensors".
        safe_name = safe_filename(placeholder, allow_spaces=False, max_length=None)
        out_path = out_dir / f"{safe_name}.safetensors"
        save_file(tensors, str(out_path))
        keys_str = ", ".join(sorted(tensors.keys()))
        print(f"  wrote {out_path.name}  [{keys_str}]")
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora", type=Path, help="Path to LoRA .safetensors")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: {lora_stem}_embeddings/ next to the LoRA)",
    )
    args = parser.parse_args()

    lora_path: Path = args.lora.resolve()
    if not lora_path.is_file():
        raise SystemExit(f"not a file: {lora_path}")

    out_dir = args.out_dir or (lora_path.parent / f"{lora_path.stem}_embeddings")
    print(f"reading: {lora_path}")
    print(f"writing into: {out_dir}")
    written = extract(lora_path, out_dir)
    print(f"\ndone -- {len(written)} embedding file(s) written")


if __name__ == "__main__":
    main()
