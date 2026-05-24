"""Extract bundled embeddings from a OneTrainer LoRA .safetensors file.

When a LoRA is trained with `bundle_additional_embeddings=true` (the default),
the additional embeddings are stored inside the LoRA file under keys of the
form `bundle_emb.{placeholder}.{encoder_key}`. The OneTrainer loader for
`additional_embeddings[].model_name` does NOT extract these on load; it
expects a standalone file with just the encoder keys (e.g. `qwen`, `t5`).

This script reads a LoRA .safetensors, groups bundled embedding tensors by
placeholder, and writes one standalone .safetensors per placeholder.

Usage:
    python extract_bundled_embeddings.py <lora.safetensors> [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

BUNDLE_PREFIX = "bundle_emb."


def extract(lora_path: Path, out_dir: Path) -> list[Path]:
    grouped: dict[str, dict[str, "torch.Tensor"]] = defaultdict(dict)

    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(BUNDLE_PREFIX):
                continue
            remainder = key[len(BUNDLE_PREFIX) :]
            placeholder, _, encoder_key = remainder.partition(".")
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
        safe_name = placeholder.replace("/", "_").replace("\\", "_")
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
    print(f"\ndone — {len(written)} embedding file(s) written")


if __name__ == "__main__":
    main()
