"""Rewrite a LoRA/LoKr adapter into the COMFY key namespace, in place of a resave.

ComfyUI matches an adapter against a model by key name, so a save written in
DIFFUSERS naming (``transformer.transformer_blocks.0.…``) loads **nothing**
against a native Anima model — no error, no keys, a LoRA that does exactly
nothing. Until upstream #1563 the fix was a fork-only converter script; #1563
deleted it and made the output format selectable instead, which fixes every
*future* save and none of the ones already on disk.

This is the missing piece for those. It is a rename and nothing else: no base
model is loaded, no weights are read for their values, no GPU is touched. The
documented alternative — ``scripts/convert_model.py --output-model-format
COMFY_LORA`` — goes through the model loader, so it needs the base model's path
on the box and several GB of I/O to change some strings.

The rename table is not written here. ``lora_namespace.nativize`` reads the very
declarations ``LoRASaverMixin._save_comfy`` reads, so this cannot drift from what
OneTrainer itself would have written had the run been configured that way.

Metadata is carried over verbatim. The header is what says which model the
adapter was trained for (``ot_config``, ``modelspec.architecture``) and it is
what this script *read* to pick the table; dropping it would make the output
un-analysable by the same code that just converted it.

Usage::

    python scripts/util/lora_nativize.py --in a.safetensors --out b.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_channel  # noqa: E402

# Before torch: the CUDA support libraries it loads write banners to fd 1 from
# C, and this script's stdout is its result. See ``result_channel``.
result_channel.claim()

import lora_namespace  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, metavar="PATH")
    parser.add_argument("--out", dest="dst", required=True, metavar="PATH")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_file():
        sys.exit(f"lora_nativize: no such file: {src}")

    with safe_open(str(src), framework="pt") as f:
        header = dict(f.metadata() or {})
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118 -- not a Mapping

    before = len(tensors)
    try:
        out = lora_namespace.nativize(tensors, header, src)
    except lora_namespace.NamespaceError as exc:
        sys.exit(f"lora_nativize: {exc}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), header)

    result_channel.emit_json(
        {
            "src": str(src),
            "dst": str(dst),
            "tensors_in": before,
            "tensors_out": len(out),
            # What the caller actually needs to know: the rename happened. A
            # count alone cannot say that -- a passthrough preserves it exactly.
            "native_prefixes": sorted(
                {k.split(".", 1)[0] for k in out}
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
