"""Run the rmatif/diffusers `convert_anima_to_diffusers.py` script against
an Anima checkpoint and produce a diffusers-pipeline directory we can load
component-by-component from OneTrainer.

This is a thin wrapper: it just locates the right paths and invokes the
upstream script as a subprocess with --save_pipeline. We do not import
it directly because it expects `convert_cosmos_to_diffusers` to be on
sys.path (it sibling-imports from scripts/), so the cleanest invocation
is to run it with cwd set to the diffusers scripts directory.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\convert_anima_base_v1.py

Example for a different finetune:
    venv\\Scripts\\python.exe scripts\\util\\convert_anima_base_v1.py \
        --transformer-ckpt "D:/models/checkpoints/anima/my-finetune.safetensors" \
        --output-dir "D:/models/diffusers/anima/my-finetune"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Inputs (verified present on the workstation before writing this script).
ANIMA_DIT = Path("D:/ai/ComfyUI/models/diffusion_models/anima/anima-base-v1.0.safetensors")
QWEN3_06B = Path("D:/models/text_encoders/anima/qwen_3_06b_base.safetensors")
QWEN_IMAGE_VAE = Path("D:/ai/ComfyUI/models/vae/anima/qwen_image_vae.safetensors")
QWEN_TOKENIZER = Path("D:/models/tokenizers/qwen3-0.6b-base")
T5_TOKENIZER = Path("D:/models/tokenizers/t5-small-tokenizer")

# Output: standard "diffusers pipeline folder" layout (model_index.json
# at the top, one subdirectory per component).
OUTPUT_DIR = Path("D:/models/diffusers/anima/anima-base-v1.0")

# The converter lives inside the editable diffusers install. It
# sibling-imports convert_cosmos_to_diffusers, so we must run it with
# its own scripts/ dir as cwd.
DIFFUSERS_SCRIPTS = Path("D:/ai/tools/OneTrainer/venv/src/diffusers/scripts")
CONVERTER = DIFFUSERS_SCRIPTS / "convert_anima_to_diffusers.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an Anima checkpoint into a diffusers pipeline directory.")
    parser.add_argument(
        "--transformer-ckpt",
        type=Path,
        default=ANIMA_DIT,
        help="Path to the Anima .safetensors checkpoint to convert.",
    )
    parser.add_argument(
        "--text-encoder-ckpt",
        type=Path,
        default=QWEN3_06B,
        help="Path to the Qwen3 text encoder weights.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=Path,
        default=QWEN_IMAGE_VAE,
        help="Path to the Qwen Image VAE weights.",
    )
    parser.add_argument(
        "--qwen-tokenizer",
        type=Path,
        default=QWEN_TOKENIZER,
        help="Directory containing the Qwen tokenizer files.",
    )
    parser.add_argument(
        "--t5-tokenizer",
        type=Path,
        default=T5_TOKENIZER,
        help="Directory containing the T5 tokenizer files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination diffusers pipeline directory.",
    )
    parser.add_argument(
        "--diffusers-scripts-dir",
        type=Path,
        default=DIFFUSERS_SCRIPTS,
        help="Directory containing convert_anima_to_diffusers.py.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Output dtype passed through to the upstream converter.",
    )
    return parser.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"missing {label}: {path}")


def _abs_path(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> int:
    args = _parse_args()
    args.transformer_ckpt = _abs_path(args.transformer_ckpt)
    args.text_encoder_ckpt = _abs_path(args.text_encoder_ckpt)
    args.vae_ckpt = _abs_path(args.vae_ckpt)
    args.qwen_tokenizer = _abs_path(args.qwen_tokenizer)
    args.t5_tokenizer = _abs_path(args.t5_tokenizer)
    args.output_dir = _abs_path(args.output_dir)
    args.diffusers_scripts_dir = _abs_path(args.diffusers_scripts_dir)

    converter = args.diffusers_scripts_dir / "convert_anima_to_diffusers.py"

    _require(args.transformer_ckpt, "Anima DiT safetensors")
    _require(args.text_encoder_ckpt, "Qwen3 0.6B base safetensors")
    _require(args.vae_ckpt, "Qwen-Image VAE safetensors")
    _require(args.qwen_tokenizer, "Qwen tokenizer dir (run fetch_anima_tokenizers.py)")
    _require(args.t5_tokenizer, "T5 tokenizer dir (run fetch_anima_tokenizers.py)")
    _require(converter, "Anima converter script in the diffusers checkout")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(converter),
        "--transformer_ckpt_path",
        str(args.transformer_ckpt),
        "--text_encoder_ckpt_path",
        str(args.text_encoder_ckpt),
        "--vae_ckpt_path",
        str(args.vae_ckpt),
        "--qwen_tokenizer_path",
        str(args.qwen_tokenizer),
        "--t5_tokenizer_path",
        str(args.t5_tokenizer),
        "--output_path",
        str(args.output_dir),
        "--save_pipeline",
        "--dtype",
        args.dtype,
    ]

    print("Running converter:")
    for tok in cmd:
        print(f"  {tok}")
    print()

    # cwd=DIFFUSERS_SCRIPTS so the converter's `from convert_cosmos_to_diffusers
    # import convert_transformer` resolves correctly.
    proc = subprocess.run(cmd, cwd=str(args.diffusers_scripts_dir), check=False)
    if proc.returncode != 0:
        return proc.returncode

    print()
    print(f"OK. Output at: {args.output_dir}")
    print(f"Inspect with: dir {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
