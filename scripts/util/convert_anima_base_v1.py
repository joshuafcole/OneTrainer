"""Run the rmatif/diffusers `convert_anima_to_diffusers.py` script against
the local anima-base-v1.0 checkpoint and produce a diffusers-pipeline
directory we can load component-by-component from OneTrainer.

This is a thin wrapper: it just locates the right paths and invokes the
upstream script as a subprocess with --save_pipeline. We do not import
it directly because it expects `convert_cosmos_to_diffusers` to be on
sys.path (it sibling-imports from scripts/), so the cleanest invocation
is to run it with cwd set to the diffusers scripts directory.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\convert_anima_base_v1.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Inputs (verified present on the workstation before writing this script).
ANIMA_DIT       = Path("D:/ai/ComfyUI/models/diffusion_models/anima/anima-base-v1.0.safetensors")
QWEN3_06B       = Path("D:/models/text_encoders/anima/qwen_3_06b_base.safetensors")
QWEN_IMAGE_VAE  = Path("D:/ai/ComfyUI/models/vae/anima/qwen_image_vae.safetensors")
QWEN_TOKENIZER  = Path("D:/models/tokenizers/qwen3-0.6b-base")
T5_TOKENIZER    = Path("D:/models/tokenizers/t5-small-tokenizer")

# Output: standard "diffusers pipeline folder" layout (model_index.json
# at the top, one subdirectory per component).
OUTPUT_DIR = Path("D:/models/diffusers/anima/anima-base-v1.0")

# The converter lives inside the editable diffusers install. It
# sibling-imports convert_cosmos_to_diffusers, so we must run it with
# its own scripts/ dir as cwd.
DIFFUSERS_SCRIPTS = Path("D:/ai/tools/OneTrainer/venv/src/diffusers/scripts")
CONVERTER = DIFFUSERS_SCRIPTS / "convert_anima_to_diffusers.py"


def _require(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"missing {label}: {path}")


def main() -> int:
    _require(ANIMA_DIT,      "Anima DiT safetensors")
    _require(QWEN3_06B,      "Qwen3 0.6B base safetensors")
    _require(QWEN_IMAGE_VAE, "Qwen-Image VAE safetensors")
    _require(QWEN_TOKENIZER, "Qwen tokenizer dir (run fetch_anima_tokenizers.py)")
    _require(T5_TOKENIZER,   "T5 tokenizer dir (run fetch_anima_tokenizers.py)")
    _require(CONVERTER,      "Anima converter script in the diffusers checkout")

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CONVERTER),
        "--transformer_ckpt_path",  str(ANIMA_DIT),
        "--text_encoder_ckpt_path", str(QWEN3_06B),
        "--vae_ckpt_path",          str(QWEN_IMAGE_VAE),
        "--qwen_tokenizer_path",    str(QWEN_TOKENIZER),
        "--t5_tokenizer_path",      str(T5_TOKENIZER),
        "--output_path",            str(OUTPUT_DIR),
        "--save_pipeline",
        "--dtype", "bf16",
    ]

    print("Running converter:")
    for tok in cmd:
        print(f"  {tok}")
    print()

    # cwd=DIFFUSERS_SCRIPTS so the converter's `from convert_cosmos_to_diffusers
    # import convert_transformer` resolves correctly.
    proc = subprocess.run(cmd, cwd=str(DIFFUSERS_SCRIPTS), check=False)
    if proc.returncode != 0:
        return proc.returncode

    print()
    print(f"OK. Output at: {OUTPUT_DIR}")
    print("Inspect with: dir D:\\models\\diffusers\\anima\\anima-base-v1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
