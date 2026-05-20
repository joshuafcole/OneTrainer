"""Load Anima components via OneTrainer's AnimaModelLoader, then plug them
into an AnimaModularPipeline and generate one image.

This is the integration-side counterpart of
``smoke_test_anima_inference.py``: that script proves the converted
checkpoint + upstream pipeline work end-to-end; this one proves our
loader can stand in for the upstream component-loading code without
changing the output.

If both produce a plausible calico cat, the loader is correct enough
to start building the trainer scaffolding on top of.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima_loader.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make `modules.*` imports resolve relative to the repo root regardless
# of the cwd this script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
from diffusers.modular_pipelines.anima import AnimaAutoBlocks

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


CHECKPOINT = Path("D:/models/diffusers/anima/anima-base-v1.0")
DEFAULT_PROMPT = (
    "a photograph of a calico cat sitting on a sun-warmed wooden porch, "
    "soft afternoon light, shallow depth of field, 35mm film"
)
DEFAULT_OUT = Path("D:/models/diffusers/anima/anima-base-v1.0/_smoke_test_loader.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")

    # All components in bf16 -- matches the upstream smoke test for
    # apples-to-apples comparison.
    weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
    model_names = ModelNames(base_model=str(CHECKPOINT))
    quantization = QuantizationConfig.default_values()

    print(f"loading via AnimaModelLoader from {CHECKPOINT}")
    t0 = time.perf_counter()
    model = AnimaModel(model_type=ModelType.ANIMA)
    loader = AnimaModelLoader()
    loader.load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=model_names,
        weight_dtypes=weight_dtypes,
        quantization=quantization,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print("  components on model:")
    for name in ("tokenizer", "t5_tokenizer", "noise_scheduler", "text_encoder",
                 "text_conditioner", "vae", "transformer"):
        comp = getattr(model, name)
        if comp is None:
            print(f"    {name:20s} -> NONE")
            return 2
        # transformers/diffusers parameter modules report .device/.dtype;
        # tokenizers don't, so guard on attribute presence.
        device = getattr(comp, "device", "n/a")
        dtype = getattr(comp, "dtype", "n/a")
        print(f"    {name:20s} -> {type(comp).__name__:32s}  device={device}  dtype={dtype}")

    # Hand the loaded components to a fresh AnimaModularPipeline. This
    # exercises AnimaModel.create_pipeline() too, but going via the raw
    # builder gives slightly better error messages if the wiring is off.
    print(f"building AnimaModularPipeline + moving to {args.device}")
    pipe = AnimaAutoBlocks().init_pipeline()
    pipe.update_components(
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        t5_tokenizer=model.t5_tokenizer,
        text_conditioner=model.text_conditioner,
        transformer=model.transformer,
        vae=model.vae,
        scheduler=model.noise_scheduler,
    )
    pipe.to(device=args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"sampling: prompt={args.prompt!r}")
    print(f"          {args.width}x{args.height}, {args.num_inference_steps} steps")
    t0 = time.perf_counter()
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
    )
    elapsed = time.perf_counter() - t0
    print(f"  sampled in {elapsed:.1f}s ({elapsed / args.num_inference_steps:.2f}s/step)")

    images = getattr(output, "images", None)
    if images is None and hasattr(output, "__getitem__"):
        images = output["images"]
    if images is None:
        print(f"  UNEXPECTED output shape: {type(output)}")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
