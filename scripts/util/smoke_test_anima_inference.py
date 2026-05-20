"""Generate one image from the converted Anima checkpoint using the
upstream AnimaModularPipeline directly.

The goal is to validate, before we write any OneTrainer wrapper code,
that:

  1. ModularPipeline.from_pretrained(...) can read our
     modular_model_index.json and instantiate the full pipeline.
  2. The converted on-disk checkpoint actually generates a coherent
     image (i.e., the conversion did not silently corrupt weights).
  3. We know which kwargs the pipeline accepts (height, width,
     num_inference_steps, guidance_scale, etc.) -- those tell us the
     argument shape AnimaSampler will need to emit.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima_inference.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from diffusers.modular_pipelines import ModularPipeline


CHECKPOINT = Path("D:/models/diffusers/anima/anima-base-v1.0")
DEFAULT_PROMPT = (
    "a photograph of a calico cat sitting on a sun-warmed wooden porch, "
    "soft afternoon light, shallow depth of field, 35mm film"
)
DEFAULT_OUT = Path("D:/models/diffusers/anima/anima-base-v1.0/_smoke_test.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}  (run convert_anima_base_v1.py first)")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"loading pipeline blocks from {CHECKPOINT}")
    t0 = time.perf_counter()
    # ModularPipeline.from_pretrained reads modular_model_index.json and
    # records ComponentSpec entries, but does NOT instantiate the
    # from_pretrained-typed components yet (only the from_config ones,
    # i.e. guider + image_processor). Components are loaded explicitly
    # below via load_components -- that's where torch_dtype takes effect.
    pipe = ModularPipeline.from_pretrained(str(CHECKPOINT))
    print(f"  loaded blocks in {time.perf_counter() - t0:.1f}s")
    print(f"  pipeline class: {type(pipe).__name__}")

    print(f"loading components ({args.dtype}) ...")
    t0 = time.perf_counter()
    pipe.load_components(torch_dtype=dtype)
    print(f"  loaded components in {time.perf_counter() - t0:.1f}s")

    # Components manifest. Useful debug if we can't tell what's loaded.
    print("  components:")
    for name in ("text_encoder", "tokenizer", "t5_tokenizer", "text_conditioner",
                 "transformer", "vae", "scheduler", "guider", "image_processor"):
        comp = getattr(pipe, name, None)
        print(f"    {name:18s} -> {type(comp).__name__ if comp is not None else 'NONE'}")

    # The guider's guidance_scale was set from the ComponentSpec config
    # (default 4.0). Override here if the user asked for a different cfg.
    if args.guidance_scale != 4.0:
        from diffusers.guiders import ClassifierFreeGuidance
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=args.guidance_scale))
        print(f"  overrode guider cfg -> {args.guidance_scale}")

    print(f"moving pipeline to {args.device}")
    pipe.to(device=args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"sampling: prompt={args.prompt!r}")
    print(f"          {args.width}x{args.height}, {args.num_inference_steps} steps, cfg={args.guidance_scale}")
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

    # Modular pipeline outputs are a PipelineState wrapping intermediates; the
    # AnimaAutoBlocks output spec exposes "images" as the final field.
    images = getattr(output, "images", None)
    if images is None:
        # Fallback: dict-style access; PipelineState supports both.
        images = output["images"] if hasattr(output, "__getitem__") else None
    if images is None:
        print(f"  UNEXPECTED output shape: {type(output)} dir={dir(output)}")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
