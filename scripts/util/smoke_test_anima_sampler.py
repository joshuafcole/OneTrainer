"""Generate an image via OneTrainer's AnimaSampler.

The third smoke test in the Anima integration triangle:

  smoke_test_anima_inference.py  -- upstream AnimaModularPipeline only
  smoke_test_anima_loader.py     -- our loader + upstream pipeline
  smoke_test_anima_sampler.py    -- our loader + OUR sampler  <-- this one

If all three produce a calico cat we are sampler-correct, which
unlocks LoRA training (the training step uses the same chain
without the denoising loop).

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima_sampler.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.modelSampler.AnimaSampler import AnimaSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


CHECKPOINT = Path("D:/models/diffusers/anima/anima-base-v1.0")
DEFAULT_PROMPT = (
    "a photograph of a calico cat sitting on a sun-warmed wooden porch, "
    "soft afternoon light, shallow depth of field, 35mm film"
)
DEFAULT_OUT = Path("D:/models/diffusers/anima/anima-base-v1.0/_smoke_test_sampler.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")

    weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
    model_names = ModelNames(base_model=str(CHECKPOINT))
    quantization = QuantizationConfig.default_values()

    print("loading via AnimaModelLoader")
    t0 = time.perf_counter()
    model = AnimaModel(model_type=ModelType.ANIMA)
    AnimaModelLoader().load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=model_names,
        weight_dtypes=weight_dtypes,
        quantization=quantization,
    )
    # train_dtype defaults to FLOAT_32 on a freshly-constructed model.
    # The sampler casts intermediates with .to(model.train_dtype.torch_dtype()),
    # and we loaded the transformer in bf16, so train_dtype must agree.
    model.train_dtype = DataType.BFLOAT_16
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print("building AnimaSampler")
    train_device = torch.device(args.device)
    sampler = AnimaSampler(
        train_device=train_device,
        # Anima fits entirely in 32 GB VRAM; cycle stages on/off train
        # device but keep everything on GPU to match the upstream timing.
        temp_device=train_device,
        model=model,
        model_type=ModelType.ANIMA,
    )

    sample_config = SampleConfig.default_values()
    sample_config.prompt = args.prompt
    sample_config.negative_prompt = args.negative_prompt
    sample_config.height = args.height
    sample_config.width = args.width
    sample_config.diffusion_steps = args.num_inference_steps
    sample_config.cfg_scale = args.cfg_scale
    sample_config.seed = args.seed
    sample_config.random_seed = False
    # NoiseScheduler is a SampleConfig hint -- AnimaSampler always uses
    # the model's FlowMatchEulerDiscreteScheduler (deep-copied per call)
    # regardless of this value, matching ZImageSampler.
    sample_config.noise_scheduler = NoiseScheduler.EULER

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"sampling -> {out_path}")
    print(f"  prompt={args.prompt!r}")
    print(f"  {args.width}x{args.height}, {args.num_inference_steps} steps, cfg={args.cfg_scale}")

    captured: list = []

    def on_sample(out):
        if out.file_type == FileType.IMAGE:
            captured.append(out.data)

    t0 = time.perf_counter()
    sampler.sample(
        sample_config=sample_config,
        destination=str(out_path),
        image_format=ImageFormat.PNG,
        on_sample=on_sample,
    )
    elapsed = time.perf_counter() - t0
    print(f"  sampled in {elapsed:.1f}s ({elapsed / args.num_inference_steps:.2f}s/step)")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
