"""Experiment #1 for slider-LoRA support: does Anima respond to conditioning?

The whole Concept-Sliders objective rests on the frozen base supplying a
non-trivial guidance direction  v(c+) - v(c-)  (see docs/slider_lora.md S2).
If Anima were guidance-distilled / conditioning-insensitive, that difference
would collapse to ~0 and the velocity-space slider objective

    v*(x_t, c_t, t) = v(c_t) + eta * ( v(c+) - v(c-) )

would have nothing to learn. This script measures the signal directly, using
OneTrainer's own AnimaModel.encode_text + Cosmos transformer forward -- the
exact path slider training will use -- so the result transfers 1:1.

For each (target c_t, positive c+, negative c-) triple and each normalized
timestep sigma, it reports:

  rel_guidance = || v(c+) - v(c-) || / || v(c_t) ||      (relative signal size)
  cos_align    = cos( v(c+) - v(c-),  v(c+) - v(c_t) )   (does + move toward c+?)
  rel_pos/neg  = || v(c+/-) - v(c_t) || / || v(c_t) ||    (each pole differs from neutral)

Verdict: a healthy, conditioning-responsive base shows rel_guidance well above
noise (rule of thumb > ~2%) across timesteps, with positive cos_align. A flat
~0 signal is the red flag that the slider method needs rethinking for Anima.

Run with the OneTrainer venv active (GPU strongly recommended):
    venv/bin/python scripts/util/probe_anima_cfg_response.py --checkpoint /path/to/anima
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

VAE_SCALE_FACTOR = 8

# Default triples mirror the Concept Sliders README age example, plus a couple
# of distinct attributes so a per-attribute response pattern is visible.
DEFAULT_TRIPLES = [
    ("a person", "an old person", "a young person"),
    ("a portrait photo", "a smiling portrait photo", "a frowning portrait photo"),
    ("a landscape", "a bright sunny landscape", "a dark gloomy landscape"),
]
DEFAULT_SIGMAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def _ensure_contexts(model: AnimaModel):
    """A raw-loaded model hasn't been through setup_optimizations, which is
    where the autocast contexts/dtypes are normally set. Fill in no-op
    defaults so encode_text + transformer run without a full trainer."""
    for attr in ("autocast_context", "text_encoder_autocast_context"):
        if getattr(model, attr, None) is None:
            setattr(model, attr, contextlib.nullcontext())


def _latent_channels(model: AnimaModel) -> int:
    cfg = getattr(model.transformer, "config", None)
    return int(getattr(cfg, "in_channels", 16) or 16)


@torch.no_grad()
def _velocity(model, x_t, t_norm, encoder_hidden_states, compute_dtype, device):
    # The transformer can carry mixed param dtypes (e.g. bf16 linears with a
    # few fp32 buffers), so model.transformer.dtype is unreliable. Run under
    # autocast -- exactly like BaseAnimaSetup.predict does -- and feed every
    # input (incl. padding_mask) in the compute dtype so the matmuls agree.
    h_pix = x_t.shape[-2] * VAE_SCALE_FACTOR
    w_pix = x_t.shape[-1] * VAE_SCALE_FACTOR
    padding_mask = x_t.new_zeros((1, 1, h_pix, w_pix), dtype=compute_dtype)
    autocast = (
        torch.autocast(device_type=device.type, dtype=compute_dtype)
        if compute_dtype != torch.float32
        else contextlib.nullcontext()
    )
    with autocast:
        out = model.transformer(
            hidden_states=x_t.to(compute_dtype),
            timestep=t_norm.to(compute_dtype),
            encoder_hidden_states=encoder_hidden_states.to(compute_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
    return out.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="D:/models/diffusers/anima/anima-base-v1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="mean rel_guidance below this flags a weak/distilled response")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    weight_dtypes = ModelWeightDtypes.from_single_dtype(
        {"fp32": DataType.FLOAT_32, "fp16": DataType.FLOAT_16, "bf16": DataType.BFLOAT_16}[args.dtype]
    )

    print(f"loading Anima via AnimaModelLoader from {args.checkpoint} ({args.dtype})")
    model = AnimaModel(model_type=ModelType.ANIMA)
    AnimaModelLoader().load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=ModelNames(base_model=str(args.checkpoint)),
        weight_dtypes=weight_dtypes,
        quantization=QuantizationConfig.default_values(),
    )
    _ensure_contexts(model)

    device = torch.device(args.device)
    model.text_encoder_to(device)
    model.text_conditioner.to(device)
    model.transformer.to(device)
    model.eval()

    in_ch = _latent_channels(model)
    h_lat, w_lat = args.height // VAE_SCALE_FACTOR, args.width // VAE_SCALE_FACTOR
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"latent: (1, {in_ch}, 1, {h_lat}, {w_lat}); sigmas={DEFAULT_SIGMAS}\n")

    all_rel = []
    for c_t, c_plus, c_minus in DEFAULT_TRIPLES:
        print(f"=== target={c_t!r}  + {c_plus!r}  - {c_minus!r} ===")
        eh_t = model.encode_text(train_device=device, text=c_t)
        eh_p = model.encode_text(train_device=device, text=c_plus)
        eh_n = model.encode_text(train_device=device, text=c_minus)

        print(f"  {'sigma':>6} {'rel_guid':>9} {'cos_align':>10} {'rel_pos':>8} {'rel_neg':>8}")
        for sigma in DEFAULT_SIGMAS:
            # A fresh random x_t per (triple, sigma); only conditioning varies.
            x_t = torch.randn((1, in_ch, 1, h_lat, w_lat), generator=gen).to(device)
            t_norm = torch.full((1,), float(sigma), device=device)

            v_t = _velocity(model, x_t, t_norm, eh_t, dtype, device)
            v_p = _velocity(model, x_t, t_norm, eh_p, dtype, device)
            v_n = _velocity(model, x_t, t_norm, eh_n, dtype, device)

            guidance = v_p - v_n
            nt = v_t.norm().clamp_min(1e-8)
            rel_guid = (guidance.norm() / nt).item()
            rel_pos = ((v_p - v_t).norm() / nt).item()
            rel_neg = ((v_n - v_t).norm() / nt).item()
            cos_align = torch.nn.functional.cosine_similarity(
                guidance.flatten(), (v_p - v_t).flatten(), dim=0
            ).item()
            all_rel.append(rel_guid)
            print(f"  {sigma:>6.2f} {rel_guid:>9.4f} {cos_align:>10.3f} {rel_pos:>8.4f} {rel_neg:>8.4f}")
        print()

    mean_rel = sum(all_rel) / len(all_rel)
    print(f"mean rel_guidance across all triples/timesteps = {mean_rel:.4f}")
    if mean_rel >= args.threshold:
        print(f"VERDICT: RESPONSIVE (>= {args.threshold}). Velocity-space slider objective is viable for Anima.")
        return 0
    print(f"VERDICT: WEAK RESPONSE (< {args.threshold}). Investigate before building the slider objective:")
    print("  - is the base guidance-distilled? does encode_text differ across prompts?")
    print("  - try larger prompt contrasts, fp32, or the unconditional/empty-prompt baseline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
