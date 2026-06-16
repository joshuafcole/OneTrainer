"""Experiment #1 for slider-LoRA support: does Anima respond to conditioning?

The whole Concept-Sliders objective rests on the frozen base supplying a
non-trivial guidance direction  v(c+) - v(c-)  (see docs/slider_lora.md S2).
If Anima were guidance-distilled / conditioning-insensitive, that difference
would collapse to ~0 and the velocity-space slider objective

    v*(x_t, c_t, t) = v(c_t) + eta * ( v(c+) - v(c-) )

would have nothing to learn. This measures the signal directly via OneTrainer's
own AnimaModel.encode_text + Cosmos transformer forward -- the exact path slider
training uses -- so the result transfers 1:1.

Two things matter and v1's pure-Gaussian probe under-measured both:

  * x_t must be ON the model's manifold. By default we integrate the conditional
    flow-matching ODE from ~noise down to each sigma (Euler on dx/dsigma = v),
    so v is measured where the model actually operates. --random-latent restores
    the old off-manifold Gaussian behaviour for comparison.
  * we need a calibration baseline for "does this model use conditioning AT ALL".
    That is rel_cfg = ||v(c_t) - v(empty)|| / ||v(c_t)||. If rel_cfg is ~0 the
    model is conditioning-blind (distilled/broken) and sliders won't work; if it
    is clearly nonzero, the model responds and the slider is a tuning problem.

Per (triple c_t/c+/c-, sigma) it reports:

  rel_guid  = || v(c+) - v(c-) || / || v(c_t) ||     attribute guidance magnitude
  rel_cfg   = || v(c_t) - v(empty) || / || v(c_t) || overall conditioning strength
  ratio     = rel_guid / rel_cfg                     attribute share of conditioning
  cos_align = cos( v(c+) - v(c-), v(c+) - v(c_t) )    is + a coherent pole?

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

# Strong, polar prompt pairs (target c_t / positive c+ / negative c-), in the
# spirit of the Concept Sliders README. Verbose, lexically opposite poles give
# the cleanest guidance direction.
DEFAULT_TRIPLES = [
    ("a portrait photo of a person",
     "a portrait photo of a very old elderly person, deeply wrinkled, aged, grey hair",
     "a portrait photo of a very young child, youthful, smooth skin, baby face"),
    ("a portrait photo of a person",
     "a portrait photo of a person broadly smiling, joyful, happy, laughing",
     "a portrait photo of a person angry, frowning, furious, scowling"),
    ("a landscape photograph",
     "a landscape photograph in bright sunny daylight, clear blue sky, vivid",
     "a landscape photograph at dark stormy night, gloomy, overcast, pitch black"),
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


@torch.no_grad()
def _onmanifold_xt(model, eh_cond, sigma, compute_dtype, device, shape, gen, step_size):
    """Integrate the conditional flow-matching ODE from ~noise (sigma~1) down to
    the target sigma under eh_cond, so x_t lands on the model's own trajectory
    rather than off-manifold Gaussian noise. Euler on dx/dsigma = v, with
    v = noise - image (no sign flip), matching BaseAnimaSetup."""
    x = torch.randn(shape, generator=gen).to(device)
    n = max(1, round((0.999 - sigma) / step_size))
    sigmas = torch.linspace(0.999, sigma, n + 1)
    for i in range(n):
        s = sigmas[i].item()
        t_norm = torch.full((1,), s, device=device)
        v = _velocity(model, x, t_norm, eh_cond, compute_dtype, device)
        x = x + (sigmas[i + 1].item() - s) * v
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="D:/models/diffusers/anima/anima-base-v1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step_size", type=float, default=0.05, help="Euler step for the on-manifold trajectory")
    ap.add_argument("--random-latent", action="store_true",
                    help="use off-manifold Gaussian x_t (the old v1 behaviour) instead of a denoised trajectory")
    ap.add_argument("--cfg-threshold", type=float, default=0.02,
                    help="mean rel_cfg below this means the base is conditioning-blind (sliders not viable)")
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
    shape = (1, in_ch, 1, h_lat, w_lat)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    latent_mode = "off-manifold Gaussian" if args.random_latent else f"on-manifold (Euler, step={args.step_size})"

    print(f"latent: {shape}; sigmas={DEFAULT_SIGMAS}; x_t={latent_mode}\n")

    eh_empty = model.encode_text(train_device=device, text="")

    guid, cfg, cos = [], [], []
    for c_t, c_plus, c_minus in DEFAULT_TRIPLES:
        print(f"=== target={c_t!r}\n    + {c_plus!r}\n    - {c_minus!r} ===")
        eh_t = model.encode_text(train_device=device, text=c_t)
        eh_p = model.encode_text(train_device=device, text=c_plus)
        eh_n = model.encode_text(train_device=device, text=c_minus)

        print(f"  {'sigma':>6} {'rel_guid':>9} {'rel_cfg':>8} {'ratio':>6} {'cos_align':>10}")
        for sigma in DEFAULT_SIGMAS:
            if args.random_latent:
                x_t = torch.randn(shape, generator=gen).to(device)
            else:
                # Trajectory generated under the neutral target conditioning.
                x_t = _onmanifold_xt(model, eh_t, sigma, dtype, device, shape, gen, args.step_size)
            t_norm = torch.full((1,), float(sigma), device=device)

            v_t = _velocity(model, x_t, t_norm, eh_t, dtype, device)
            v_p = _velocity(model, x_t, t_norm, eh_p, dtype, device)
            v_n = _velocity(model, x_t, t_norm, eh_n, dtype, device)
            v_e = _velocity(model, x_t, t_norm, eh_empty, dtype, device)

            nt = v_t.norm().clamp_min(1e-8)
            rel_guid = ((v_p - v_n).norm() / nt).item()
            rel_cfg = ((v_t - v_e).norm() / nt).item()
            ratio = rel_guid / max(rel_cfg, 1e-8)
            cos_align = torch.nn.functional.cosine_similarity(
                (v_p - v_n).flatten(), (v_p - v_t).flatten(), dim=0
            ).item()
            guid.append(rel_guid); cfg.append(rel_cfg); cos.append(cos_align)
            print(f"  {sigma:>6.2f} {rel_guid:>9.4f} {rel_cfg:>8.4f} {ratio:>6.2f} {cos_align:>10.3f}")
        print()

    mean_guid = sum(guid) / len(guid)
    mean_cfg = sum(cfg) / len(cfg)
    mean_cos = sum(cos) / len(cos)
    print(f"means: rel_guid={mean_guid:.4f}  rel_cfg={mean_cfg:.4f}  cos_align={mean_cos:.3f}\n")

    if mean_cfg < args.cfg_threshold:
        print(f"VERDICT: CONDITIONING-BLIND (mean rel_cfg={mean_cfg:.4f} < {args.cfg_threshold}).")
        print("  The base barely distinguishes a real prompt from the empty prompt -- the")
        print("  guidance-difference objective has little to work with. Re-check the encode")
        print("  path / try fp32, and reconsider the slider design before building S3.")
        return 1
    if mean_cos <= 0.0:
        print(f"VERDICT: RESPONSIVE but INCOHERENT direction (rel_cfg={mean_cfg:.4f}, cos={mean_cos:.3f}).")
        print("  The model uses conditioning, but c+/c- don't form a clean axis -- revisit")
        print("  prompt-pair design (more polar, add preservation prompts) before S3.")
        return 1
    print(f"VERDICT: VIABLE. The base responds to conditioning (rel_cfg={mean_cfg:.4f}) with a")
    print(f"  coherent attribute axis (cos={mean_cos:.3f}). The raw attribute gap rel_guid=")
    print(f"  {mean_guid:.4f} is amplified by eta (~3-4) and accumulated over training, so the")
    print("  velocity-space slider objective can be built; eta/steps tune the final strength.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
