"""Per-layer activation-weighted contribution of N adapters, as JSON on stdout.

`block_gram` answers *how big is this layer's delta and which way does it
point*. That is a fact about the weights alone. This answers the question a
merge actually cares about: **how much does this layer's delta move the model
on the inputs the model actually sees**, which is

    contribution(i, l, p) = || dW[i,l] @ X[l,p]^T ||_F^2 / rows(X[l,p])

for adapter ``i``, layer ``l``, and the activations ``X[l,p]`` that prompt line
``p`` produces at that layer. It is the term in
``alpha[i,l] ~ sum_good ||dW[i,l] x|| - lambda * sum_bad ||dW[i,l] x||`` that
the Gram cannot supply.

Why it is a different measurement, and when it isn't
----------------------------------------------------
Under isotropic activations the two coincide *exactly*:

    E || dW x ||^2 = ||dW||_F^2 * E||x||^2 / d_in

so if activations were white, this script would be an expensive way to reprint
the Gram diagonal and every merge coefficient derived from it would be
derivable from weights alone. They are not white, and the size of the gap is
the entire reason to run this. So the gap is emitted, not assumed: every layer
carries ``frobenius_sq`` (the isotropic prediction) beside ``contribution_sq``
(the measurement), and the analysis can divide. A run where the two are
proportional to within noise is a *result* -- it says the activation term buys
nothing here and the cheap route was sufficient.

The base model is the shared reference
--------------------------------------
``X`` is captured from the **base** model, once per prompt line, with no
adapter applied. Two consequences, both deliberate:

- It is a first-order approximation. The true contribution of adapter ``i``
  inside a merge uses the *merged* model's activations, which differ. For the
  small deltas this initiative works with (a few percent of the base weight)
  the difference is second-order, and paying it buys the thing that matters
  more: comparability.
- It is what makes checkpoints comparable at all. Every adapter is scored
  against the identical ``X``, down to the same sampled token rows, so the
  *ratio* between two adapters at one layer is far more accurate than either
  number alone -- the sampling noise is common-mode and divides out. Absolute
  contributions carry the token-sampling error; relative ones very nearly do
  not. Compare adapters, not layers.

Two phases, and only the second needs the adapters
--------------------------------------------------
``capture_activations`` loads Anima, hooks the LoRA-target linears, runs the
sampler once per prompt line, and keeps a bounded random subsample of each
layer's input rows. The model is then released. ``score`` takes those
activations and the adapters and is pure linear algebra -- no model, no GPU, no
OneTrainer imports -- which is why it is the part that has tests.

Usage::

    python scripts/util/block_contribution.py a.safetensors b.safetensors ... \
        --prompt "..." --prompt "..." [--base-model PATH] [--max-tokens N] \
        [--granularity LEVEL] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


class ContributionError(Exception):
    """The adapters and the activations do not describe the same model."""


# Higher than block_gram's ceiling on purpose: that one is quadratic in the
# adapter count (every pair's inner product), this one is linear (each adapter
# against one activation matrix). The whole c016 mixture is 65 saves, which the
# Gram refuses and this does not.
MAX_ADAPTERS = 128

#: Anima's own sampler resolution. Kept as the default so a capture run and a
#: rehearsal render see the same token geometry.
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_DIFFUSION_STEPS = 25
DEFAULT_CAPTURE_STEPS = 4
DEFAULT_MAX_TOKENS = 256
DEFAULT_BASE_MODEL = "D:/models/diffusers/anima/anima-base-v1.0"


# --------------------------------------------------------------------------
# Phase 2: scoring. Pure, tested, model-free.
# --------------------------------------------------------------------------


def score(
    paths: Sequence[Path],
    activations: Mapping[str, Mapping[str, torch.Tensor]],
    prompts: Sequence[str],
    config_path: str | None = None,
    granularity: str | None = None,
    capture: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """``{layers, per-layer contribution per (prompt, adapter), Frobenius}``.

    ``activations`` is keyed by prompt label then by layer prefix; each value is
    a 2-D ``(rows, in_features)`` matrix of that layer's captured inputs. One
    layer's deltas are materialised at a time and reduced to scalars
    immediately, as in ``block_gram`` and for the same reason.
    """
    if not paths:
        raise ContributionError("need at least 1 adapter, got 0")
    if len(paths) > MAX_ADAPTERS:
        raise ContributionError(f"at most {MAX_ADAPTERS} adapters, got {len(paths)}")
    if len(prompts) != len(activations):
        raise ContributionError(
            f"{len(prompts)} prompt(s) but {len(activations)} activation set(s) — "
            "the labels and the prompts are positional and must correspond"
        )

    loaded = [lora_soup.load_lora(p, 1.0) for p in paths]
    n = len(loaded)

    shared_set = set(loaded[0].layers)
    for other in loaded[1:]:
        shared_set &= set(other.layers)
    # A layer nobody captured cannot be scored, so it is not shared for this
    # purpose either -- but it is reported below rather than quietly dropped.
    captured_set: set[str] = set()
    for per_layer in activations.values():
        captured_set |= set(per_layer)
    scorable = sorted(shared_set & captured_set)
    if not scorable:
        raise ContributionError(
            "no layer is both shared by all adapters and present in the capture — "
            "these adapters and these activations are not about the same model"
        )
    dropped = {
        str(p): sorted(set(adapter.layers) - shared_set)
        for p, adapter in zip(paths, loaded, strict=True)
        if set(adapter.layers) - shared_set
    }
    uncaptured = sorted(shared_set - captured_set)
    unscored = sorted(captured_set - shared_set)

    config = block_groups.load_groups(config_path)
    fitted = block_groups.fit(scorable, config, granularity)
    group_of = {
        prefix: group for group, members in fitted.groups.items() for prefix in members
    }

    labels = list(activations)
    layers: list[dict[str, object]] = []
    for prefix in scorable:
        # float64 throughout: a contribution is a sum of hundreds of thousands
        # of squared terms spanning several orders of magnitude, and the ratio
        # against the Frobenius prediction is precisely where a lost tail turns
        # into a spurious few-percent "activation effect".
        deltas = [
            adapter.layers[prefix].delta().to(torch.float64) for adapter in loaded
        ]
        widths = {tuple(d.shape) for d in deltas}
        if len(widths) > 1:
            raise ContributionError(
                f"layer {prefix!r} has different sizes across adapters ({sorted(widths)}) "
                "— they were trained against different base geometry"
            )
        in_features = deltas[0].shape[1]
        frobenius_sq = [float((d * d).sum()) for d in deltas]

        energy: list[float] = []
        rows: list[int] = []
        contribution_sq: list[list[float]] = []
        for label in labels:
            x = activations[label].get(prefix)
            if x is None:
                # Present for some prompt lines and not others: a capture that
                # was interrupted, or a layer the model only reaches on some
                # inputs. Either way a zero here would read as "contributes
                # nothing", which is the one thing it does not mean.
                raise ContributionError(
                    f"layer {prefix!r} was captured for some prompts but not {label!r}"
                )
            if x.ndim != 2:
                raise ContributionError(
                    f"layer {prefix!r} activations for {label!r} are {x.ndim}-D, expected 2-D "
                    "(rows, in_features)"
                )
            if x.shape[1] != in_features:
                raise ContributionError(
                    f"layer {prefix!r} takes {in_features} inputs but the capture for "
                    f"{label!r} has {x.shape[1]} — the adapters and the capture are not "
                    "about the same base model"
                )
            if x.shape[0] == 0:
                raise ContributionError(
                    f"layer {prefix!r} captured no rows for {label!r}"
                )
            xd = x.to(torch.float64)
            count = int(xd.shape[0])
            rows.append(count)
            energy.append(float((xd * xd).sum()) / count)
            contribution_sq.append(
                [float(((d @ xd.T) ** 2).sum()) / count for d in deltas]
            )

        layers.append({
            "layer": prefix,
            "group": group_of[prefix],
            "in_features": int(in_features),
            "frobenius_sq": frobenius_sq,
            "rows": rows,
            "activation_energy": energy,
            "contribution_sq": contribution_sq,
        })

    return {
        "paths": [str(p) for p in paths],
        "adapter_count": n,
        "prompts": list(prompts),
        "prompt_labels": labels,
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "scored_layer_count": len(scorable),
        # Named rather than intersected away, as in block_gram: a table over a
        # quietly reduced key set describes a different object than the caller
        # asked about.
        "dropped_layers": dropped,
        "uncaptured_layers": uncaptured,
        "unscored_layers": unscored,
        "unrecognized_parts": list(fitted.unrecognized_parts),
        "capture": dict(capture or {}),
        "layers": layers,
    }


# --------------------------------------------------------------------------
# Phase 1: capture. Model-shaped, and only smoke-testable on a box with Anima.
# --------------------------------------------------------------------------


def resolve_modules(
    root: torch.nn.Module,
    layer_prefixes: Iterable[str],
    root_name: str = "transformer",
) -> tuple[dict[str, torch.nn.Module], list[str]]:
    """Map adapter layer prefixes onto the base modules they wrap.

    ``LoRAModuleWrapper(model.transformer, "transformer", ...)`` names its
    layers ``<root_name>.<named_modules path>`` with ``.checkpoint.`` elided,
    so this inverts exactly that. Unmatched prefixes are **returned, not
    raised**: an adapter that also targets the text encoder is a normal thing
    to hand this script, and the right response is to score what can be scored
    and say what could not.

    ``Linear`` only, where the wrapper also takes ``Conv2d``. A conv delta is
    flattened to ``(out, in * kernel)`` while the captured rows are plain
    ``in_channels``, so the two would not multiply -- and a conv contribution
    needs the unfolded patches, not the module input. Anima is attention-only,
    so this costs nothing today; leaving conv unresolved names the gap instead
    of producing a shape error from inside the scoring loop.
    """
    targets: dict[str, torch.nn.Module] = {}
    for name, child in root.named_modules():
        if isinstance(child, torch.nn.Linear):
            targets[name.replace(".checkpoint.", ".")] = child

    head = root_name + "." if root_name else ""
    resolved: dict[str, torch.nn.Module] = {}
    unmatched: list[str] = []
    for prefix in layer_prefixes:
        if head and not prefix.startswith(head):
            unmatched.append(prefix)
            continue
        module = targets.get(prefix[len(head):])
        if module is None:
            unmatched.append(prefix)
        else:
            resolved[prefix] = module
    return resolved, sorted(unmatched)


class ActivationCapture:
    """Bounded, deterministic subsample of each hooked layer's input rows.

    Two things it is careful about:

    - **Which forward.** With CFG enabled the sampler makes two transformer
      calls per denoise step, conditional first (``AnimaSampler.__sample_base``).
      Only the conditional one is the prompt's activations, so only call
      ``index % calls_per_step == 0`` is recorded. The count is checked against
      the step count at the end, so a sampler change that breaks this
      assumption fails loudly instead of quietly averaging in the negative
      prompt.
    - **How many rows.** A per-``(layer, call)`` quota, recorded once and
      ignored on any repeat of the same call index. Gradient checkpointing can
      re-run a block's forward; without the guard those tokens would be counted
      twice, and only for the checkpointed layers.
    """

    def __init__(
        self,
        capture_calls: Sequence[int],
        max_tokens: int,
        seed: int,
        store_dtype: torch.dtype = torch.float32,
    ):
        self.capture_calls = list(capture_calls)
        self.max_tokens = max_tokens
        self.seed = seed
        self.store_dtype = store_dtype
        self.call_index = -1
        self.rows: dict[str, list[torch.Tensor]] = {}
        self._seen: set[tuple[str, int]] = set()
        self._handles: list[object] = []
        # ceil, so the quota over all capture calls is never short of the cap;
        # the concatenated result is trimmed to exactly max_tokens.
        n = max(1, len(self.capture_calls))
        self.per_call = max(1, -(-max_tokens // n))

    def _quota_generator(self, prefix: str, call: int) -> torch.Generator:
        """Deterministic per (layer, call), so a re-run picks the same tokens."""
        generator = torch.Generator()
        # Python's str hash is salted per process; a stable digest is not.
        digest = sum((i + 1) * b for i, b in enumerate(prefix.encode())) & 0xFFFFFFFF
        generator.manual_seed((self.seed * 1_000_003 + digest * 31 + call) % (2**63 - 1))
        return generator

    def attach(self, modules: Mapping[str, torch.nn.Module], root: torch.nn.Module):
        def on_root(_module, _args):
            self.call_index += 1

        self._handles.append(root.register_forward_pre_hook(on_root))

        for prefix, module in modules.items():
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(prefix))
            )

    def _make_hook(self, prefix: str):
        def hook(_module, args):
            call = self.call_index
            if call not in self.capture_calls:
                return
            key = (prefix, call)
            if key in self._seen:
                return
            self._seen.add(key)
            if not args:
                # A pre-hook without ``with_kwargs`` sees only positional args,
                # and the sampler calls the *transformer* entirely by keyword.
                # Linears are called positionally, so this is a diagnosis rather
                # than a case to handle -- but an IndexError three frames deep
                # inside diffusers is not one.
                raise ContributionError(
                    f"layer {prefix!r} was called with keyword arguments only; the "
                    "capture hook reads args[0] and cannot see its input"
                )
            x = args[0]
            flat = x.reshape(-1, x.shape[-1]).detach()
            take = min(self.per_call, flat.shape[0])
            index = torch.randperm(
                flat.shape[0], generator=self._quota_generator(prefix, call)
            )[:take]
            picked = flat[index.to(flat.device)].to(
                device="cpu", dtype=self.store_dtype, copy=True
            )
            self.rows.setdefault(prefix, []).append(picked)

        return hook

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        return {
            prefix: torch.cat(chunks, dim=0)[: self.max_tokens]
            for prefix, chunks in self.rows.items()
        }


def capture_activations(
    prompts: Sequence[str],
    layer_prefixes: Sequence[str],
    base_model: str,
    device: str = "cuda",
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    capture_steps: Sequence[int] | None = None,
    cfg_scale: float = 1.0,
    negative_prompt: str = "",
    seed: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object], list[str]]:
    """Run the base model once per prompt and keep a subsample of each layer's inputs.

    Everything Anima-shaped is imported here rather than at module scope, so
    ``score`` and its tests do not need OneTrainer's dependency tree.
    """
    import tempfile

    from modules.model.AnimaModel import AnimaModel
    from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
    from modules.modelSampler.AnimaSampler import AnimaSampler
    from modules.util.config.SampleConfig import SampleConfig
    from modules.util.config.TrainConfig import QuantizationConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ImageFormat import ImageFormat
    from modules.util.enum.ModelType import ModelType
    from modules.util.ModelNames import ModelNames
    from modules.util.ModelWeightDtypes import ModelWeightDtypes

    if capture_steps is None:
        capture_steps = spread_steps(diffusion_steps, DEFAULT_CAPTURE_STEPS)
    bad = [s for s in capture_steps if not 0 <= s < diffusion_steps]
    if bad:
        raise ContributionError(
            f"capture step(s) {bad} outside 0..{diffusion_steps - 1}"
        )

    model = AnimaModel(model_type=ModelType.ANIMA)
    AnimaModelLoader().load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=ModelNames(base_model=base_model),
        weight_dtypes=ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16),
        quantization=QuantizationConfig.default_values(),
    )
    model.train_dtype = DataType.BFLOAT_16

    train_device = torch.device(device)
    sampler = AnimaSampler(
        train_device=train_device,
        temp_device=train_device,
        model=model,
        model_type=ModelType.ANIMA,
    )

    resolved, unmatched = resolve_modules(model.transformer, layer_prefixes)
    if not resolved:
        raise ContributionError(
            "none of the adapter's layers resolve to a Linear in model.transformer — "
            "the adapter does not target this base model"
        )

    # Conditional pass first, then the unconditional one; see ActivationCapture.
    calls_per_step = 2 if cfg_scale > 1.0 else 1
    capture_calls = [s * calls_per_step for s in capture_steps]

    per_prompt: dict[str, dict[str, torch.Tensor]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, prompt in enumerate(prompts):
            capture = ActivationCapture(capture_calls, max_tokens, seed + i)
            capture.attach(resolved, model.transformer)
            try:
                sample_config = SampleConfig.default_values()
                sample_config.prompt = prompt
                sample_config.negative_prompt = negative_prompt
                sample_config.height = height
                sample_config.width = width
                sample_config.diffusion_steps = diffusion_steps
                sample_config.cfg_scale = cfg_scale
                sample_config.seed = seed
                sample_config.random_seed = False
                sampler.sample(
                    sample_config=sample_config,
                    destination=str(Path(tmp) / f"line{i}.png"),
                    image_format=ImageFormat.PNG,
                    on_sample=lambda _out: None,
                )
            finally:
                capture.detach()

            observed = capture.call_index + 1
            expected = diffusion_steps * calls_per_step
            if observed != expected:
                raise ContributionError(
                    f"the transformer was called {observed} times over {diffusion_steps} "
                    f"steps, expected {expected} — the sampler no longer makes "
                    f"{calls_per_step} call(s) per step, so 'the conditional pass is "
                    "call 0' is no longer true and the capture would be silently mixing "
                    "in the negative prompt"
                )
            per_prompt[f"line{i}"] = capture.collected()

    meta: dict[str, object] = {
        "base_model": base_model,
        "device": device,
        "height": height,
        "width": width,
        "diffusion_steps": diffusion_steps,
        "capture_steps": list(capture_steps),
        "calls_per_step": calls_per_step,
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    return per_prompt, meta, unmatched


def spread_steps(total: int, count: int) -> list[int]:
    """``count`` step indices spread over ``0..total-1``, endpoints included.

    Contribution varies over the trajectory -- early steps are layout, late
    ones are texture -- so a single timestep would measure one regime and call
    it the adapter's.
    """
    if count <= 1 or total <= 1:
        return [0]
    count = min(count, total)
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapters", nargs="+", help="one or more .safetensors adapters")
    parser.add_argument("--prompt", action="append", default=[], metavar="TEXT",
                        help="a prompt line; repeat for each (order is the label order)")
    parser.add_argument("--prompt-file", default=None, metavar="PATH",
                        help="file with one prompt per line, appended after any --prompt")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, metavar="PATH")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument("--capture-steps", default=None, metavar="I,J,...",
                        help=f"step indices to capture at (default: {DEFAULT_CAPTURE_STEPS} spread over the trajectory)")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="1.0 (default) makes one transformer call per step, all conditional")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="rows kept per layer per prompt; memory is layers x this x in_features x 4 bytes")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity for the per-layer group label")
    args = parser.parse_args()

    for raw in args.adapters:
        if not Path(raw).is_file():
            sys.exit(f"block_contribution: no such file: {raw}")

    prompts = list(args.prompt)
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        prompts += [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if not prompts:
        sys.exit("block_contribution: no prompts — pass --prompt or --prompt-file")

    capture_steps = None
    if args.capture_steps:
        try:
            capture_steps = [int(s) for s in args.capture_steps.split(",") if s.strip()]
        except ValueError:
            sys.exit(f"block_contribution: malformed --capture-steps: {args.capture_steps!r}")

    paths = [Path(a) for a in args.adapters]
    try:
        # The layer set comes from the adapters, so only what an adapter
        # actually targets is ever hooked.
        probe = lora_soup.load_lora(paths[0], 1.0)
        activations, meta, unmatched = capture_activations(
            prompts=prompts,
            layer_prefixes=sorted(probe.layers),
            base_model=args.base_model,
            device=args.device,
            height=args.height,
            width=args.width,
            diffusion_steps=args.diffusion_steps,
            capture_steps=capture_steps,
            cfg_scale=args.cfg_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
        del probe
        meta["unmatched_layers"] = unmatched
        out = score(
            paths=paths,
            activations=activations,
            prompts=prompts,
            config_path=args.config,
            granularity=args.granularity,
            capture=meta,
        )
    except (ContributionError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_contribution: {e}")
    json.dump(out, sys.stdout, indent=None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
