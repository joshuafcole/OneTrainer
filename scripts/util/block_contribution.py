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

The split is not just tidiness. **A capture depends on the base model and the
prompts and on nothing about the adapters**, so ``--activations PATH`` writes
one and reuses it: the GPU half runs once, and every adapter set anyone ever
scores against those prompts is free. ``--activations-only`` runs the expensive
half alone (schedule it when the card is idle); ``--require-cached`` runs the
cheap half alone, and refuses rather than quietly loading a model on a box that
is training. Reuse is gated on a content key over every capture parameter --
change one and the run stops rather than serving activations that answer a
different question.

Usage::

    python scripts/util/block_contribution.py a.safetensors b.safetensors ... \
        --prompt "..." --prompt "..." [--base-model PATH] [--max-tokens N] \
        [--activations CACHE.safetensors] [--granularity LEVEL] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_channel  # noqa: E402

# Before torch: the CUDA support libraries it loads write banners to fd 1 from
# C, and this script's stdout is its result. See ``result_channel``.
result_channel.claim()

import torch  # noqa: E402

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


class ContributionError(Exception):
    """The adapters and the activations do not describe the same model."""


# Higher than block_gram's ceiling on purpose. The *memory* here is linear in
# the adapter count -- one delta at a time per layer, as in block_gram -- and it
# is the deltas that dominate, not the pair table: at 2048x2048 float64 a single
# layer's delta is 33 MB, so N of them is what sets the ceiling either way. The
# pairwise reduction added below is quadratic in arithmetic but negligible in
# both (each pair is one dot product over an already-materialised matrix).
MAX_ADAPTERS = 128

#: Anima's own sampler resolution. Kept as the default so a capture run and the
#: render it is compared against see the same token geometry.
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_DIFFUSION_STEPS = 25
DEFAULT_CAPTURE_STEPS = 4
DEFAULT_MAX_TOKENS = 256
DEFAULT_BASE_MODEL = "D:/models/diffusers/anima/anima-base-v1.0"


# --------------------------------------------------------------------------
# Phase 2: scoring. Pure, tested, model-free.
# --------------------------------------------------------------------------


def _pair_table(vectors: Sequence[torch.Tensor], scale: float = 1.0) -> list[list[float]]:
    """``[i][j] = <v_i, v_j> * scale``, computed once per unordered pair.

    Symmetric by construction rather than by arithmetic: the lower triangle is
    copied from the upper, so a Gram this returns cannot be asymmetric to
    floating-point noise and a solver reading it cannot get a complex
    eigenvalue out of a real symmetric problem.
    """
    n = len(vectors)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = float(torch.dot(vectors[i], vectors[j])) * scale
            table[i][j] = value
            table[j][i] = value
    return table


def _empty_group(n: int, prompt_count: int) -> dict[str, object]:
    return {
        "layer_count": 0,
        "frobenius_gram": [[0.0] * n for _ in range(n)],
        "contribution_gram": [
            [[0.0] * n for _ in range(n)] for _ in range(prompt_count)
        ],
    }


def _accumulate(into: list[list[float]], addend: Sequence[Sequence[float]]) -> None:
    for i, row in enumerate(addend):
        target = into[i]
        for j, value in enumerate(row):
            target[j] += value


def score(
    paths: Sequence[Path],
    activations: Mapping[str, Mapping[str, torch.Tensor]],
    prompts: Sequence[str],
    config_path: str | None = None,
    granularity: str | None = None,
    capture: Mapping[str, object] | None = None,
    emit_layer_gram: bool = False,
) -> dict[str, object]:
    """``{layers, groups, per-(prompt, adapter) contribution, Frobenius}``.

    ``activations`` is keyed by prompt label then by layer prefix; each value is
    a 2-D ``(rows, in_features)`` matrix of that layer's captured inputs. One
    layer's deltas are materialised at a time and reduced to scalars
    immediately, as in ``block_gram`` and for the same reason.

    **The off-diagonals are the point.** A merge's effect is not the sum of its
    inputs' effects: ``|| sum_i c_i dW_i X ||^2 = sum_ij c_i c_j <dW_i X, dW_j X>``,
    so a coefficient vector can only be solved against the full pair table. The
    diagonal alone -- what a per-adapter contribution is -- predicts the merge
    only when the inputs are orthogonal *in activation space*, which is the
    assumption this initiative has already found to be false in weight space.

    Grams are emitted **per group** always, because that is the resolution a
    coefficient is solved at and the whole table is a few hundred numbers.
    Per *layer* they are gated on ``emit_layer_gram``: at 224 layers, 5 prompts
    and N adapters it is 1120*N^2 floats, which is diagnostic detail rather than
    something a planner reads.

    Summing a Gram across the layers of a group treats those layers' effects as
    commensurable. At ``fine`` granularity a group is one part at one depth
    band (``mid.attn2.to_q``), so they are. At ``coarse`` a group mixes parts
    whose outputs enter the network very differently -- a ``to_q`` perturbation
    moves attention logits through a softmax, a ``to_out`` one lands straight in
    the residual stream -- and the sum is correspondingly harder to read.
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
    group_totals: dict[str, dict[str, object]] = {}
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
        # Flat views, not copies: reshape on a contiguous tensor aliases it, so
        # the pair tables below cost arithmetic and no memory.
        flat_deltas = [d.reshape(-1) for d in deltas]
        frobenius_gram = _pair_table(flat_deltas)
        frobenius_sq = [frobenius_gram[i][i] for i in range(n)]

        energy: list[float] = []
        rows: list[int] = []
        contribution_sq: list[list[float]] = []
        contribution_gram: list[list[list[float]]] = []
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
            # Every adapter's projected delta at once, so the off-diagonals are
            # available. dW_i X^T is (out, rows) -- for the shapes this runs at,
            # two orders of magnitude smaller than the delta it came from.
            projected = [(d @ xd.T).reshape(-1) for d in deltas]
            gram = _pair_table(projected, scale=1.0 / count)
            contribution_gram.append(gram)
            contribution_sq.append([gram[i][i] for i in range(n)])

        entry: dict[str, object] = {
            "layer": prefix,
            "group": group_of[prefix],
            "in_features": int(in_features),
            "frobenius_sq": frobenius_sq,
            "rows": rows,
            "activation_energy": energy,
            "contribution_sq": contribution_sq,
        }
        if emit_layer_gram:
            entry["frobenius_gram"] = frobenius_gram
            entry["contribution_gram"] = contribution_gram
        layers.append(entry)

        group = group_of[prefix]
        bucket = group_totals.setdefault(group, _empty_group(n, len(labels)))
        bucket["layer_count"] += 1
        _accumulate(bucket["frobenius_gram"], frobenius_gram)
        for p in range(len(labels)):
            _accumulate(bucket["contribution_gram"][p], contribution_gram[p])

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
        "groups": [
            {
                "group": name,
                "layer_count": group_totals[name]["layer_count"],
                "frobenius_gram": group_totals[name]["frobenius_gram"],
                "contribution_gram": group_totals[name]["contribution_gram"],
            }
            for name in sorted(group_totals)
        ],
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

    # OT's own bootstrap: puts the checkout root on sys.path so ``modules.*``
    # resolves. Every upstream script that imports ``modules`` calls it, because
    # running a file by path puts only *its own* directory on sys.path -- not the
    # repo root, and not the cwd. Called here rather than at module scope so
    # ``score`` and the tests still import without OneTrainer's dependency tree.
    from import_util import script_imports

    script_imports()

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


# --------------------------------------------------------------------------
# The capture cache. Activations are a function of the base model and the
# prompts, and of nothing about the adapters -- so the expensive half is
# reusable across every adapter set anyone ever scores.
# --------------------------------------------------------------------------

#: Separates the prompt label from the layer prefix in a cache file's keys.
#: A layer prefix is dot-separated and a label is generated, so neither can
#: contain this.
CACHE_SEP = "|"

#: Which capture parameters make two activation sets the same object. A
#: parameter absent here is one a caller could change without invalidating the
#: cache, which is how a stale answer gets served with confidence -- so this
#: list is the whole contract, not a summary of it.
CACHE_KEY_FIELDS = (
    "base_model",
    "height",
    "width",
    "diffusion_steps",
    "capture_steps",
    "cfg_scale",
    "negative_prompt",
    "seed",
    "max_tokens",
)


def cache_key(meta: Mapping[str, object], prompts: Sequence[str],
              layer_prefixes: Sequence[str]) -> str:
    """A content identity for one capture, derived rather than assigned.

    The layer set is in the key because a capture taken for an attention-only
    adapter cannot score one that also reaches the MLPs: the layers it needs
    were never hooked. Reusing it would silently score a subset and report a
    smaller ``scored_layer_count``, which reads as an adapter fact.
    """
    import hashlib

    payload = json.dumps(
        {
            **{field: meta.get(field) for field in CACHE_KEY_FIELDS},
            "prompts": list(prompts),
            "layers": sorted(layer_prefixes),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def save_activations(
    path: Path,
    activations: Mapping[str, Mapping[str, torch.Tensor]],
    meta: Mapping[str, object],
    prompts: Sequence[str],
) -> None:
    """One self-describing safetensors file: the tensors and their provenance.

    The metadata travels *inside* the file rather than in a sidecar, so a cache
    can never be separated from the parameters that make it valid.
    """
    from safetensors.torch import save_file

    flat = {
        f"{label}{CACHE_SEP}{prefix}": tensor.contiguous()
        for label, per_layer in activations.items()
        for prefix, tensor in per_layer.items()
    }
    layer_prefixes = sorted({
        prefix for per_layer in activations.values() for prefix in per_layer
    })
    header = {
        "block_contribution": "1",
        "meta": json.dumps(dict(meta), sort_keys=True),
        "prompts": json.dumps(list(prompts)),
        "labels": json.dumps(list(activations)),
        "key": cache_key(meta, prompts, layer_prefixes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(path), metadata=header)


def load_activations(
    path: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object], list[str], str]:
    """``(activations, meta, prompts, key)`` from a cache file."""
    from safetensors import safe_open

    activations: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt") as f:
        header = f.metadata() or {}
        if "block_contribution" not in header:
            raise ContributionError(
                f"{path} is a safetensors file but not a block_contribution capture"
            )
        meta = json.loads(header["meta"])
        prompts = json.loads(header["prompts"])
        labels = json.loads(header["labels"])
        for label in labels:
            activations[label] = {}
        for name in f.keys():  # noqa: SIM118 - safe_open is not a Mapping
            label, _, prefix = name.partition(CACHE_SEP)
            if label not in activations:
                raise ContributionError(
                    f"{path}: tensor {name!r} names a prompt label that is not in the "
                    "file's own label list"
                )
            activations[label][prefix] = f.get_tensor(name)
    return activations, meta, prompts, header.get("key", "")


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
    parser.add_argument("--layer-gram", action="store_true",
                        help="also emit the pair tables per layer, not only per "
                             "group — diagnostic, and O(layers * prompts * N^2)")
    parser.add_argument("--activations", default=None, metavar="PATH",
                        help="reuse this capture if it matches, otherwise write it "
                             "— the capture depends on the base model and prompts, "
                             "never on the adapters")
    parser.add_argument("--activations-only", action="store_true",
                        help="capture and write, then stop; scores nothing. Lets the "
                             "GPU half run when the card is free and the cheap half "
                             "run whenever")
    parser.add_argument("--require-cached", action="store_true",
                        help="refuse rather than load the model — for scoring on a box "
                             "whose GPU is busy, or on one that has no model at all")
    args = parser.parse_args()

    if args.activations_only and args.require_cached:
        sys.exit("block_contribution: --activations-only and --require-cached ask for "
                 "opposite things")
    if (args.activations_only or args.require_cached) and not args.activations:
        sys.exit("block_contribution: --activations-only/--require-cached need --activations PATH")

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
    cache = Path(args.activations) if args.activations else None
    try:
        # The layer set comes from the adapters, so only what an adapter
        # actually targets is ever hooked.
        probe = lora_soup.load_lora(paths[0], 1.0)
        layer_prefixes = sorted(probe.layers)
        del probe

        wanted = {
            "base_model": args.base_model,
            "height": args.height,
            "width": args.width,
            "diffusion_steps": args.diffusion_steps,
            "capture_steps": (
                capture_steps
                if capture_steps is not None
                else spread_steps(args.diffusion_steps, DEFAULT_CAPTURE_STEPS)
            ),
            "cfg_scale": args.cfg_scale,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        }
        wanted_key = cache_key(wanted, prompts, layer_prefixes)

        activations = meta = None
        if cache is not None and cache.is_file():
            activations, meta, cached_prompts, key = load_activations(cache)
            if key != wanted_key:
                # Named, not silently re-captured: overwriting the file another
                # scoring run is about to reuse is a worse surprise than
                # stopping, and the mismatch is usually a typo in one prompt.
                raise ContributionError(
                    f"{cache} was captured under different parameters "
                    f"(key {key[:12]} vs {wanted_key[:12]}); pass a different "
                    "--activations path, or delete that one to re-capture"
                )
            prompts = cached_prompts
            print(f"reusing capture {cache} ({key[:12]})", file=sys.stderr)

        if activations is None:
            if args.require_cached:
                raise ContributionError(
                    f"--require-cached, but {cache} does not exist — capture it on a "
                    "box with the model first (--activations-only)"
                )
            activations, meta, unmatched = capture_activations(
                prompts=prompts,
                layer_prefixes=layer_prefixes,
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
            meta["unmatched_layers"] = unmatched
            if cache is not None:
                save_activations(cache, activations, meta, prompts)
                print(f"wrote capture {cache} ({wanted_key[:12]})", file=sys.stderr)

        if args.activations_only:
            # JSON on stdout here too, so a caller that pre-warms a capture and
            # a caller that scores one read the same channel. Deliberately not
            # a truncated score result: it has no adapters in it.
            result_channel.emit_json(
                {
                    "cache_key": wanted_key,
                    "path": str(cache),
                    "prompts": list(prompts),
                    "prompt_labels": list(activations),
                    "layer_count": len(next(iter(activations.values()))),
                    "capture": meta,
                }
            )
            return 0

        out = score(
            paths=paths,
            activations=activations,
            prompts=prompts,
            config_path=args.config,
            granularity=args.granularity,
            emit_layer_gram=args.layer_gram,
            capture={**meta, "cache_key": wanted_key},
        )
    except (ContributionError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_contribution: {e}")
    result_channel.emit_json(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
