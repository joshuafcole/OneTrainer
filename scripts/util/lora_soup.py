"""Merge N LoRA safetensors files into one, in delta-weight space.

This is the merge engine for the preference-soup arc (cinema-studio phases
454-457): a greedy soup, a block ablation, a coefficient search and a warm
start are all "combine these adapters with these coefficients", differing only
in where the coefficients come from.

Method
------
For every LoRA-decomposable layer, OneTrainer stores three tensors under a
common prefix (``modules/module/LoRAModule.py``)::

    <prefix>.lora_down.weight   A, shape (rank, in, *kernel)
    <prefix>.lora_up.weight     B, shape (out, rank, *ones)
    <prefix>.alpha              scalar

and the delta the model actually sees is (``LoRAModule.forward``, and
``DoRAModule`` via ``PeftBase.make_weight``)::

    dW = (alpha / rank) * B @ A

We reconstruct each input's ``dW_i``, form ``dW = sum_i c_i * dW_i``, and only
then re-factor. **Never average (B, A) factors directly** -- the average of
factorizations is not a factorization of the average (``test_lora_soup.py``
pins this).

Coefficients are used exactly as given. They are *not* normalized behind the
user's back: a soup passes coefficients that already sum to 1, a rescale passes
a single coefficient of 1.5, and both must mean what they say.

All arithmetic happens in float32 regardless of the stored dtype -- an SVD of an
fp16 matrix is not an acceptable approximation of the SVD -- and the result is
cast back on write.

Re-factoring, and the alpha convention
--------------------------------------
``--method svd`` (default) truncates an SVD of ``dW`` back to a target rank
(default: the largest rank any input used for that layer), so the output is
loadable by the same plan config that produced the inputs -- which is the whole
point for a warm start.

**The output always sets ``alpha = rank``, so the loader's ``alpha/rank`` scale
is exactly 1.0 and ``dW == B @ A``.** The alternative -- preserving an input's
alpha/rank ratio and pre-dividing the factors by it -- carries an arbitrary
scalar through every downstream tool for no benefit, and becomes ambiguous the
moment inputs disagree on alpha or the target rank differs from the input rank.
With ``alpha = rank`` there is nothing to reconcile. The singular values are
split evenly between the factors (``B = U*sqrt(S)``, ``A = sqrt(S)*Vh``) so
neither side carries the whole magnitude into fp16 storage.

``--method concat`` is exact instead of approximate: it stacks the per-input
factors (B columns / A rows) with each input's ``c_i * alpha_i / rank_i`` folded
into its B block, giving an output of rank ``sum_i rank_i`` whose ``B @ A`` is
the weighted sum with no truncation error at all. It requires every input to
contribute the same key set (there is no meaningful "absent" block to stack).

What this refuses
-----------------
Only plain LoRA decomposes as ``(alpha/rank)*B@A``. LoHa (``hada_w*``), LoKr
(``lokr_w*``), OFT (``oft_*``) and DoRA (``dora_scale``) do not, and this script
**refuses a file containing any key it does not understand**, naming the keys.
It never merges them approximately. That refusal is a deliverable.

Bundled TI vectors (``bundle_emb.<placeholder>.<qwen|qwen_out|t5>``, written by
``AnimaLoRASaver`` when ``bundle_additional_embeddings`` is set) are carried
**verbatim** from the anchor input -- byte-identical, not averaged, not even
dtype-converted. Averaging text-encoder embeddings is a separate question and
this script does not open it.

Provenance
----------
The output header is the anchor input's header (so an ``ot_config`` survives --
a warm start is only reproducible if the config that made it is still attached),
with the model-spec hash recomputed over the new tensors and a ``soup`` block
stamped in: every input's path, file hash, and coefficient, plus the method,
target rank, block scales and the fork revision.

Usage
-----
::

    python scripts/util/lora_soup.py \\
        --output soup.safetensors \\
        --input a.safetensors:0.5 --input b.safetensors:0.5 \\
        [--method svd|concat] [--rank 32] [--dtype bf16] \\
        [--block-scale '*attn1*=0.0']

Importable, too: 454's greedy walk and 456's search call ``load_lora`` /
``soup`` in-process rather than shelling out per candidate.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import math
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor

from safetensors import safe_open
from safetensors.torch import save_file

DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"
BUNDLE_PREFIX = "bundle_emb."

METHOD_SVD = "svd"
METHOD_CONCAT = "concat"

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}

# Distinctive key fragments of the PEFT types that do *not* decompose as
# (alpha/rank)*B@A, so the refusal can say which one it is looking at.
FOREIGN_PEFT_MARKERS: list[tuple[str, str]] = [
    ("hada_w", "LoHa"),
    ("lokr_w", "LoKr"),
    ("oft_", "OFT"),
    ("dora_scale", "DoRA"),
]


class SoupError(Exception):
    """Anything that makes a merge unsafe. The CLI turns these into exits."""


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


@dataclasses.dataclass
class LoraLayer:
    """One LoRA-decomposable layer, exactly as stored in a safetensors file."""

    down: Tensor  # A, (rank, in, *kernel)
    up: Tensor  # B, (out, rank, *ones)
    alpha: float

    @property
    def rank(self) -> int:
        return self.down.shape[0]

    @property
    def out_features(self) -> int:
        return self.up.shape[0]

    @property
    def a_trailing(self) -> tuple[int, ...]:
        return tuple(self.down.shape[1:])

    @property
    def b_trailing(self) -> tuple[int, ...]:
        return tuple(self.up.shape[2:])

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def geometry(self) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """What must agree across inputs. The rank deliberately is not in here."""
        return (self.out_features, self.a_trailing, self.b_trailing)

    def delta(self) -> Tensor:
        """(alpha/rank)*B@A, flattened to 2-D (out, prod(in, *kernel)), float32.

        The flattening reproduces ``PeftBase.make_weight``: B is viewed as
        (out, -1) and A as (rank, -1), which is a plain matmul for Linear and
        the correct kernel-folding for Conv2d.
        """
        a = self.down.to(torch.float32).reshape(self.rank, -1)
        b = self.up.to(torch.float32).reshape(self.out_features, -1)
        return (b @ a) * self.scale


@dataclasses.dataclass
class LoadedLora:
    """A parsed LoRA file plus the coefficient it enters the merge with."""

    path: Path
    coefficient: float
    layers: dict[str, LoraLayer]
    bundle: dict[str, Tensor]
    header: dict[str, str]
    dtype: torch.dtype
    file_sha256: str


@dataclasses.dataclass
class MergedLayer:
    """A summed delta plus the geometry needed to re-factor it."""

    delta: Tensor  # 2-D (out, prod(in, *kernel)), float32
    a_trailing: tuple[int, ...]
    b_trailing: tuple[int, ...]
    max_rank: int
    contributors: int


@dataclasses.dataclass
class SoupReport:
    """What the merge did, for the header and for the operator."""

    layers: int
    partial_layers: int
    output_ranks: dict[str, int]


def parse_input_spec(spec: str) -> tuple[Path, float]:
    """Split ``FILE:COEFF``.

    Rsplit, not split: Windows paths start ``d:/ai/...`` and the workspace this
    is aimed at lives on one.
    """
    path_text, _, coeff_text = spec.rpartition(":")
    if not path_text or not coeff_text:
        raise SoupError(f"--input expects FILE:COEFF, got {spec!r}")
    try:
        coefficient = float(coeff_text)
    except ValueError as e:
        raise SoupError(f"--input coefficient is not a number: {spec!r}") from e
    return Path(path_text), coefficient


def parse_block_scale(spec: str) -> tuple[str, float]:
    """Split ``PATTERN=COEFF``."""
    pattern, _, coeff_text = spec.rpartition("=")
    if not pattern or not coeff_text:
        raise SoupError(f"--block-scale expects PATTERN=COEFF, got {spec!r}")
    try:
        coefficient = float(coeff_text)
    except ValueError as e:
        raise SoupError(f"--block-scale coefficient is not a number: {spec!r}") from e
    return pattern, coefficient


def block_scale_for(prefix: str, block_scales: Sequence[tuple[str, float]]) -> float:
    """Product of every matching pattern's coefficient.

    Patterns are ``fnmatch`` globs matched against the **whole layer prefix**
    (case-sensitively), so a substring needs its own wildcards:
    ``'*attn1*=0.0'``, not ``'attn1=0.0'``. One rule, no guessing at intent.
    Overlapping patterns compose by multiplication, so a broad group scale and
    a narrow exception can both be expressed.
    """
    scale = 1.0
    for pattern, coefficient in block_scales:
        if fnmatch.fnmatchcase(prefix, pattern):
            scale *= coefficient
    return scale


def _describe_foreign_keys(keys: Sequence[str]) -> str:
    kinds = sorted({name for key in keys for marker, name in FOREIGN_PEFT_MARKERS if marker in key})
    return f" (looks like {', '.join(kinds)})" if kinds else ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"0x{digest.hexdigest()}"


def state_dict_sha256(state_dict: dict[str, Tensor]) -> str:
    """The tensor-data hash OneTrainer stamps as the model spec's ``hash_sha256``.

    Reproduced from ``DtypeModelSaverMixin.__calculate_safetensors_hash`` so the
    output's model spec keeps meaning the same thing it means in every other
    file the fork writes.
    """
    digest = hashlib.sha256()
    for tensor in OrderedDict(sorted(state_dict.items())).values():
        digest.update(tensor.contiguous().cpu().flatten().view(torch.uint8).numpy().tobytes())
    return f"0x{digest.hexdigest()}"


def fork_revision() -> str:
    """The fork's short revision.

    ``modules.util.git_util`` runs git in the *current working directory*, which
    for a script invoked from a workspace elsewhere records the wrong tree (or
    nothing). Ask about this file's own repo instead.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_lora(path: Path | str, coefficient: float) -> LoadedLora:
    """Read one LoRA safetensors file, refusing anything not plain LoRA."""
    path = Path(path)
    if not path.exists():
        raise SoupError(f"input does not exist: {path}")

    tensors: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        header = dict(f.metadata() or {})
        for key in f.keys():  # noqa: SIM118 -- safe_open is not a Mapping
            tensors[key] = f.get_tensor(key)

    layers: dict[str, LoraLayer] = {}
    bundle: dict[str, Tensor] = {}
    downs: dict[str, Tensor] = {}
    ups: dict[str, Tensor] = {}
    alphas: dict[str, Tensor] = {}
    foreign: list[str] = []

    for key, tensor in tensors.items():
        if key.startswith(BUNDLE_PREFIX):
            bundle[key] = tensor
        elif key.endswith(DOWN_SUFFIX):
            downs[key[: -len(DOWN_SUFFIX)]] = tensor
        elif key.endswith(UP_SUFFIX):
            ups[key[: -len(UP_SUFFIX)]] = tensor
        elif key.endswith(ALPHA_SUFFIX):
            alphas[key[: -len(ALPHA_SUFFIX)]] = tensor
        else:
            foreign.append(key)

    if foreign:
        shown = ", ".join(sorted(foreign)[:5])
        more = f" (+{len(foreign) - 5} more)" if len(foreign) > 5 else ""
        raise SoupError(
            f"{path}: {len(foreign)} key(s) are not plain-LoRA and cannot be merged in delta space"
            f"{_describe_foreign_keys(foreign)}: {shown}{more}. "
            "Only lora_down/lora_up/alpha and bundle_emb.* are understood; "
            "LoHa/LoKr/OFT/DoRA do not decompose as (alpha/rank)*B@A and this script "
            "will not merge them approximately."
        )

    for prefix in sorted(set(downs) | set(ups)):
        if prefix not in downs or prefix not in ups:
            missing = "lora_down.weight" if prefix not in downs else "lora_up.weight"
            raise SoupError(f"{path}: layer {prefix!r} is missing {missing}")
        down, up = downs[prefix], ups[prefix]
        rank = down.shape[0]
        if up.numel() // up.shape[0] != rank:
            # Two different files land here and the message has to distinguish
            # them: a genuinely inconsistent pair, and a well-formed adapter
            # whose lora_up carries a real kernel (not the 1x1 OneTrainer emits).
            # The second is refused because the flattening delta() relies on
            # would silently mis-fold it -- correct to refuse, wrong to call it
            # "inconsistent".
            if up.shape[1] == rank:
                raise SoupError(
                    f"{path}: layer {prefix!r} has a non-1x1 lora_up kernel "
                    f"{tuple(up.shape[2:])}; this is not OneTrainer plain LoRA and "
                    "will not be merged approximately"
                )
            raise SoupError(
                f"{path}: layer {prefix!r} has inconsistent rank: "
                f"lora_down says {rank}, lora_up is {tuple(up.shape)}"
            )
        # A missing alpha means an unscaled adapter; alpha == rank is scale 1.0.
        alpha = float(alphas[prefix].item()) if prefix in alphas else float(rank)
        layers[prefix] = LoraLayer(down=down, up=up, alpha=alpha)

    orphan_alphas = sorted(set(alphas) - set(layers))
    if orphan_alphas:
        raise SoupError(f"{path}: alpha without factors for layer(s): {', '.join(orphan_alphas[:5])}")

    if not layers:
        raise SoupError(f"{path}: no LoRA layers found")

    dtype = next(iter(layers.values())).down.dtype
    return LoadedLora(
        path=path,
        coefficient=coefficient,
        layers=layers,
        bundle=bundle,
        header=header,
        dtype=dtype,
        file_sha256=_file_sha256(path),
    )


def anchor_of(inputs: Sequence[LoadedLora]) -> LoadedLora:
    """The highest-coefficient input: the one whose header, dtype and bundled
    embeddings the output inherits. Ties go to the first listed."""
    return max(inputs, key=lambda loaded: loaded.coefficient)


def merge_deltas(
    inputs: Sequence[LoadedLora],
    block_scales: Sequence[tuple[str, float]] = (),
    log: Callable[[str], None] = _stderr,
) -> tuple[dict[str, MergedLayer], SoupReport]:
    """Sum ``c_i * dW_i`` over every layer any input contributes.

    A key present in some inputs and absent in others contributes ``dW = 0`` for
    the absent ones -- a layer an adapter never trained adds nothing. That is a
    judgement, so it is counted and reported, never silent. A key whose geometry
    disagrees across inputs is a hard error: there is no defensible sum.
    """
    if not inputs:
        raise SoupError("no inputs")

    all_prefixes = sorted({prefix for loaded in inputs for prefix in loaded.layers})
    merged: dict[str, MergedLayer] = {}
    partial = 0

    for prefix in all_prefixes:
        contributors = [loaded for loaded in inputs if prefix in loaded.layers]
        if len(contributors) != len(inputs):
            partial += 1

        geometry = contributors[0].layers[prefix].geometry()
        for loaded in contributors[1:]:
            other = loaded.layers[prefix].geometry()
            if other != geometry:
                raise SoupError(
                    f"layer {prefix!r} has disagreeing shapes: "
                    f"{contributors[0].path.name} is {geometry}, {loaded.path.name} is {other}"
                )

        scale = block_scale_for(prefix, block_scales)
        total: Tensor | None = None
        for loaded in contributors:
            term = loaded.layers[prefix].delta() * loaded.coefficient
            total = term if total is None else total + term
        assert total is not None
        total = total * scale

        merged[prefix] = MergedLayer(
            delta=total,
            a_trailing=geometry[1],
            b_trailing=geometry[2],
            max_rank=max(loaded.layers[prefix].rank for loaded in contributors),
            contributors=len(contributors),
        )

    if partial:
        log(
            f"note: {partial} of {len(all_prefixes)} layer(s) are absent from at least one input "
            "and were treated as delta = 0 there"
        )

    return merged, SoupReport(layers=len(all_prefixes), partial_layers=partial, output_ranks={})


def refactor_svd(layer: MergedLayer, rank: int | None = None) -> tuple[Tensor, Tensor, float]:
    """Factor a summed delta back to ``(A, B, alpha)`` at a target rank.

    Returns float32 factors with ``alpha == rank``, i.e. ``dW == B @ A`` with the
    loader's ``alpha/rank`` scale equal to 1.0. Singular values are split evenly
    between the factors so neither carries the whole magnitude.
    """
    target = layer.max_rank if rank is None else rank
    if target < 1:
        raise SoupError(f"target rank must be >= 1, got {target}")
    target = min(target, min(layer.delta.shape))

    u, s, vh = torch.linalg.svd(layer.delta.to(torch.float32), full_matrices=False)
    root = s[:target].clamp_min(0.0).sqrt()
    down = (root.unsqueeze(1) * vh[:target, :]).reshape(target, *layer.a_trailing)
    up = (u[:, :target] * root.unsqueeze(0)).reshape(-1, target, *layer.b_trailing)
    return down.contiguous(), up.contiguous(), float(target)


def refactor_concat(
    prefix: str,
    inputs: Sequence[LoadedLora],
    block_scales: Sequence[tuple[str, float]] = (),
) -> tuple[Tensor, Tensor, float]:
    """Exact rank-concatenation of every input's factors for one layer.

    Each input's ``c_i * block_scale * alpha_i / rank_i`` is folded into its
    blocks, and ``alpha == sum_i rank_i`` keeps the loader's scale at 1.0, so
    ``B @ A`` is the weighted sum with no truncation whatsoever.

    That factor is applied as its square root to *both* blocks rather than whole
    to B, matching the SVD path's even split. Folded entirely into B, a small
    coefficient lands on one stored tensor: at c=0.01 with fp16 output ~99% of
    B's entries go subnormal and the error is several times the SVD path's; at
    c=0.001 entries flush to zero outright. 456 searches coefficients down
    there, so the unbalanced form would degrade precisely the candidates the
    search is exploring. The sign rides on B, since a negative coefficient is
    legal (bounded task-arithmetic extrapolation) and has no square root.
    """
    scale = block_scale_for(prefix, block_scales)
    downs: list[Tensor] = []
    ups: list[Tensor] = []
    for loaded in inputs:
        layer = loaded.layers[prefix]
        factor = loaded.coefficient * scale * layer.scale
        root = math.sqrt(abs(factor))
        downs.append(layer.down.to(torch.float32) * root)
        ups.append(layer.up.to(torch.float32) * math.copysign(root, factor))
    down = torch.cat(downs, dim=0)
    up = torch.cat(ups, dim=1)
    return down.contiguous(), up.contiguous(), float(down.shape[0])


def build_soup_header(
    inputs: Sequence[LoadedLora],
    state_dict: dict[str, Tensor],
    method: str,
    rank: int | None,
    block_scales: Sequence[tuple[str, float]],
    dtype: torch.dtype,
    report: SoupReport,
) -> dict[str, str]:
    """The anchor's header, its model-spec hash refreshed, plus a ``soup`` block.

    Everything else -- crucially ``ot_config``, if the inputs were saved with
    one -- is carried through untouched.
    """
    anchor = anchor_of(inputs)
    header = dict(anchor.header)
    if "hash_sha256" in header:
        header["hash_sha256"] = state_dict_sha256(state_dict)
    header["soup"] = json.dumps(
        {
            "version": 1,
            "method": method,
            "target_rank": "max-input-rank" if rank is None else int(rank),
            "compute_dtype": "float32",
            "output_dtype": str(dtype).removeprefix("torch."),
            "alpha_convention": "alpha == output rank, so (alpha/rank) == 1.0 and dW == B @ A",
            "anchor": anchor.path.name,
            "block_scales": [{"pattern": p, "coefficient": c} for p, c in block_scales],
            "layers": report.layers,
            "partial_layers": report.partial_layers,
            "ot_revision": fork_revision(),
            "inputs": [
                {
                    "name": loaded.path.name,
                    "path": str(loaded.path),
                    "coefficient": loaded.coefficient,
                    # sha256 of the *file bytes*, computed here. Distinct from the
                    # input's own header "hash_sha256", which OneTrainer computes
                    # over sorted tensor data only; both are recorded, named apart.
                    "file_sha256": loaded.file_sha256,
                    "header_hash_sha256": loaded.header.get("hash_sha256"),
                }
                for loaded in inputs
            ],
        },
        sort_keys=True,
    )
    return header


def soup(
    inputs: Sequence[LoadedLora],
    method: str = METHOD_SVD,
    rank: int | None = None,
    block_scales: Sequence[tuple[str, float]] = (),
    dtype: torch.dtype | None = None,
    log: Callable[[str], None] = _stderr,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Merge loaded LoRAs into one state dict plus its safetensors header."""
    if not inputs:
        raise SoupError("no inputs")
    if method not in (METHOD_SVD, METHOD_CONCAT):
        raise SoupError(f"unknown method {method!r}")

    anchor = anchor_of(inputs)
    out_dtype = anchor.dtype if dtype is None else dtype

    merged, report = merge_deltas(inputs, block_scales, log=log)

    if method == METHOD_CONCAT:
        if report.partial_layers:
            raise SoupError(
                f"--method concat requires every input to contribute the same key set, but "
                f"{report.partial_layers} layer(s) are absent from at least one input. "
                "Use --method svd, which treats an absent layer as delta = 0."
            )
        if rank is not None:
            raise SoupError("--rank is meaningless for --method concat (output rank is the sum of input ranks)")

    state_dict: dict[str, Tensor] = {}
    for prefix in sorted(merged):
        if method == METHOD_CONCAT:
            down, up, alpha = refactor_concat(prefix, inputs, block_scales)
        else:
            down, up, alpha = refactor_svd(merged[prefix], rank)
        report.output_ranks[prefix] = down.shape[0]
        state_dict[prefix + DOWN_SUFFIX] = down.to(out_dtype)
        state_dict[prefix + UP_SUFFIX] = up.to(out_dtype)
        # alpha stays float32: it is one scalar per layer, and rounding it to
        # bf16 would silently rescale the whole layer.
        state_dict[prefix + ALPHA_SUFFIX] = torch.tensor(alpha, dtype=torch.float32)

    # Bundled TI vectors ride along verbatim from the anchor -- same tensor,
    # same dtype, byte-identical. Averaging text-encoder embeddings is a
    # separate question and this script does not open it.
    state_dict.update(anchor.bundle)
    if anchor.bundle:
        log(f"note: carried {len(anchor.bundle)} bundle_emb tensor(s) verbatim from {anchor.path.name}")

    header = build_soup_header(inputs, state_dict, method, rank, block_scales, out_dtype, report)
    return state_dict, header


def soup_files(
    specs: Sequence[tuple[Path | str, float]],
    method: str = METHOD_SVD,
    rank: int | None = None,
    block_scales: Sequence[tuple[str, float]] = (),
    dtype: torch.dtype | None = None,
    log: Callable[[str], None] = _stderr,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """``load_lora`` every spec, then ``soup`` them. The one-call entry point."""
    inputs = [load_lora(path, coefficient) for path, coefficient in specs]
    return soup(inputs, method=method, rank=rank, block_scales=block_scales, dtype=dtype, log=log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="where to write the merged .safetensors")
    parser.add_argument(
        "--input", action="append", required=True, metavar="FILE:COEFF",
        help="an input LoRA and its coefficient; repeat once per input",
    )
    parser.add_argument(
        "--method", choices=[METHOD_SVD, METHOD_CONCAT], default=METHOD_SVD,
        help="svd: re-factor at the target rank (default). concat: exact, output rank is the sum of input ranks",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="target rank for --method svd (default: the largest input rank, per layer)",
    )
    parser.add_argument(
        "--dtype", choices=sorted(DTYPE_ALIASES), default=None,
        help="output dtype (default: the highest-coefficient input's). Arithmetic is float32 regardless",
    )
    parser.add_argument(
        "--block-scale", action="append", default=[], metavar="PATTERN=COEFF",
        help="multiply matching layers' delta by COEFF; PATTERN is an fnmatch glob "
             "over the whole layer prefix, e.g. '*attn1*=0.0'. Repeatable; matches compose",
    )
    args = parser.parse_args()

    try:
        specs = [parse_input_spec(spec) for spec in args.input]
        block_scales = [parse_block_scale(spec) for spec in args.block_scale]
        dtype = DTYPE_ALIASES[args.dtype] if args.dtype else None

        for path, coefficient in specs:
            print(f"input   {path}  x {coefficient}", flush=True)
        state_dict, header = soup_files(
            specs, method=args.method, rank=args.rank, block_scales=block_scales, dtype=dtype
        )
    except SoupError as e:
        sys.exit(f"lora_soup: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(out_path), header)
    print(f"wrote   {out_path}  ({len(state_dict)} keys, {out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
