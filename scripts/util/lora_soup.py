"""Merge N LoRA/LoKr safetensors files into one, in delta-weight space.

This is the merge engine for the preference-soup arc (cinema-studio phases
454-457): a greedy soup, a block ablation, a coefficient search and a warm
start are all "combine these adapters with these coefficients", differing only
in where the coefficients come from.

Method
------
The engine works in **delta space**. All it ever needs from an adapter is a
closed-form ``dW`` per layer; the factor shapes that produced it are an input
detail, not part of the arithmetic. We reconstruct each input's ``dW_i``, form
``dW = sum_i c_i * dW_i``, and only then re-factor. **Never average factors
directly** -- the average of factorizations is not a factorization of the
average (``test_lora_soup.py`` pins this).

For every LoRA-decomposable layer, OneTrainer stores three tensors under a
common prefix (``modules/module/LoRAModule.py``)::

    <prefix>.lora_down.weight   A, shape (rank, in, *kernel)
    <prefix>.lora_up.weight     B, shape (out, rank, *ones)
    <prefix>.alpha              scalar

and the delta the model actually sees is (``LoRAModule.forward``, and
``DoRAModule`` via ``PeftBase.make_weight``)::

    dW = (alpha / rank) * B @ A

**LoKr has a closed-form delta too**, so it merges here on exactly the same
footing. Per ``LoKrModule._get_factors`` / ``get_weight`` / ``forward``::

    w1 = lokr_w1                                        if stored whole
       = lokr_w1_a @ lokr_w1_b                          if decompose_both
    w2 = lokr_w2                                        if stored whole
       = rebuild_tucker(lokr_t2, lokr_w2_a, lokr_w2_b)  if tucker
       = lokr_w2_a @ lokr_w2_b                          otherwise

    dW = make_kron(w1, w2).view(shape) * (alpha / dim)

exact, no approximation. ``make_kron`` and ``rebuild_tucker`` are imported from
``modules.util.lokr_utils`` rather than reimplemented -- a second copy of the
Kronecker/Tucker index convention is exactly how a plausible-but-wrong delta
gets written.

The ``dim`` in that scale is the *factor* rank (``lokr_dim``), and the
checkpoint does not store it directly, so it is read back off the factor
shapes: ``lokr_w1_a``'s inner dim, else ``lokr_t2``'s leading dim, else
``lokr_w2_a``'s inner dim. When **both** factors are stored whole there is no
inner dim to read -- and in exactly that case ``LoKrModule.initialize_weights``
does ``self.alpha.fill_(lokr_dim)``, so the scale is 1.0 and ``dim`` is both
unrecoverable and irrelevant. Every shape that carries ``dim`` is cross-checked
against every other; a disagreement is refused, not averaged.

``.view(shape)``: the checkpoint has no ``shape`` key, but it does not need one.
``make_kron`` already returns ``(out_l*out_k, in_m*in_n)`` for Linear, and
unsqueezes ``w1`` so a 4-D ``w2`` yields ``(out, in, k1, k2)`` for Conv2d -- the
view is a no-op in both. The one case where it is *not* a no-op is a Conv2d
whose ``w2`` is factored, where ``lokr_w2_b`` folds the kernel into its trailing
dim: ``make_kron`` returns ``(out, in*k1*k2)`` and only ``.view(shape)`` splits
it back out. Nothing in the file distinguishes that from a Linear layer of
``in_features = in*k1*k2``, so this reads it as Linear. The **delta is exact
either way** -- same numbers, same order -- only the emitted factor *shape*
differs, and if a plain-LoRA input in the same soup says otherwise,
``merge_deltas``'s geometry check is what fires.

Coefficients are used exactly as given. They are *not* normalized behind the
user's back: a soup passes coefficients that already sum to 1, a rescale passes
a single coefficient of 1.5, and both must mean what they say.

All arithmetic happens in float32 regardless of the stored dtype -- an SVD of an
fp16 matrix is not an acceptable approximation of the SVD -- and the result is
cast back on write.

Re-factoring, and the alpha convention
--------------------------------------
**The output is always plain LoRA**, whatever went in. A merged sum of
Kronecker products is not generally a Kronecker product, so there is no exact
LoKr-shaped answer to emit; plain LoRA is the format the sum actually has.

``--method svd`` (default) truncates an SVD of ``dW`` back to a target rank
(default: the largest rank any input used for that layer), so the output is
loadable by the same plan config that produced the inputs -- which is the whole
point for a warm start. For a LoKr input "the rank it used" is *not*
``lokr_dim``: ``rank(kron(w1, w2)) = rank(w1) * rank(w2)``, so the default is
that algebraic bound. Defaulting to ``lokr_dim`` would truncate the delta hard
and silently, which is the opposite of what a default is for.

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
contribute the same key set (there is no meaningful "absent" block to stack),
and it is **plain-LoRA only**: the trick works because ``B @ A`` is bilinear in
a shared rank axis, and a LoKr layer has no such axis to stack along. That
combination is refused by name rather than quietly falling back to the SVD path.

There is no LoKr-shaped output mode, not even a lossy one.
``modules.util.lokr_utils.nearest_kron_factors`` would give the best
``kron(w1, w2)`` approximation of the merged delta, but a LoKr file's scale is
``alpha/dim`` with ``dim`` coming from the *training config*, not the file --
so a whole-``w1``/whole-``w2`` output (the only shape a Van Loan rearrangement
produces) has no way to pin its own scale to 1.0 the way ``alpha == rank`` does
for LoRA. It would load correctly only under the config that happened to match.
An approximation that is also silently mis-scaled is worse than no mode at all.

What this refuses
-----------------
This script **refuses a file containing any key it does not understand**, naming
the keys, and never merges them approximately. That refusal is a deliverable.
Understood: plain LoRA (``lora_down``/``lora_up``/``alpha``), LoKr (``lokr_*``),
and ``bundle_emb.*``.

The reasons the rest are refused are *not* the same reason, and the message says
which one applies:

- **OFT** (``oft_*``) is an orthogonal, **multiplicative** transform of the base
  weight. There is no additive ``dW`` to sum.
- **DoRA** (``dora_scale``) renormalizes the *combined* weight --
  ``dora_scale * (W + dW) / ||W + dW||`` -- which is not a sum of additive
  deltas either. Two DoRAs' contributions do not add.
- **LoHa** (``hada_w*``) *does* have a closed-form additive delta
  (``(W1 * W2) * alpha/rank``, a Hadamard product of two low-rank products) and
  could be carried here on the same footing as LoKr. It simply is not, yet.
  That is a gap, not an impossibility, and it is refused as one.

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

import abc
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
from math import prod
from pathlib import Path

import torch
from torch import Tensor

from safetensors import safe_open
from safetensors.torch import save_file

# Reach ``modules/`` the way ``scripts/util/import_util.py`` does -- repo root at
# the head of sys.path -- but without its ZLUDA half: this module is *imported*
# as a library by 454's greedy walk, and a library import must not have platform
# side effects. ``modules`` is a namespace package and ``lokr_utils`` imports
# nothing but torch, so this costs an already-paid import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.util.lokr_utils import make_kron, rebuild_tucker  # noqa: E402

# Sibling module: the same directory the block_* scripts put at the head of sys.path before importing this
# one. Added here too so ``lora_soup`` is importable on its own (454's greedy walk does exactly that).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import lora_namespace  # noqa: E402

DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"
BUNDLE_PREFIX = "bundle_emb."

# The LoKr key families, as ``LoKrModule`` registers them. Matched with
# ``endswith``, so ``.lokr_w1`` never swallows ``.lokr_w1_a``.
LOKR_W1 = "lokr_w1"
LOKR_W1_A = "lokr_w1_a"
LOKR_W1_B = "lokr_w1_b"
LOKR_W2 = "lokr_w2"
LOKR_W2_A = "lokr_w2_a"
LOKR_W2_B = "lokr_w2_b"
LOKR_T2 = "lokr_t2"
LOKR_NAMES: tuple[str, ...] = (LOKR_W1, LOKR_W1_A, LOKR_W1_B, LOKR_W2, LOKR_W2_A, LOKR_W2_B, LOKR_T2)

METHOD_SVD = "svd"
METHOD_CONCAT = "concat"

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}

# Distinctive key fragments of the PEFT types this engine does not merge, each
# with *why* refusing it is correct. The reasons are not interchangeable and the
# message must not lump them: OFT and DoRA have no additive delta to sum at all,
# whereas LoHa has one and merely isn't wired up. Naming the wrong reason is how
# a future reader concludes LoHa is impossible.
FOREIGN_PEFT_MARKERS: list[tuple[str, str, str]] = [
    (
        "hada_w", "LoHa",
        ("its delta is additive and closed-form ((W1 * W2) * alpha/rank), so this "
         "engine could carry it -- it just does not, yet"),
    ),
    (
        "oft_", "OFT",
        ("it is an orthogonal, multiplicative transform of the base weight; there "
         "is no additive delta to sum"),
    ),
    (
        "dora_scale", "DoRA",
        ("it renormalizes the combined weight (dora_scale * W/||W||), so its "
         "contributions are not additive deltas"),
    ),
]

# Said in one place because ``soup`` refuses this up front and ``refactor_concat``
# refuses it again for a direct caller.
CONCAT_NEEDS_LORA = (
    "--method concat stacks LoRA (A, B) factor blocks, which is exact only "
    "because dW = (alpha/rank)*B@A is bilinear in a rank axis the blocks share. "
    "A LoKr layer has no such axis: its delta is a Kronecker product, and a sum "
    "of Kronecker products is not generally a Kronecker product, so there is "
    "nothing to concatenate and no exact LoKr-shaped result to concatenate into. "
    "Use --method svd, which re-factors the summed delta to plain LoRA."
)


class SoupError(Exception):
    """Anything that makes a merge unsafe. The CLI turns these into exits."""


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


class AdapterLayer(abc.ABC):
    """One adapted layer, reduced to what a delta-space merge actually needs.

    That is the whole contract: a ``dW``, the geometry to re-factor it, and the
    rank to re-factor it *at*. A PEFT type belongs behind this interface exactly
    when it has a closed-form additive delta -- not when it merely has factors,
    and not when it merely has a ``get_weight()``. ``merge_deltas`` sees nothing
    else, which is why adding LoKr did not touch it.
    """

    @property
    @abc.abstractmethod
    def rank(self) -> int:
        """The rank the SVD path should default to for this layer.

        For LoRA that is the stored rank. For LoKr it is the algebraic rank
        bound of the Kronecker delta, **not** ``lokr_dim``.
        """

    @abc.abstractmethod
    def geometry(self) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """``(out_features, a_trailing, b_trailing)`` -- what must agree across
        inputs before their deltas may be summed. The rank deliberately is not
        in here, and neither is the PEFT type: two adapters of different types
        over the same layer describe the same weight and add fine."""

    @abc.abstractmethod
    def delta(self) -> Tensor:
        """``dW``, flattened to 2-D ``(out, prod(in, *kernel))``, float32."""

    @property
    @abc.abstractmethod
    def storage_dtype(self) -> torch.dtype:
        """The dtype this layer's factors are stored at -- what the output
        inherits when no ``--dtype`` is given. Arithmetic is float32 regardless."""


@dataclasses.dataclass
class LoraLayer(AdapterLayer):
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

    @property
    def storage_dtype(self) -> torch.dtype:
        return self.down.dtype

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
class LokrLayer(AdapterLayer):
    """One LoKr-decomposable layer, exactly as stored in a safetensors file.

    ``dim`` and ``alpha`` are resolved once at load time (see
    ``build_lokr_layer``), which is where every structural refusal lives; by the
    time one of these exists, the factor set is known well-formed and mutually
    consistent, so the methods below assert rather than re-check.
    """

    w1: Tensor | None  # (out_l, in_m), whole
    w1_a: Tensor | None  # (out_l, dim)
    w1_b: Tensor | None  # (dim, in_m)
    w2: Tensor | None  # (out_k, in_n[, k1, k2]), whole
    w2_a: Tensor | None  # (out_k, dim), or (dim, out_k) under Tucker
    w2_b: Tensor | None  # (dim, in_n*prod(kernel)), or (dim, in_n) under Tucker
    t2: Tensor | None  # (dim, dim, k1, k2), Tucker only
    alpha: float
    dim: int

    @property
    def scale(self) -> float:
        return self.alpha / self.dim

    @property
    def storage_dtype(self) -> torch.dtype:
        w1 = self.w1 if self.w1 is not None else self.w1_a
        assert w1 is not None
        return w1.dtype

    def _w1_shape(self) -> tuple[int, int]:
        if self.w1 is not None:
            return (self.w1.shape[0], self.w1.shape[1])
        assert self.w1_a is not None and self.w1_b is not None
        return (self.w1_a.shape[0], self.w1_b.shape[1])

    def _w2_shape(self) -> tuple[int, ...]:
        if self.w2 is not None:
            return tuple(self.w2.shape)
        assert self.w2_a is not None and self.w2_b is not None
        if self.t2 is not None:
            # rebuild_tucker's einsum 'ijkl,ip,jr->prkl': (out_k, in_n, k1, k2).
            return (self.w2_a.shape[1], self.w2_b.shape[1], *self.t2.shape[2:])
        return (self.w2_a.shape[0], self.w2_b.shape[1])

    def weight_shape(self) -> tuple[int, ...]:
        """The shape ``make_kron(w1, w2)`` produces -- which is ``get_weight``'s
        ``self.shape`` in every case the file can distinguish (see the module
        docstring on ``.view(shape)``)."""
        out_l, in_m = self._w1_shape()
        w2_shape = self._w2_shape()
        return (out_l * w2_shape[0], in_m * w2_shape[1], *w2_shape[2:])

    @property
    def out_features(self) -> int:
        return self.weight_shape()[0]

    @property
    def rank(self) -> int:
        """The algebraic rank bound of the Kronecker delta.

        ``rank(kron(w1, w2)) = rank(w1) * rank(w2)``, so this is the product of
        each factor's own bound -- ``lokr_dim`` where that factor is stored
        low-rank (a Tucker ``w2`` too: every row of the rebuilt ``w2`` lies in
        the span of ``lokr_t2``'s ``dim`` slices), its smaller side where it is
        stored whole -- capped at the delta's own smaller side.

        Emphatically **not** ``lokr_dim`` itself. A dim-4 LoKr over a 256x256
        Linear carries a delta of rank up to 64; defaulting the SVD to 4 would
        throw away 94% of it without a word.
        """
        out_l, in_m = self._w1_shape()
        w2_shape = self._w2_shape()
        w1_rank = min(out_l, in_m) if self.w1 is not None else min(self.dim, out_l, in_m)
        w2_rows, w2_cols = w2_shape[0], prod(w2_shape[1:])
        w2_rank = min(w2_rows, w2_cols) if self.w2 is not None else min(self.dim, w2_rows, w2_cols)
        shape = self.weight_shape()
        return max(1, min(w1_rank * w2_rank, shape[0], prod(shape[1:])))

    def geometry(self) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """The same triple a LoRA over this layer would report, so the two merge.

        ``b_trailing`` is reconstructed rather than read: OneTrainer emits a 1x1
        ``lora_up`` per kernel dim, so a delta with a kernel implies ``(1, 1)``
        and a Linear one implies ``()``.
        """
        shape = self.weight_shape()
        a_trailing = tuple(shape[1:])
        return (shape[0], a_trailing, (1,) * (len(a_trailing) - 1))

    def factors(self) -> tuple[Tensor, Tensor]:
        """``(w1, w2)`` in float32, reproducing ``LoKrModule._get_factors``
        minus the training-only dropout."""
        if self.w1 is not None:
            w1 = self.w1.to(torch.float32)
        else:
            assert self.w1_a is not None and self.w1_b is not None
            w1 = self.w1_a.to(torch.float32) @ self.w1_b.to(torch.float32)

        if self.w2 is not None:
            w2 = self.w2.to(torch.float32)
        elif self.t2 is not None:
            assert self.w2_a is not None and self.w2_b is not None
            w2 = rebuild_tucker(
                self.t2.to(torch.float32),
                self.w2_a.to(torch.float32),
                self.w2_b.to(torch.float32),
            )
        else:
            assert self.w2_a is not None and self.w2_b is not None
            w2 = self.w2_a.to(torch.float32) @ self.w2_b.to(torch.float32)
        return w1, w2

    def weight(self) -> Tensor:
        """``make_kron(w1, w2).view(shape) * (alpha/dim)`` -- the delta in its
        natural (out, in, *kernel) shape, float32."""
        w1, w2 = self.factors()
        return make_kron(w1, w2).reshape(self.weight_shape()) * self.scale

    def delta(self) -> Tensor:
        """The same weight flattened to 2-D (out, prod(in, *kernel)), float32.

        Flattening a ``.view(shape)`` back to (out, -1) is the identity on
        contiguous row-major data, so this is exactly ``weight()`` reshaped --
        no second convention to keep in step.
        """
        return self.weight().reshape(self.out_features, -1)


@dataclasses.dataclass
class LoadedLora:
    """A parsed adapter file plus the coefficient it enters the merge with.

    Named for the common case; the layers inside may be LoRA, LoKr, or both.
    """

    path: Path
    coefficient: float
    layers: dict[str, AdapterLayer]
    bundle: dict[str, Tensor]
    header: dict[str, str]
    dtype: torch.dtype
    file_sha256: str

    block_scales: tuple[tuple[str, float], ...] = ()
    """This input's own per-layer coefficients, on top of :attr:`coefficient`.

    The difference from the merge-wide ``block_scales`` is the whole point of
    having both. That one scales a layer *after* the sum -- ``dW[l] = s[l] *
    sum_i c_i dW_i[l]`` -- which is a per-layer strength knob and cannot express
    a preference between inputs. These make the coefficient itself a function of
    the layer -- ``dW[l] = sum_i c_i[l] dW_i[l]`` -- which is what a merge
    weighted by per-layer *contribution* needs. The two compose: a global shape
    times a per-input one."""


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


def parse_input_block_scale(spec: str) -> tuple[int, str, float]:
    """Split ``INDEX:PATTERN=COEFF``.

    Split on the *first* colon, unlike ``--input``: the index is a bare integer
    and what follows is a layer-prefix glob, which never contains one. (The
    rsplit there exists for ``d:/ai/...`` drive letters, which cannot appear
    here.)
    """
    index_text, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise SoupError(f"--input-block-scale expects INDEX:PATTERN=COEFF, got {spec!r}")
    try:
        index = int(index_text)
    except ValueError as e:
        raise SoupError(f"--input-block-scale index is not an integer: {spec!r}") from e
    pattern, coefficient = parse_block_scale(rest)
    return index, pattern, coefficient


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
    """Name the PEFT type(s) the stray keys look like, *and why each is out*.

    "LoHa/LoKr/OFT/DoRA are not plain LoRA" was true and useless: it read as one
    verdict over four types, three of which are now wrong (LoKr merges here, and
    LoHa is a gap rather than an impossibility).
    """
    kinds = {
        name: reason
        for key in keys
        for marker, name, reason in FOREIGN_PEFT_MARKERS
        if marker in key
    }
    if not kinds:
        return ""
    return " (looks like " + "; ".join(f"{name}, which {reason}" for name, reason in sorted(kinds.items())) + ")"


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


def _lokr_key_name(key: str) -> str | None:
    """The LoKr family a key belongs to, or None.

    ``endswith`` on the dotted name, so ``.lokr_w1`` cannot swallow
    ``.lokr_w1_a`` -- the trap a startswith/prefix split would walk into.
    """
    return next((name for name in LOKR_NAMES if key.endswith("." + name)), None)


def build_lokr_layer(
    path: Path,
    prefix: str,
    parts: dict[str, Tensor],
    alpha_tensor: Tensor | None,
) -> LokrLayer:
    """Validate one layer's LoKr factor set and resolve its ``dim`` and ``alpha``.

    Every structural refusal for LoKr lives here, so ``LokrLayer``'s methods can
    assert. The three ``w2`` forms (whole / factored / Tucker) are mutually
    exclusive by construction in ``LoKrModule.initialize_weights``; a file
    carrying two of them is corrupt, not ambiguous, and is refused rather than
    resolved by precedence.
    """
    def need(name: str, why: str) -> Tensor:
        tensor = parts.get(name)
        if tensor is None:
            raise SoupError(f"{path}: LoKr layer {prefix!r} {why} but has no {name}")
        return tensor

    def dims(name: str, tensor: Tensor, expected: int) -> None:
        if tensor.dim() != expected:
            raise SoupError(
                f"{path}: LoKr layer {prefix!r} has {expected}-D {name} expected, "
                f"got shape {tuple(tensor.shape)}"
            )

    w1, w1_a, w1_b = parts.get(LOKR_W1), parts.get(LOKR_W1_A), parts.get(LOKR_W1_B)
    w2, w2_a, w2_b, t2 = parts.get(LOKR_W2), parts.get(LOKR_W2_A), parts.get(LOKR_W2_B), parts.get(LOKR_T2)

    # -- w1: whole, or a_b pair, never both, never neither.
    if w1 is not None and (w1_a is not None or w1_b is not None):
        raise SoupError(
            f"{path}: LoKr layer {prefix!r} carries both a whole lokr_w1 and a "
            "decomposed lokr_w1_a/lokr_w1_b; only one of the two forms is written"
        )
    if w1 is not None:
        dims(LOKR_W1, w1, 2)
    elif w1_a is not None or w1_b is not None:
        w1_a, w1_b = need(LOKR_W1_A, "decomposes w1"), need(LOKR_W1_B, "decomposes w1")
        dims(LOKR_W1_A, w1_a, 2)
        dims(LOKR_W1_B, w1_b, 2)
        if w1_a.shape[1] != w1_b.shape[0]:
            raise SoupError(
                f"{path}: LoKr layer {prefix!r} has a broken w1 pair: "
                f"lokr_w1_a is {tuple(w1_a.shape)}, lokr_w1_b is {tuple(w1_b.shape)}"
            )
    else:
        raise SoupError(f"{path}: LoKr layer {prefix!r} has no w1 factor (lokr_w1 or lokr_w1_a/_b)")

    # -- w2: whole, or a_b pair, or Tucker (t2 + the pair). Same exclusivity.
    if w2 is not None and (w2_a is not None or w2_b is not None or t2 is not None):
        raise SoupError(
            f"{path}: LoKr layer {prefix!r} carries a whole lokr_w2 alongside "
            "lokr_w2_a/lokr_w2_b/lokr_t2; only one w2 form is written"
        )
    if w2 is not None:
        if w2.dim() not in (2, 4):
            raise SoupError(
                f"{path}: LoKr layer {prefix!r} has a whole lokr_w2 of shape "
                f"{tuple(w2.shape)}; only 2-D (Linear) and 4-D (Conv2d) are written"
            )
    else:
        w2_a, w2_b = need(LOKR_W2_A, "decomposes w2"), need(LOKR_W2_B, "decomposes w2")
        dims(LOKR_W2_A, w2_a, 2)
        dims(LOKR_W2_B, w2_b, 2)
        if t2 is not None:
            dims(LOKR_T2, t2, 4)
            if t2.shape[0] != w2_a.shape[0] or t2.shape[1] != w2_b.shape[0]:
                raise SoupError(
                    f"{path}: LoKr layer {prefix!r} has a broken Tucker w2: lokr_t2 is "
                    f"{tuple(t2.shape)}, lokr_w2_a is {tuple(w2_a.shape)}, "
                    f"lokr_w2_b is {tuple(w2_b.shape)}"
                )
        elif w2_a.shape[1] != w2_b.shape[0]:
            raise SoupError(
                f"{path}: LoKr layer {prefix!r} has a broken w2 pair: "
                f"lokr_w2_a is {tuple(w2_a.shape)}, lokr_w2_b is {tuple(w2_b.shape)}"
            )

    # -- dim. Not stored; read off whichever factor shapes carry it, and made to
    # agree. Every candidate below is lokr_dim by construction in
    # LoKrModule.initialize_weights, so a disagreement means the file is not
    # what it claims and there is no defensible scale to pick.
    candidates: list[tuple[str, int]] = []
    if w1_a is not None and w1_b is not None:
        candidates.append((LOKR_W1_A, w1_a.shape[1]))
    if t2 is not None:
        candidates.append((LOKR_T2, t2.shape[0]))
        candidates.append((f"{LOKR_T2}[1]", t2.shape[1]))
    elif w2_a is not None:
        candidates.append((LOKR_W2_A, w2_a.shape[1]))

    if candidates:
        distinct = sorted({value for _name, value in candidates})
        if len(distinct) > 1:
            detail = ", ".join(f"{name} says {value}" for name, value in candidates)
            raise SoupError(
                f"{path}: LoKr layer {prefix!r} disagrees with itself about lokr_dim ({detail}); "
                "the alpha/dim scale is not recoverable from this file"
            )
        dim = distinct[0]
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(dim)
    else:
        # Both factors whole: no inner dim anywhere, so lokr_dim cannot be read
        # back. It does not need to be. LoKrModule.initialize_weights does
        # `self.alpha.fill_(lokr_dim)` in exactly this case, so alpha == dim and
        # the scale is 1.0 -- the stored alpha is deliberately *not* consulted,
        # because on its own it is a numerator with no denominator.
        alpha, dim = 1.0, 1

    if dim < 1:
        raise SoupError(f"{path}: LoKr layer {prefix!r} resolved a lokr_dim of {dim}, which is not a rank")

    return LokrLayer(w1=w1, w1_a=w1_a, w1_b=w1_b, w2=w2, w2_a=w2_a, w2_b=w2_b, t2=t2, alpha=alpha, dim=dim)


def load_lora(
    path: Path | str,
    coefficient: float,
    block_scales: Sequence[tuple[str, float]] = (),
) -> LoadedLora:
    """Read one adapter safetensors file, refusing anything without a closed-form
    additive delta.

    Plain LoRA and LoKr are both understood, per layer, in the same file: they
    reduce to the same ``dW`` and the merge never learns which was which.
    """
    path = Path(path)
    if not path.exists():
        raise SoupError(f"input does not exist: {path}")

    tensors: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        header = dict(f.metadata() or {})
        for key in f.keys():  # noqa: SIM118 -- safe_open is not a Mapping
            tensors[key] = f.get_tensor(key)

    # Bring the file to the canonical namespace BEFORE anything reads a key. A layer is identified here by
    # its key prefix, and which prefix (and which value suffix) a save carries is a choice the training
    # config made -- so without this, two saves of the same run in two output formats are two unrelated
    # adapters, and a merge or a Gram over them is a comparison of nothing.
    try:
        tensors = lora_namespace.canonicalize(tensors, header, path)
    except lora_namespace.NamespaceError as exc:
        raise SoupError(str(exc)) from exc

    layers: dict[str, AdapterLayer] = {}
    bundle: dict[str, Tensor] = {}
    downs: dict[str, Tensor] = {}
    ups: dict[str, Tensor] = {}
    alphas: dict[str, Tensor] = {}
    lokrs: dict[str, dict[str, Tensor]] = {}
    foreign: list[str] = []

    for key, tensor in tensors.items():
        lokr_name = _lokr_key_name(key)
        if key.startswith(BUNDLE_PREFIX):
            bundle[key] = tensor
        elif lokr_name is not None:
            lokrs.setdefault(key[: -(len(lokr_name) + 1)], {})[lokr_name] = tensor
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
            f"{path}: {len(foreign)} key(s) belong to no PEFT type this script can merge "
            f"in delta space{_describe_foreign_keys(foreign)}: {shown}{more}. "
            "Understood: plain LoRA (lora_down/lora_up/alpha), LoKr (lokr_*), and "
            "bundle_emb.*. Nothing else is merged approximately."
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

    for prefix in sorted(lokrs):
        if prefix in layers:
            raise SoupError(
                f"{path}: layer {prefix!r} carries both plain-LoRA and LoKr factors; "
                "one layer has one delta and this file claims two"
            )
        layers[prefix] = build_lokr_layer(path, prefix, lokrs[prefix], alphas.get(prefix))

    orphan_alphas = sorted(set(alphas) - set(layers))
    if orphan_alphas:
        raise SoupError(f"{path}: alpha without factors for layer(s): {', '.join(orphan_alphas[:5])}")

    if not layers:
        raise SoupError(f"{path}: no LoRA or LoKr layers found")

    dtype = next(iter(layers.values())).storage_dtype
    return LoadedLora(
        path=path,
        coefficient=coefficient,
        layers=layers,
        bundle=bundle,
        header=header,
        dtype=dtype,
        file_sha256=_file_sha256(path),
        block_scales=tuple(block_scales),
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
            # Per-input first, so the coefficient is c_i[l]; the merge-wide
            # scale then multiplies the finished sum, as it always has.
            weight = loaded.coefficient * block_scale_for(prefix, loaded.block_scales)
            term = loaded.layers[prefix].delta() * weight
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
        if not isinstance(layer, LoraLayer):
            # ``soup`` refuses this up front, with the offending files named.
            # Repeated here so a direct caller gets the reason, not an
            # AttributeError about a missing ``.down``.
            raise SoupError(f"{loaded.path}: layer {prefix!r} is not plain LoRA. {CONCAT_NEEDS_LORA}")
        factor = (
            loaded.coefficient
            * block_scale_for(prefix, loaded.block_scales)
            * scale
            * layer.scale
        )
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
                    "block_scales": [
                        {"pattern": p, "coefficient": c} for p, c in loaded.block_scales
                    ],
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
        # First, and distinguished from the partial-key-set refusal below: a
        # LoKr input is not a concat that happens to be missing blocks, it is a
        # concat that has no blocks. Falling back to SVD silently would hand
        # back an approximation under a flag that promises exactness.
        offenders = {
            loaded.path.name: sorted(p for p, layer in loaded.layers.items() if not isinstance(layer, LoraLayer))
            for loaded in inputs
        }
        offenders = {name: prefixes for name, prefixes in offenders.items() if prefixes}
        if offenders:
            named = "; ".join(
                f"{name} ({len(prefixes)} layer(s), e.g. {prefixes[0]})"
                for name, prefixes in sorted(offenders.items())
            )
            raise SoupError(f"{CONCAT_NEEDS_LORA} Non-LoRA input(s): {named}")
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
    input_block_scales: Sequence[Sequence[tuple[str, float]]] | None = None,
    log: Callable[[str], None] = _stderr,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """``load_lora`` every spec, then ``soup`` them. The one-call entry point.

    ``input_block_scales`` is parallel to ``specs`` -- one list of per-layer
    patterns per input -- and defaults to none for every input.
    """
    if input_block_scales is None:
        input_block_scales = [()] * len(specs)
    if len(input_block_scales) != len(specs):
        raise SoupError(
            f"input_block_scales has {len(input_block_scales)} entries for {len(specs)} inputs "
            "-- they are positional, so a mismatch would silently scale the wrong adapter"
        )
    inputs = [
        load_lora(path, coefficient, per_input)
        for (path, coefficient), per_input in zip(specs, input_block_scales)
    ]
    return soup(inputs, method=method, rank=rank, block_scales=block_scales, dtype=dtype, log=log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="where to write the merged .safetensors")
    parser.add_argument(
        "--input", action="append", required=True, metavar="FILE:COEFF",
        help="an input LoRA/LoKr and its coefficient; repeat once per input",
    )
    parser.add_argument(
        "--method", choices=[METHOD_SVD, METHOD_CONCAT], default=METHOD_SVD,
        help="svd: re-factor at the target rank (default). concat: exact, output rank is the sum of "
             "input ranks -- plain-LoRA inputs only",
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
    parser.add_argument(
        "--input-block-scale", action="append", default=[], metavar="INDEX:PATTERN=COEFF",
        help="scale ONE input's delta on matching layers, making its coefficient a function "
             "of the layer: dW[l] = sum_i c_i[l] dW_i[l]. INDEX is the 0-based position of the "
             "--input it applies to. Repeatable; composes with --block-scale",
    )
    args = parser.parse_args()

    try:
        specs = [parse_input_spec(spec) for spec in args.input]
        block_scales = [parse_block_scale(spec) for spec in args.block_scale]
        dtype = DTYPE_ALIASES[args.dtype] if args.dtype else None

        per_input: list[list[tuple[str, float]]] = [[] for _ in specs]
        for spec in args.input_block_scale:
            index, pattern, coefficient = parse_input_block_scale(spec)
            if not 0 <= index < len(specs):
                raise SoupError(
                    f"--input-block-scale index {index} is out of range for {len(specs)} input(s)"
                )
            per_input[index].append((pattern, coefficient))

        for i, (path, coefficient) in enumerate(specs):
            extra = f"  [{len(per_input[i])} block scale(s)]" if per_input[i] else ""
            print(f"input   {path}  x {coefficient}{extra}", flush=True)
        state_dict, header = soup_files(
            specs, method=args.method, rank=args.rank, block_scales=block_scales, dtype=dtype,
            input_block_scales=per_input,
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
