"""Per-layer Kronecker-rank spectrum over N LoKr adapters, as JSON on stdout.

The question this answers: a LoKr delta is
``dW = kron(w1, w2) * (alpha/dim)``, a single Kronecker product. Merging N of
them sums to ``dW = sum_i c_i * kron(w1_i, w2_i)``, which is Kronecker rank
*N*, not 1 -- so the sum is not itself a LoKr, and the only way to get a LoKr
back out is a **fit**: the best rank-1 Kronecker approximation
(``modules.util.lokr_utils.nearest_kron_factors``). How lossy that fit is
depends on how correlated the inputs' ``w1``/``w2`` factors already are, and
that is an empirical question about the population in hand -- this script
measures it, per layer and per block group, without ever forming a dense
delta.

The reduction: Gram matrices are enough
----------------------------------------
``lokr_utils.vl_rearrange`` is the Van Loan-Pitsianis rearrangement:
``R(kron(A, B)) = vec(A) @ vec(B).T`` (``vec`` = row-major flatten), so the
best single-Kronecker fit of ``M`` is the top singular pair of ``R(M)`` --
that identity is exactly what ``nearest_kron_factors`` uses. It extends by
linearity: for the sum,

    R(sum_i c_i * kron(A_i, B_i)) = sum_i c_i * vec(A_i) @ vec(B_i).T
                                   = U @ diag(c) @ V.T

with ``U = [vec(A_1) .. vec(A_N)]`` and ``V = [vec(B_1) .. vec(B_N)]`` as
columns. Thin-QR each: ``U = Q_u R_u``, ``V = Q_v R_v``, giving

    U @ diag(c) @ V.T = Q_u @ (R_u @ diag(c) @ R_v.T) @ Q_v.T

``Q_u`` and ``Q_v`` have orthonormal columns, and left/right-multiplying by a
matrix with orthonormal columns does not change singular values. So the
*entire* Kronecker spectrum of the sum -- however large ``A``/``B`` are --
equals the singular values of the ``N x N`` matrix ``R_u @ diag(c) @ R_v.T``.

``R_u`` and ``R_v`` are never obtained via an actual QR of ``U``/``V`` here
(``U``/``V`` are never materialized as N tall columns): ``R_u.T @ R_u ==
U.T @ U`` is exactly the ``N x N`` Gram matrix of Frobenius inner products
between the ``w1``s (and likewise ``R_v`` from the ``w2``s' Gram), so any
matrix square root of the Gram works -- this uses a symmetric eigendecomposition
rather than a Cholesky factor, because Cholesky demands a *positive definite*
Gram and two resumed runs sharing a near-duplicate ``w1`` (exactly the
lineage case this script exists to look at) sits right on that edge; ``eigh``
degrades gracefully to a rank-deficient Gram, Cholesky does not.

Net effect: the whole per-layer computation is two ``N x N`` Gram matrices
built from ``w1``/``w2`` (tiny -- a LoKr's factors, not its dense delta) and
one ``N x N`` SVD. Verified against a brute-force dense ``R(M)`` SVD in
``tests/test_block_kron_spectrum.py`` before anything was built on it, per
independent, equicorrelated, rectangular-shape, and non-uniform-coefficient
cases -- max deviation ~1e-14 in every case tried, i.e. float64 roundoff, not
approximation error.

What is reported
-----------------
Per layer: the full singular spectrum (length ``N``), the rank-1 fit's
relative Frobenius error (``sqrt(sum_{i>=2} sigma_i^2) / sqrt(sum_i
sigma_i^2)`` -- the energy a single Kronecker term *cannot* reach), and that
term's energy share (``sigma_1^2 / sum sigma_i^2``, i.e. 1 minus the error
squared).

Per block group and overall: the same two numbers, but from *summed energies*
across the group's layers, not an average of per-layer errors. That is the
right combination because each layer's residual/total energy is itself a sum
of squared coordinates (Parseval), and concatenating layers concatenates
coordinates -- so ``sqrt(sum_layers residual) / sqrt(sum_layers total)`` is
the relative error of the same rank-1-per-layer fit strategy applied to the
whole group at once, not a second statistic invented on top. (Mirrors why
``block_gram``'s ``total_gram`` is a sum of per-layer Grams, not their mean.)

What this refuses
------------------
Everything ``lora_soup.load_lora`` refuses, for the same reasons (OFT, DoRA,
LoHa, any other unrecognized key -- see its module docstring): this script
calls it unmodified and propagates its ``SoupError``. On top of that, it
refuses a layer whose Kronecker partition disagrees across adapters (the
``(out_l, in_m)`` / ``(out_k, in_n, *kernel)`` split read off ``factors()``'s
actual tensor shapes) -- that would mean two inputs factorized the same
weight differently (a ``lokr_decompose_factor`` mismatch, most likely), and
there is no defensible common coordinate system to rearrange into. A layer
that is plain LoRA in even one input is not an error -- LoRA merges fine in
delta space, just not through this script, which is Kronecker-structure-only
-- and is named in ``non_lokr_layers`` rather than silently dropped.

Usage::

    python scripts/util/block_kron_spectrum.py \\
        a.safetensors b.safetensors:0.5 c.safetensors \\
        [--granularity LEVEL] [--config PATH]

Each adapter is ``FILE`` (coefficient 1.0, an equal-weight soup) or
``FILE:COEFF``, the same split ``lora_soup.parse_input_spec`` uses for its
``--input``. The coefficient that actually enters ``c_i`` above is
``coefficient * (alpha/dim)`` -- the file's own LoKr scale folded in, so a
plain equal-weight list here really does mean "the merge you would otherwise
have made" (same phrase, same intent, as ``block_gram``'s docstring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_channel  # noqa: E402

# Before torch: the CUDA support libraries it loads write banners to fd 1 from
# C, and this script's stdout is its result. See ``result_channel``.
result_channel.claim()

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


class KronSpectrumError(Exception):
    """The adapters cannot be jointly fit to a single LoKr."""


# Same ceiling as block_gram/block_subspace, same reason: past a few dozen the
# N^2-ish bookkeeping (here: two N x N Grams and an N x N SVD, per layer) stops
# being the cheap part, and a caller with more than that almost certainly wants
# a subset.
MAX_ADAPTERS = 64


def parse_adapter_spec(raw: str) -> tuple[Path, float]:
    """``FILE`` or ``FILE:COEFF`` -- a coefficient defaults to 1.0.

    Rsplit on ``:``, matching ``lora_soup.parse_input_spec``, but tolerant of
    the no-coefficient case that spec always requires: if the text after the
    last ``:`` is not a number (including "empty", including the rest of a
    Windows ``d:/ai/...`` path), the whole argument is the path.
    """
    path_text, sep, coeff_text = raw.rpartition(":")
    if not sep:
        return Path(raw), 1.0
    try:
        return Path(path_text), float(coeff_text)
    except ValueError:
        return Path(raw), 1.0


def _sqrt_factor(gram: torch.Tensor) -> torch.Tensor:
    """``R`` with ``R.T @ R == gram``, via symmetric eigendecomposition.

    Not a Cholesky factor: Cholesky needs ``gram`` positive *definite*, and a
    near-duplicate pair of adapters (two checkpoints a few steps apart in a
    resumed lineage) makes it near-singular by construction -- exactly the
    population this script exists to look at. ``eigh`` degrades to a
    rank-deficient Gram by construction (clamped eigenvalues just zero out the
    corresponding row of ``R``) instead of raising.
    """
    eigval, eigvec = torch.linalg.eigh(gram)
    eigval = eigval.clamp(min=0.0)
    return torch.diag(eigval.sqrt()) @ eigvec.T


def layer_spectrum(w1s: list[Tensor], w2s: list[Tensor], cs: list[float]) -> Tensor:
    """The full Kronecker singular spectrum of ``sum_i c_i * kron(w1_i, w2_i)``,
    descending, length ``len(w1s)`` -- see the module docstring for the
    derivation. ``w2`` may be 2-D (Linear) or 4-D (Conv2d); flattening each to
    ``(out_k, -1)`` before vectorizing folds the kernel into the "in" axis,
    which is the same row-major merge ``LokrLayer.weight_shape()`` +
    ``.reshape(out_features, -1)`` performs on the *dense* delta -- so this
    is the identical partition, not a reinterpretation of it.

    float64 throughout: the Gram entries are sums of a few thousand products
    at whatever scale ``w1``/``w2`` carry (kaiming-init magnitude for the
    "whole" factor, ~1e-3 for the trained one -- see ``test_lora_soup.py``'s
    fixture comment), and the eigendecomposition downstream of a lost tail is
    exactly where a spurious near-zero singular value would appear.
    """
    vec_a = torch.stack([w1.reshape(-1).to(torch.float64) for w1 in w1s])
    vec_b = torch.stack([w2.reshape(w2.shape[0], -1).reshape(-1).to(torch.float64) for w2 in w2s])
    gram_a = vec_a @ vec_a.T
    gram_b = vec_b @ vec_b.T
    r_u = _sqrt_factor(gram_a)
    r_v = _sqrt_factor(gram_b)
    c = torch.diag(torch.tensor(cs, dtype=torch.float64))
    m = r_u @ c @ r_v.T
    return torch.sort(torch.linalg.svdvals(m), descending=True).values


def fit_stats(sigmas: Tensor) -> tuple[float, float, float, float]:
    """``(total_energy, top_energy, rank1_fit_relative_error, top_energy_share)``
    from a singular spectrum. ``total_energy`` is returned (not just the two
    ratios) because it is what group/overall aggregation sums -- see the
    module docstring on why summed energies, not averaged errors, are the
    correct combination.
    """
    energies = sigmas ** 2
    total = float(energies.sum())
    top = float(energies[0]) if energies.numel() else 0.0
    if total <= 0.0:
        # A layer whose merged delta is exactly zero (every coefficient
        # cancelled) has no error to report and no energy to weight it by;
        # calling it a perfect fit is more honest than dividing by zero.
        return 0.0, 0.0, 0.0, 1.0
    residual = total - top
    return total, top, (residual / total) ** 0.5, top / total


def kron_spectrum(
    adapters: list[tuple[Path, float]],
    config_path: str | None = None,
    granularity: str | None = None,
) -> dict[str, object]:
    """``{overall, per-group, per-layer}`` rank-1-Kronecker-fit spectrum for
    an equal- or weighted-coefficient merge of ``adapters``.

    One layer's ``w1``/``w2`` factors are held at a time (never a dense
    delta, and never more than ``N`` small factor tensors) -- the entire
    reason this is cheap enough to run per-checkpoint-set on the box.
    """
    if len(adapters) < 2:
        raise KronSpectrumError(f"need at least 2 adapters, got {len(adapters)}")
    if len(adapters) > MAX_ADAPTERS:
        raise KronSpectrumError(f"at most {MAX_ADAPTERS} adapters, got {len(adapters)}")

    loaded = [lora_soup.load_lora(p, c) for p, c in adapters]
    n = len(loaded)

    shared_set = set(loaded[0].layers)
    for other in loaded[1:]:
        shared_set &= set(other.layers)
    if not shared_set:
        raise KronSpectrumError(
            "no layers common to all adapters — these do not target the same model"
        )
    dropped = {
        str(p): sorted(set(entry.layers) - shared_set)
        for (p, _c), entry in zip(adapters, loaded, strict=True)
        if set(entry.layers) - shared_set
    }

    # Restrict to layers that are LoKr in *every* input. A layer that is plain
    # LoRA even in one input has no w1/w2 to rearrange -- named, not dropped,
    # same reasoning as `dropped_layers` above: a survey over a quietly
    # reduced key set describes a different object than the caller asked for.
    lokr_shared: list[str] = []
    non_lokr: list[str] = []
    for prefix in sorted(shared_set):
        if all(isinstance(entry.layers[prefix], lora_soup.LokrLayer) for entry in loaded):
            lokr_shared.append(prefix)
        else:
            non_lokr.append(prefix)
    if not lokr_shared:
        raise KronSpectrumError(
            "no layer is LoKr in every adapter — a plain-LoRA delta has no Kronecker "
            "structure to rearrange, so there is nothing for this script to fit"
        )

    config = block_groups.load_groups(config_path)
    fitted = block_groups.fit(lokr_shared, config, granularity)
    group_of = {
        prefix: group for group, members in fitted.groups.items() for prefix in members
    }

    layers_out: list[dict[str, object]] = []
    # group -> (summed total_energy, summed residual_energy). Residual, not
    # the error ratio, is what gets summed -- see fit_stats' docstring.
    group_energy: dict[str, list[float]] = {}
    global_total = 0.0
    global_residual = 0.0

    for prefix in lokr_shared:
        contributors = [entry.layers[prefix] for entry in loaded]
        w1_shape: tuple[int, ...] | None = None
        w2_shape: tuple[int, ...] | None = None
        w1s: list[Tensor] = []
        w2s: list[Tensor] = []
        cs: list[float] = []
        for loaded_i, layer in zip(loaded, contributors, strict=True):
            assert isinstance(layer, lora_soup.LokrLayer)
            w1, w2 = layer.factors()
            if w1_shape is None:
                w1_shape, w2_shape = tuple(w1.shape), tuple(w2.shape)
            elif tuple(w1.shape) != w1_shape or tuple(w2.shape) != w2_shape:
                raise KronSpectrumError(
                    f"layer {prefix!r} is factorized differently across adapters: "
                    f"w1 {w1_shape} / w2 {w2_shape} vs w1 {tuple(w1.shape)} / "
                    f"w2 {tuple(w2.shape)} — likely a differing lokr_decompose_factor, "
                    "and there is no common coordinate system to rearrange into"
                )
            w1s.append(w1)
            w2s.append(w2)
            # The file's own alpha/dim scale is not optional here the way it
            # is in merge_deltas: c_i has to be the coefficient this layer's
            # *unscaled* kron(w1, w2) actually enters the sum with, and
            # factors() deliberately returns w1/w2 before that scale is
            # applied (see LokrLayer.weight()).
            cs.append(loaded_i.coefficient * layer.scale)

        sigmas = layer_spectrum(w1s, w2s, cs)
        total, _top, rel_err, top_share = fit_stats(sigmas)

        group = group_of[prefix]
        bucket = group_energy.setdefault(group, [0.0, 0.0])
        bucket[0] += total
        bucket[1] += total * (1.0 - top_share)  # residual energy
        global_total += total
        global_residual += total * (1.0 - top_share)

        layers_out.append({
            "layer": prefix,
            "group": group,
            "spectrum": [round(float(s), 8) for s in sigmas],
            "total_energy": total,
            "top_energy_share": round(top_share, 6),
            "rank1_fit_relative_error": round(rel_err, 6),
        })

    def _summary(total: float, residual: float) -> dict[str, float]:
        if total <= 0.0:
            return {"total_energy": 0.0, "top_energy_share": 1.0, "rank1_fit_relative_error": 0.0}
        return {
            "total_energy": total,
            "top_energy_share": round(1.0 - residual / total, 6),
            "rank1_fit_relative_error": round((residual / total) ** 0.5, 6),
        }

    groups_out = {
        group: {"layer_count": len(fitted.groups[group]), **_summary(total, residual)}
        for group, (total, residual) in group_energy.items()
    }

    return {
        "paths": [str(p) for p, _c in adapters],
        "coefficients": [c for _p, c in adapters],
        "adapter_count": n,
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "shared_layer_count": len(shared_set),
        "lokr_layer_count": len(lokr_shared),
        # Named rather than silently excluded, for the same reason block_gram
        # names its dropped_layers: each describes a way this run's layer set
        # is narrower than "everything shared", and burying that in a count
        # would make a partial survey look like a complete one.
        "dropped_layers": dropped,
        "non_lokr_layers": non_lokr,
        "unrecognized_parts": list(fitted.unrecognized_parts),
        "overall": _summary(global_total, global_residual),
        "groups": groups_out,
        "layers": layers_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "adapters", nargs="+", metavar="FILE[:COEFF]",
        help="two or more .safetensors LoKr adapters; COEFF defaults to 1.0",
    )
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity for the per-layer group label")
    args = parser.parse_args()

    specs: list[tuple[Path, float]] = []
    for raw in args.adapters:
        path, coefficient = parse_adapter_spec(raw)
        if not path.is_file():
            sys.exit(f"block_kron_spectrum: no such file: {path}")
        specs.append((path, coefficient))

    try:
        out = kron_spectrum(specs, args.config, args.granularity)
    except (KronSpectrumError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_kron_spectrum: {e}")
    result_channel.emit_json(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
