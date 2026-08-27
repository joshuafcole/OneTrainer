"""Top-k singular subspace overlap between N adapters, per layer, as JSON.

``block_gram.py`` reports ⟨dW_i, dW_j⟩ — a whole-vector inner product. That
number is dominated by the *bulk* of the delta, and the bulk of a LoRA delta is
mostly training noise: the original LoRA paper (§7.2) found that two runs from
different random seeds share their **top singular directions** substantially
while agreeing on nothing below them, and concluded the remaining directions
"potentially contain mostly random noises accumulated during training". A
cosine near zero between two adapters is therefore compatible with their top-k
subspaces being strongly aligned, and it is the top-k that a merge acts on.

So this measures the quantity that survives that objection: the normalized
subspace similarity used in that paper,

    φ(A, B, k) = ‖ U_A[:, :k]ᵀ U_B[:, :k] ‖_F² / k         ∈ [0, 1]

for the left singular vectors of each layer's dW, and the same for the right.
φ = 1 is a complete overlap of the two k-dimensional subspaces, φ = 0 complete
separation, and φ(A, A, k) = 1 by construction — the triangle's diagonal is
therefore a free check that the bases came back orthonormal.

**The null is not zero, and reporting φ without it says nothing.** Two
*independent* uniformly-random k-dimensional subspaces of ℝ^m have
E‖UᵀU'‖_F² = k²/m, so E[φ] = k/m. For a 3072-row layer at k=16 that floor is
0.005 — small, but the same order as the whole-adapter cosines that prompted
this measurement, which is exactly why the floor has to be on the page. Both
sides' floors are derivable from what is emitted (``rows``/``cols`` and
``k_effective``); they are not precomputed here for the same reason centering
is not done in ``block_gram`` — a floor is one division, and baking one choice
of null into the transport format is a decision the analysis should own.

⚠️ **φ above the floor proves shared structure, not shared concept.** Every
adapter in a single-concept population could sit above chance simply because
any adaptation of this base model prefers these directions. Separating the two
needs an adapter trained on a *different* concept as a control; nothing in this
script can do it, and a run without that control should say so.

Bases come from ``torch.svd_lowrank``: the delta is exactly low-rank, so a
randomized range finder with oversampling recovers the leading subspace far
more cheaply than a dense SVD (which for a few hundred layers × a few dozen
adapters is hours, not seconds). Oversampling is always clamped below the
smallest rank in play, so no null-space direction — which would be arbitrary,
and would drag φ toward the floor — can enter a basis. The random test matrix
is seeded per layer, because a measurement that moves between runs is not a
measurement.

Usage::

    python scripts/util/block_subspace.py a.safetensors b.safetensors ... \
        [--k 1 4 16] [--granularity LEVEL] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from math import prod
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


class SubspaceError(Exception):
    """The adapters cannot be jointly compared."""


MAX_ADAPTERS = 64

# 1 is the direction the LoRA paper found agrees across seeds; 16 is past where
# a rank-16 run has anything left to say. A ladder rather than a single k
# because the whole question is *where* agreement stops.
DEFAULT_K_VALUES = (1, 4, 16)

SVD_SEED = 0
SVD_OVERSAMPLE = 4
SVD_POWER_ITERATIONS = 4

# φ lives in [0, 1] and is read against a floor of order 1e-3; six decimals is
# three orders of magnitude of headroom under that, and keeps a 224-layer,
# 18-adapter payload in single-digit megabytes rather than tens.
PHI_DECIMALS = 6


def _upper_triangle(matrix: list[list[float]], n: int) -> list[float]:
    """Row-major upper triangle including the diagonal, matching ``block_gram``.

    The diagonal is φ(i, i, k) = 1.0, which is not redundant: it is the only
    in-band evidence that each basis came back orthonormal.
    """
    return [matrix[i][j] for i in range(n) for j in range(i, n)]


def _basis(
    layer: lora_soup.AdapterLayer, q: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], float]:
    """``(U, V, shape, energy)`` for one layer's dW, top-``q`` directions only.

    float64 throughout: the singular vectors below the first few carry values
    whose products are what φ sums, and float32 loses that tail into the same
    noise this is trying to look past.
    """
    dw = layer.delta().to(torch.float64)
    if dw.ndim != 2:
        raise SubspaceError(
            f"expected a 2-D delta, got shape {tuple(dw.shape)} — "
            "lora_soup.AdapterLayer.delta() is documented to flatten to (out, -1)"
        )
    energy = float((dw * dw).sum())
    torch.manual_seed(SVD_SEED)
    u, _s, v = torch.svd_lowrank(dw, q=q, niter=SVD_POWER_ITERATIONS)
    return u, v, (int(dw.shape[0]), int(dw.shape[1])), energy


def subspace(
    paths: list[Path],
    config_path: str | None = None,
    granularity: str | None = None,
    k_values: list[int] | None = None,
) -> dict[str, object]:
    """Per-layer, per-pair top-k subspace overlap over N adapters.

    One layer at a time, and within a layer only the N bases (m×q and n×q) are
    held — never a whole-adapter anything. The pairwise overlap is computed once
    at ``q`` and sliced down for each k, since φ at k is the top-left k×k corner
    of the same product.
    """
    if len(paths) < 2:
        raise SubspaceError(f"need at least 2 adapters, got {len(paths)}")
    if len(paths) > MAX_ADAPTERS:
        raise SubspaceError(f"at most {MAX_ADAPTERS} adapters, got {len(paths)}")

    ks = sorted({int(k) for k in (k_values or DEFAULT_K_VALUES)})
    if ks[0] < 1:
        raise SubspaceError(f"k values must be at least 1, got {ks}")

    loaded = [lora_soup.load_lora(p, 1.0) for p in paths]
    n = len(loaded)

    shared_set = set(loaded[0].layers)
    for other in loaded[1:]:
        shared_set &= set(other.layers)
    shared = sorted(shared_set)
    if not shared:
        raise SubspaceError(
            "no layers common to all adapters — these do not target the same model"
        )
    dropped = {
        str(p): sorted(set(l.layers) - shared_set)
        for p, l in zip(paths, loaded)
        if set(l.layers) - shared_set
    }

    config = block_groups.load_groups(config_path)
    fitted = block_groups.fit(shared, config, granularity)
    group_of = {
        prefix: group for group, members in fitted.groups.items() for prefix in members
    }

    layers: list[dict[str, object]] = []
    for prefix in shared:
        here = [adapter.layers[prefix] for adapter in loaded]
        ranks = [int(layer.rank) for layer in here]

        # The delta is rank-bounded by `.rank`, so every k above the *smallest*
        # rank in play would be asking for directions one adapter does not have.
        # Clamping is reported per k rather than refused: a rank-8 run in a
        # population of rank-32 runs is a fact about the population, not an
        # error, and silently dropping the layer would bias the survey toward
        # whichever layers happen to be uniformly wide.
        # From geometry() rather than a materialised delta: q has to be known
        # before the first SVD, and building a delta twice to learn its shape
        # is the one cost this whole script is arranged to avoid. The loop below
        # checks each realised delta against it, which incidentally pins
        # geometry() and delta() to the same answer.
        out_features, a_trailing, _b_trailing = here[0].geometry()
        shape = (int(out_features), int(prod(a_trailing)))
        rows, cols = shape
        limit = min(min(ranks), rows, cols)
        k_effective = [min(k, limit) for k in ks]
        q = min(max(k_effective) + SVD_OVERSAMPLE, limit)

        lefts: list[torch.Tensor] = []
        rights: list[torch.Tensor] = []
        energies: list[float] = []
        for layer in here:
            u, v, layer_shape, energy = _basis(layer, q)
            if layer_shape != shape:
                raise SubspaceError(
                    f"layer {prefix!r} has different shapes across adapters "
                    f"({shape} vs {layer_shape}) — they were trained against "
                    "different base geometry"
                )
            lefts.append(u)
            rights.append(v)
            energies.append(energy)

        left = [[[0.0] * n for _ in range(n)] for _ in k_effective]
        right = [[[0.0] * n for _ in range(n)] for _ in k_effective]
        for i in range(n):
            for j in range(i, n):
                overlap_l = lefts[i].T @ lefts[j]
                overlap_r = rights[i].T @ rights[j]
                for t, k_eff in enumerate(k_effective):
                    corner_l = overlap_l[:k_eff, :k_eff]
                    corner_r = overlap_r[:k_eff, :k_eff]
                    left[t][i][j] = left[t][j][i] = float((corner_l**2).sum()) / k_eff
                    right[t][i][j] = right[t][j][i] = float((corner_r**2).sum()) / k_eff

        layers.append({
            "layer": prefix,
            "group": group_of[prefix],
            "rows": rows,
            "cols": cols,
            "ranks": ranks,
            "energy": energies,
            "k_effective": k_effective,
            "phi_left": [
                [round(x, PHI_DECIMALS) for x in _upper_triangle(m, n)] for m in left
            ],
            "phi_right": [
                [round(x, PHI_DECIMALS) for x in _upper_triangle(m, n)] for m in right
            ],
        })

    return {
        "paths": [str(p) for p in paths],
        "adapter_count": n,
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "shared_layer_count": len(shared),
        "k_values": ks,
        # Named rather than intersected away, for the same reason block_gram
        # names them: a survey over a quietly reduced key set describes a
        # different object than the caller asked about.
        "dropped_layers": dropped,
        "unrecognized_parts": list(fitted.unrecognized_parts),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapters", nargs="+", help="two or more .safetensors adapters")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity for the per-layer group label")
    parser.add_argument("--k", type=int, nargs="+", default=None, metavar="K",
                        help=f"subspace dimensions to report (default: "
                             f"{' '.join(str(k) for k in DEFAULT_K_VALUES)})")
    args = parser.parse_args()

    for raw in args.adapters:
        if not Path(raw).is_file():
            sys.exit(f"block_subspace: no such file: {raw}")
    try:
        out = subspace(
            [Path(a) for a in args.adapters], args.config, args.granularity, args.k
        )
    except (SubspaceError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_subspace: {e}")
    json.dump(out, sys.stdout, indent=None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
