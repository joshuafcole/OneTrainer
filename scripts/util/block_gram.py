"""Per-layer Gram tensor over N adapters, as JSON on stdout.

This is the primitive the whole attribution story reduces to. For each layer
shared by all N adapters it emits the ``N x N`` matrix of inner products
``<dW_i, dW_j>``. Nothing is interpreted here: no PCA, no clustering, no
normalisation. Those are one matrix multiply each and belong wherever the
analysis lives, not on the training box.

**Why the Gram is enough.** Every quantity the analysis wants is a function of
inner products, and inner products are all that survives the trip:

- summed over layers it is the global Gram, whose eigenvectors are the
  principal directions of the *span of the adapters themselves*. Each such
  direction is a vector of per-adapter coefficients -- that is, a merge recipe
  `lora_soup` can already produce. Atoms come out directly actuatable.
- normalised per layer it is a fixed-length feature vector per layer, which is
  what a clustering over layers needs to find groups that were not chosen in
  advance.
- aggregated over any layer subset it gives that subset's norms, cosines and
  differences, since all three are inner-product expressions.

**Why it is small.** The adapters are hundreds of megabytes; the Gram is
``layers * N * (N+1) / 2`` floats -- for 17 adapters over 224 layers, about
34k numbers. Only the upper triangle is emitted, because the matrix is
symmetric by construction and shipping both halves would double the payload to
transmit a fact about arithmetic.

**Centering is deliberately not done here.** PCA around the mean adapter is the
useful frame (the mean is itself the equal-weight soup, so atoms become
deviations from a merge you would otherwise have made), but double-centering a
Gram matrix is a local operation on an ``N x N`` array. Doing it on the box
would bake one choice of origin into the transport format.

Usage::

    python scripts/util/block_gram.py a.safetensors b.safetensors ... \
        [--granularity LEVEL] [--config PATH]
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

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


class GramError(Exception):
    """The adapters cannot be jointly compared."""


# Two adapters is a comparison; beyond a few dozen the N^2 layer-wise inner
# products stop being the cheap part and the caller almost certainly wants a
# subset anyway.
MAX_ADAPTERS = 64


def _upper_triangle(matrix: list[list[float]], n: int) -> list[float]:
    """Row-major upper triangle including the diagonal.

    The diagonal is each adapter's squared norm over that layer, so norms need
    no separate field."""
    return [matrix[i][j] for i in range(n) for j in range(i, n)]


def gram(
    paths: list[Path],
    config_path: str | None = None,
    granularity: str | None = None,
) -> dict[str, object]:
    """``{layers, per-layer upper-triangle Gram, summed global Gram}``.

    One layer's deltas are materialised at a time and reduced to ``N(N+1)/2``
    scalars immediately. Holding all N full adapters as dense deltas would be
    the one thing this is built to avoid.
    """
    if len(paths) < 2:
        raise GramError(f"need at least 2 adapters, got {len(paths)}")
    if len(paths) > MAX_ADAPTERS:
        raise GramError(f"at most {MAX_ADAPTERS} adapters, got {len(paths)}")

    loaded = [lora_soup.load_lora(p, 1.0) for p in paths]
    n = len(loaded)

    shared_set = set(loaded[0].layers)
    for other in loaded[1:]:
        shared_set &= set(other.layers)
    shared = sorted(shared_set)
    if not shared:
        raise GramError(
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
    total = [[0.0] * n for _ in range(n)]
    for prefix in shared:
        deltas: list[torch.Tensor] = []
        for adapter in loaded:
            # float64: a few hundred per-layer terms at ~1e-6 lose their tail in
            # float32, and an eigendecomposition of the summed Gram is exactly
            # where a lost tail becomes a spurious small component.
            deltas.append(adapter.layers[prefix].delta().to(torch.float64).flatten())
        widths = {d.numel() for d in deltas}
        if len(widths) > 1:
            raise GramError(
                f"layer {prefix!r} has different sizes across adapters ({sorted(widths)}) "
                "— they were trained against different base geometry"
            )
        stacked = torch.stack(deltas)
        layer_gram = (stacked @ stacked.T).tolist()
        for i in range(n):
            for j in range(n):
                total[i][j] += layer_gram[i][j]
        layers.append({
            "layer": prefix,
            "group": group_of[prefix],
            "gram": _upper_triangle(layer_gram, n),
        })

    return {
        "paths": [str(p) for p in paths],
        "adapter_count": n,
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "shared_layer_count": len(shared),
        # Named rather than intersected away: a Gram over a quietly reduced key
        # set describes a different object than the caller asked about.
        "dropped_layers": dropped,
        "unrecognized_parts": list(fitted.unrecognized_parts),
        "total_gram": _upper_triangle(total, n),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapters", nargs="+", help="two or more .safetensors adapters")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity for the per-layer group label")
    args = parser.parse_args()

    for raw in args.adapters:
        if not Path(raw).is_file():
            sys.exit(f"block_gram: no such file: {raw}")
    try:
        out = gram([Path(a) for a in args.adapters], args.config, args.granularity)
    except (GramError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_gram: {e}")
    result_channel.emit_json(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
