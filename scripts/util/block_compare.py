"""Per-block-group comparison of two adapters' ΔW, as JSON on stdout.

``block_summary.py`` answers "where does this adapter's change live". It cannot
answer the question a merge actually turns on, which is whether two adapters
that changed the same place changed it the *same way*. Two runs can carry
identical per-group energy profiles and point in unrelated directions; energy
is a magnitude and says nothing about sign or orientation.

So this reports, per group, the three quantities that separate the cases a
merge has to tell apart:

- **cosine** — orientation. Near 1 means the two adapters agree on the
  direction of change and a merge is interpolation. Near 0 means they encode
  unrelated changes, and averaging them destroys both rather than combining
  them. Negative means they actively oppose.
- **scale_ratio** — ‖ΔB‖/‖ΔA‖. Read *with* cosine: cosine≈1 with ratio≠1 is
  the same adapter at a different strength, which is a knob, not a choice.
- **relative_diff** — ‖ΔA−ΔB‖ / max(‖ΔA‖,‖ΔB‖), the scale-aware size of the
  disagreement, so a group with a low cosine but negligible energy cannot pose
  as an important difference.

Everything is accumulated per layer (dot, ‖A‖², ‖B‖²) and never materialised as
a whole-adapter vector: the group cosine is the cosine of the concatenation,
which is exactly the sum of the per-layer dots over the product of the summed
norms. That identity is what keeps a multi-gigabyte comparison inside a few
hundred megabytes of working set.

Layers are matched **by name**. A layer present in one file and not the other
is reported, never silently intersected away: a comparison run over a quietly
reduced key set would report agreement it never measured.

Usage::

    python scripts/util/block_compare.py <a.safetensors> <b.safetensors> \
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


class CompareError(Exception):
    """The two adapters cannot be meaningfully compared."""


def _accumulate(
    a: lora_soup.AdapterLayer, b: lora_soup.AdapterLayer
) -> tuple[float, float, float, float]:
    """``(dot, |a|^2, |b|^2, |a-b|^2)`` for one layer, in float64.

    float64 because a few hundred per-layer terms at ~1e-6 lose their tail in
    float32, and the tail is precisely the small groups this is meant to
    resolve. The deltas are freed as soon as the four scalars are out.
    """
    da = a.delta().to(torch.float64)
    db = b.delta().to(torch.float64)
    if da.shape != db.shape:
        raise CompareError(
            f"shape mismatch: {tuple(da.shape)} vs {tuple(db.shape)} — the two "
            "adapters were trained against different base geometry"
        )
    diff = da - db
    return (
        float((da * db).sum()),
        float((da * da).sum()),
        float((db * db).sum()),
        float((diff * diff).sum()),
    )


def _stats(dot: float, sq_a: float, sq_b: float, sq_diff: float) -> dict[str, float]:
    norm_a, norm_b = sq_a**0.5, sq_b**0.5
    denom = norm_a * norm_b
    return {
        # A group with no energy on one side has no orientation to report; 0.0
        # is the honest value there, not a division that would raise or NaN.
        "cosine": (dot / denom) if denom > 0 else 0.0,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "scale_ratio": (norm_b / norm_a) if norm_a > 0 else 0.0,
        "norm_diff": sq_diff**0.5,
        "relative_diff": (
            sq_diff**0.5 / max(norm_a, norm_b) if max(norm_a, norm_b) > 0 else 0.0
        ),
    }


def compare(
    path_a: Path,
    path_b: Path,
    config_path: str | None = None,
    granularity: str | None = None,
) -> dict[str, object]:
    """Per-group orientation and disagreement between two adapters.

    The coordinate system is fitted to the layers the two files **share**, so
    the groups are the ones the comparison actually covers. Layers unique to
    either side are named separately rather than folded in.
    """
    config = block_groups.load_groups(config_path)
    loaded_a = lora_soup.load_lora(path_a, 1.0)
    loaded_b = lora_soup.load_lora(path_b, 1.0)

    keys_a, keys_b = set(loaded_a.layers), set(loaded_b.layers)
    shared = sorted(keys_a & keys_b)
    if not shared:
        raise CompareError(
            f"no layers in common: {len(keys_a)} vs {len(keys_b)} layers, disjoint "
            "key sets — these adapters do not target the same model"
        )

    fitted = block_groups.fit(shared, config, granularity)

    groups: list[dict[str, object]] = []
    tot_dot = tot_a = tot_b = tot_diff = 0.0
    for group, members in fitted.groups.items():
        g_dot = g_a = g_b = g_diff = 0.0
        for prefix in members:
            dot, sq_a, sq_b, sq_diff = _accumulate(
                loaded_a.layers[prefix], loaded_b.layers[prefix]
            )
            g_dot += dot
            g_a += sq_a
            g_b += sq_b
            g_diff += sq_diff
        tot_dot += g_dot
        tot_a += g_a
        tot_b += g_b
        tot_diff += g_diff
        band, _, part = group.partition(".")
        groups.append({
            "group": group,
            "band": band,
            "part": part,
            "layer_count": len(members),
            **_stats(g_dot, g_a, g_b, g_diff),
        })

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "shared_layer_count": len(shared),
        # Named, not intersected away — a silently reduced key set would report
        # an agreement the comparison never measured.
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "unrecognized_parts": list(fitted.unrecognized_parts),
        **_stats(tot_dot, tot_a, tot_b, tot_diff),
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", help="first adapter (.safetensors)")
    parser.add_argument("b", help="second adapter (.safetensors)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity (default: the config's default)")
    args = parser.parse_args()

    for label, raw in (("a", args.a), ("b", args.b)):
        if not Path(raw).is_file():
            sys.exit(f"block_compare: no such file ({label}): {raw}")
    try:
        out = compare(Path(args.a), Path(args.b), args.config, args.granularity)
    except (CompareError, lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_compare: {e}")
    result_channel.emit_json(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
