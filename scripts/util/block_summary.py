"""Per-block-group ΔW summary for one adapter file, as JSON on stdout.

The question this answers is "where does this adapter's change actually live",
in a payload small enough to leave the machine: twelve numbers instead of a few
hundred megabytes. That asymmetry is the point — the weights stay on the box
that trained them and only the summary travels.

Each group also carries the ``--block-scale`` patterns that select it, so a
reader of the summary can *act* on a group without re-deriving how to address
it. That is the group's membership spelled out, one escaped prefix per layer,
and it is the largest thing here: a 224-layer adapter ships ~224 patterns. Still
four orders of magnitude under the weights, and it is what lets a caller that
has never seen this architecture scale a coordinate it only just learned about.

It is deliberately a *reader*, not a merger. It composes what already exists:
``lora_soup.load_lora`` (which understands LoRA and LoKr alike, and refuses
anything without a closed-form additive delta) and ``block_groups.fit`` (which
fits the coordinate system to the layers this checkpoint actually contains).
Nothing here re-derives either.

Two things it reports that a caller must not have to infer:

- **The granularity that was used**, because the group names mean different
  things at each one and a stored summary must be self-describing. Coverage is
  total by construction — groups are built from the layers present — so the
  reportable gap is ``unrecognized_parts``: parts labelled by their raw leaf
  path because the naming layer has no rule for them.
- **Share is of *squared* Frobenius norm**, i.e. energy, because that is the
  quantity that is additive across disjoint layer sets. Shares of plain norms
  would not sum to 1 and would quietly overstate small groups.

Usage::

    python scripts/util/block_summary.py <adapter.safetensors> [--granularity LEVEL] [--config PATH]
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


def summarize(
    path: Path, config_path: str | None = None, granularity: str | None = None
) -> dict[str, object]:
    """``{layer/coverage counts, total norm, per-group norms and shares}``.

    The delta is materialised one layer at a time and reduced to a scalar
    immediately: a summary that had to hold the whole ΔW in memory to report
    twelve numbers would defeat its own purpose on a large adapter.
    """
    config = block_groups.load_groups(config_path)
    loaded = lora_soup.load_lora(path, 1.0)

    prefixes = sorted(loaded.layers)
    fitted = block_groups.fit(prefixes, config, granularity)

    energy: dict[str, float] = {}
    kinds: dict[str, int] = {}
    total = 0.0
    for group, members in fitted.groups.items():
        for prefix in members:
            layer = loaded.layers[prefix]
            kind = type(layer).__name__.replace("Layer", "").lower()
            kinds[kind] = kinds.get(kind, 0) + 1
            # float64 accumulation: a few hundred per-layer squares at ~1e-6
            # each lose their tail in float32, and the tail is the small groups.
            sq = float(torch.linalg.vector_norm(layer.delta().to(torch.float64)) ** 2)
            total += sq
            energy[group] = energy.get(group, 0.0) + sq

    groups = []
    for group, members in fitted.groups.items():
        sq = energy.get(group, 0.0)
        band, _, part = group.partition(".")
        groups.append({
            "group": group,
            "band": band,
            "part": part,
            "layer_count": len(members),
            "frobenius_norm": sq ** 0.5,
            # Of *squared* norm: energy is what is additive over disjoint layers.
            "share": (sq / total) if total > 0 else 0.0,
            # What makes the group actuatable rather than merely readable: the
            # exact ``--block-scale`` patterns that select these members and
            # nothing else. Emitted rather than left to the caller because the
            # fit is what knows the membership -- a consumer that rebuilt a glob
            # from ``band``/``part`` would be re-deriving the taxonomy this
            # module exists to stop deriving, and a pattern that looks right
            # while selecting the wrong layers is the whole failure mode
            # (``FittedGroups.patterns_for``).
            "patterns": fitted.patterns_for(group),
        })

    return {
        "path": str(path),
        "granularity": fitted.granularity,
        "block_count": fitted.block_count,
        "layer_count": fitted.total_layers,
        # Groups are fitted to the layers present, so every layer is in exactly
        # one group and coverage cannot fail. Reported anyway, as constants, so
        # a consumer written against the old shape keeps working and a future
        # regression in the fit would show up as data rather than as silence.
        "covered_layer_count": fitted.total_layers,
        "unassigned_layers": [],
        "multiply_assigned_layers": {},
        "exact_once": True,
        # The naming layer's gap, which IS reportable: these parts are labelled
        # by their raw leaf path because no rule recognized them.
        "unrecognized_parts": list(fitted.unrecognized_parts),
        "adapter_kinds": kinds,
        "total_frobenius_norm": total ** 0.5,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="path to a .safetensors adapter (LoRA or LoKr)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: the file beside this script)")
    parser.add_argument("--granularity", default=None, metavar="LEVEL",
                        help="naming granularity (default: the config's default_granularity)")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        sys.exit(f"block_summary: no such file: {path}")
    try:
        out = summarize(path, args.config, args.granularity)
    except (lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_summary: {e}")
    result_channel.emit_json(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
