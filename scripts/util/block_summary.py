"""Per-block-group ΔW summary for one adapter file, as JSON on stdout.

The question this answers is "where does this adapter's change actually live",
in a payload small enough to leave the machine: twelve numbers instead of a few
hundred megabytes. That asymmetry is the point — the weights stay on the box
that trained them and only the summary travels.

It is deliberately a *reader*, not a merger. It composes what already exists:
``lora_soup.load_lora`` (which understands LoRA and LoKr alike, and refuses
anything without a closed-form additive delta) and ``block_groups`` (the
band × part taxonomy). Nothing here re-derives either.

Two things it reports that a caller must not have to infer:

- **Coverage.** ``uncovered_layers`` is returned, not asserted away. A real
  checkpoint whose key set disagrees with the assumed anatomy is a finding
  about the taxonomy, and the whole reason to run this against real weights
  rather than a synthetic key set.
- **Share is of *squared* Frobenius norm**, i.e. energy, because that is the
  quantity that is additive across disjoint layer sets. Shares of plain norms
  would not sum to 1 and would quietly overstate small groups.

Usage::

    python scripts/util/block_summary.py <adapter.safetensors> [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_groups  # noqa: E402
import lora_soup  # noqa: E402


def summarize(path: Path, config_path: str | None = None) -> dict[str, object]:
    """``{layer/coverage counts, total norm, per-group norms and shares}``.

    The delta is materialised one layer at a time and reduced to a scalar
    immediately: a summary that had to hold the whole ΔW in memory to report
    twelve numbers would defeat its own purpose on a large adapter.
    """
    config = block_groups.load_groups(config_path)
    loaded = lora_soup.load_lora(path, 1.0)

    prefixes = sorted(loaded.layers)
    block_count = block_groups.observed_block_count(prefixes, config)
    report = block_groups.coverage(prefixes, config)

    energy: dict[str, float] = {}
    counts: dict[str, int] = {}
    kinds: dict[str, int] = {}
    total = 0.0
    for prefix in prefixes:
        layer = loaded.layers[prefix]
        kind = type(layer).__name__.replace("Layer", "").lower()
        kinds[kind] = kinds.get(kind, 0) + 1
        # float64 accumulation: a few hundred per-layer squares at ~1e-6 each
        # lose their tail in float32, and the tail is the small groups.
        sq = float(torch.linalg.vector_norm(layer.delta().to(torch.float64)) ** 2)
        total += sq
        group = block_groups.assign(prefix, config, block_count)
        if group is None:
            continue
        energy[group] = energy.get(group, 0.0) + sq
        counts[group] = counts.get(group, 0) + 1

    groups = []
    for group in block_groups.all_group_names(config):
        sq = energy.get(group, 0.0)
        band, _, part = group.partition(".")
        groups.append({
            "group": group,
            "band": band,
            "part": part,
            "layer_count": counts.get(group, 0),
            "frobenius_norm": sq ** 0.5,
            # Of *squared* norm: energy is what is additive over disjoint layers.
            "share": (sq / total) if total > 0 else 0.0,
        })

    return {
        "path": str(path),
        "block_count": block_count,
        "layer_count": report.total,
        "covered_layer_count": report.covered,
        # Both halves of "exactly once" are reported. A layer claimed by two
        # groups corrupts a share as surely as one claimed by none, and only
        # a real key set can expose either.
        "unassigned_layers": sorted(report.unassigned),
        "multiply_assigned_layers": {k: sorted(v) for k, v in sorted(report.multiply_assigned.items())},
        "exact_once": report.exact_once,
        "adapter_kinds": kinds,
        "total_frobenius_norm": total ** 0.5,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="path to a .safetensors adapter (LoRA or LoKr)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="block_groups.json to use (default: the file beside this script)")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        sys.exit(f"block_summary: no such file: {path}")
    try:
        out = summarize(path, args.config)
    except (lora_soup.SoupError, block_groups.BlockGroupError) as e:
        sys.exit(f"block_summary: {e}")
    json.dump(out, sys.stdout, indent=None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
