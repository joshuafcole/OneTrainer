#!/usr/bin/env python3
"""Summarize an OT_PERF=1 run log (ot_perf.jsonl).

Usage:
    python scripts/analyze_perf.py [ot_perf.jsonl]

Prints:
  * a per-step training breakdown bucketed by latent_tokens, i.e. by resolution
  * a log-log power-law fit of each region against latent_tokens. The exponent is
    the whole point: transformer forward at ~1.0 is linear in sequence length
    (flash-like), at ~2.0 it is quadratic self-attention. Tokens go as
    resolution^2, so the exponent a user feels per side is twice the one on the
    step-total row.
  * caching cost per (label, variation) and in aggregate MP/s
  * recompile and offload-transfer totals

Region columns are discovered from the log rather than hardcoded, so a hook added
in the trainer shows up here without editing this file.
"""

import json
import math
import sys
from collections import defaultdict

# Columns the eye wants first when they exist; anything else follows alphabetically.
_PREFERRED_REGIONS = ("predict", "prior_predict", "transformer_fwd", "text_encode", "backward", "optimizer")
# Emitted by the probe itself rather than by a region, and printed in fixed places.
_DERIVED_MS = ("data_wait_ms", "step_total_ms")
# Header aliases, only so a long key does not blow the column out.
_HEADERS = {"vram_peak_reserved_gb": "vram_gb", "offload_xfers": "offl/step"}


def load(path):
    """Split a JSONL perf log into (step rows, cache rows). Blank lines are skipped."""
    steps, caches = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            (caches if row.get("kind") == "cache" else steps).append(row)
    return steps, caches


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def powerlaw_slope(xs, ys):
    """Least-squares slope/intercept of log(y) against log(x).

    Non-positive pairs are dropped rather than crashing the report -- a region that
    was never timed on some rows should not cost you the rows that were.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=False) if x > 0 and y > 0]
    n = len(pts)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return float("nan"), float("nan")
    slope = num / den
    return slope, my - slope * mx


def region_columns(steps):
    """The `<region>_ms` keys present in the log, preferred ones first."""
    found = {k for row in steps for k in row if k.endswith("_ms") and k not in _DERIVED_MS}
    preferred = [f"{name}_ms" for name in _PREFERRED_REGIONS if f"{name}_ms" in found]
    return preferred + sorted(found.difference(preferred))


def _med(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), int | float)]
    return median(vals) if vals else float("nan")


def report_steps(steps):
    by_tokens = defaultdict(list)
    for s in steps:
        t = s.get("latent_tokens")
        if t is not None:
            by_tokens[t].append(s)
    if not by_tokens:
        print("== no step rows carried latent_tokens; nothing to bucket by resolution ==")
        return

    regions = region_columns(steps)
    columns = [*regions, "data_wait_ms", "step_total_ms", "vram_peak_reserved_gb", "offload_xfers"]
    # Short headers: the `_ms` suffix is in the units line, not in every column.
    headers = [_HEADERS.get(c, c[:-3] if c.endswith("_ms") else c) for c in columns]

    print("== training step breakdown, by latent_tokens (resolution); medians, ms unless noted ==")
    print(("{:>9} {:>4}" + " {:>13}" * len(headers)).format("tokens", "n", *headers))

    fits = defaultdict(lambda: ([], []))
    for t in sorted(by_tokens):
        # The first step at a new token count carries compile/warmup, which is not the
        # steady-state cost being fitted. Drop it when there is anything left.
        rows = by_tokens[t][1:] or by_tokens[t]
        values = [_med(rows, c) for c in columns]
        print(("{:>9} {:>4}" + " {:>13.1f}" * len(values)).format(t, len(rows), *values))
        for col, value in zip(columns, values, strict=True):
            if not math.isnan(value):
                fits[col][0].append(t)
                fits[col][1].append(value)

    printed_fit = False
    for col in [*regions, "step_total_ms"]:
        xs, ys = fits[col]
        if len(xs) < 2:
            continue
        slope, _ = powerlaw_slope(xs, ys)
        if math.isnan(slope):
            continue
        if not printed_fit:
            print("\n  power-law fit vs latent_tokens (1.0 = linear in sequence length, ~2.0 = quadratic):")
            printed_fit = True
        note = f"   -> ~ resolution^{2 * slope:.2f} per side" if col == "step_total_ms" else ""
        print(f"    {col[:-3]:<20} ~ tokens^{slope:.2f}{note}")

    print(f"\n  total dynamo recompiles across run: {sum(r.get('recompiles') or 0 for r in steps)}")


def report_caches(caches):
    if not caches:
        return
    print("\n== caching cost ==")
    per_label = defaultdict(lambda: [0, 0.0])  # label -> [items, seconds]
    for c in caches:
        print("  {:<16} grp{} var{}: {:>4} items in {:>7.1f}s  ({:>6.1f} items/s, {:>6.1f} MP/s)".format(
            c.get("label", "?"), c.get("group"), c.get("variation"),
            c.get("items", 0), c.get("wall_s", 0),
            c.get("items_per_s") or 0, c.get("mp_per_s_encode") or 0))
        per_label[c.get("label")][0] += c.get("items", 0)
        per_label[c.get("label")][1] += c.get("wall_s", 0)
    print()
    for label, (items, secs) in per_label.items():
        print(f"  TOTAL {label}: {items} encodes, {secs:.0f}s wall ({secs / 60:.1f} min)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "ot_perf.jsonl"
    steps, caches = load(path)
    print(f"# {path}: {len(steps)} step rows, {len(caches)} cache rows\n")
    report_steps(steps)
    report_caches(caches)


if __name__ == "__main__":
    main()
