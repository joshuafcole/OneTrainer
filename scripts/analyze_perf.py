#!/usr/bin/env python3
"""Summarize an OT_PERF=1 run log (ot_perf.jsonl).

Usage:
    python scripts/analyze_perf.py [ot_perf.jsonl]

Prints:
  * per-step training breakdown, bucketed by latent_tokens (i.e. by resolution)
  * a log-log power-law fit of transformer_fwd vs latent_tokens
    (slope ~1 => attention is linear in tokens / flash-like; slope ~2 => O(tokens^2))
  * a fit of total step time vs latent_tokens (the wall-clock the user feels)
  * caching cost: per (label, variation) and aggregate MP/s
  * recompile + offload-transfer totals
"""

import json
import math
import sys
from collections import defaultdict


def load(path):
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
    """Least-squares slope of log(y) vs log(x). Returns (slope, intercept)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ot_perf.jsonl"
    steps, caches = load(path)
    # ignore the first step at each new token-count (compile / warmup spike)
    print(f"# {path}: {len(steps)} step rows, {len(caches)} cache rows\n")

    by_tokens = defaultdict(list)
    for s in steps:
        t = s.get("latent_tokens")
        if t is not None:
            by_tokens[t].append(s)

    print("== training step breakdown, by latent_tokens (resolution) ==")
    hdr = ("tokens", "n", "fwd_ms", "bwd_ms", "txt_ms", "opt_ms", "wait_ms",
           "step_ms", "vram_gb", "offl/stp")
    print("{:>9} {:>4} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8}".format(*hdr))
    fit_tokens, fit_fwd, fit_step = [], [], []
    for t in sorted(by_tokens):
        rows = by_tokens[t][1:] or by_tokens[t]  # drop warmup/compile step if possible

        def med(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return median(vals) if vals else float("nan")

        fwd, step = med("transformer_fwd_ms"), med("step_total_ms")
        print("{:>9} {:>4} {:>8.1f} {:>8.1f} {:>7.1f} {:>7.1f} {:>8.1f} {:>8.1f} {:>8.2f} {:>8.1f}".format(
            t, len(rows), fwd, med("backward_ms"), med("text_encode_ms"),
            med("optimizer_ms"), med("data_wait_ms"), step,
            med("vram_peak_reserved_gb"), med("offload_xfers")))
        if not math.isnan(fwd):
            fit_tokens.append(t); fit_fwd.append(fwd); fit_step.append(step)

    if len(fit_tokens) >= 2:
        s_fwd, _ = powerlaw_slope(fit_tokens, fit_fwd)
        s_step, _ = powerlaw_slope(fit_tokens, fit_step)
        print(f"\n  transformer_fwd ~ tokens^{s_fwd:.2f}   "
              f"(1.0=linear/flash, ~2.0=O(tokens^2) self-attn)")
        print(f"  step_total      ~ tokens^{s_step:.2f}   "
              f"(tokens ~ resolution^2, so resolution exponent ~ {2 * s_step:.2f})")

    rec = sum(r.get("recompiles") or 0 for r in steps)
    print(f"\n  total dynamo recompiles across run: {rec}")

    if caches:
        print("\n== caching cost ==")
        per_label = defaultdict(lambda: [0, 0.0])  # label -> [items, seconds]
        for c in caches:
            print("  {:<16} grp{} var{}: {:>4} items in {:>7.1f}s  "
                  "({:>6.1f} items/s, {:>6.1f} MP/s)".format(
                      c.get("label", "?"), c.get("group"), c.get("variation"),
                      c.get("items", 0), c.get("wall_s", 0),
                      c.get("items_per_s") or 0, c.get("mp_per_s_encode") or 0))
            per_label[c.get("label")][0] += c.get("items", 0)
            per_label[c.get("label")][1] += c.get("wall_s", 0)
        print()
        for label, (items, secs) in per_label.items():
            print(f"  TOTAL {label}: {items} encodes, {secs:.0f}s wall "
                  f"({secs / 60:.1f} min)")


if __name__ == "__main__":
    main()
