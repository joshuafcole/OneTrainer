"""Synthetic characterisation: rank-1 Kronecker fit error vs. ``w1``-factor
correlation, for a merge of N = 2, 4, 7 LoKr adapters. Answers: how lossy is
it to fit a single LoKr to a merged delta?

Why this exists
----------------
``block_kron_spectrum.py`` measures the fit error on real checkpoints, but a
real checkpoint population is a **lineage** — 10 of a prior 17-run population
resume from each other, and independently-trained pairs have separately been
measured sharing 6.2/16 delta directions at 43x chance (project memory). If
resumed runs' ``w1`` factors are correlated, the merged sum sits closer to
Kronecker rank 1 than a naive "independent adapters" model predicts, and the
fit could be far cheaper than a worst-case estimate says. This script
generates ``w1`` factors at a *controlled* pairwise cosine similarity and
reports how the fit error moves — the curve a real measured Gram (from
``block_kron_spectrum.py``'s output, off the box) gets read against.

Model
-----
``w1_i = sqrt(rho)*base + sqrt(1-rho)*noise_i``, with ``base`` and every
``noise_i`` iid entrywise-unit-variance Gaussian of the same shape. This is
the standard equicorrelated-Gaussian construction: for i != j,
``E[cos(w1_i, w1_j)] = rho``, and ``E[||w1_i||^2]`` is identical across ``i``
and *independent of rho* (the two components' variances sum to 1), so ``rho``
alone moves the correlation without confounding it with a magnitude change.
Verified empirically below (the reported ``rho_measured`` column) rather than
taken on faith.

``w2`` is left fully independent throughout (unit-variance Gaussian,
uncorrelated): the lineage hypothesis under test is specifically that ``w1``
(the "whole" factor under this fork's default LoKr config —
``lokr_decompose_both=False``) inherits across resumed runs, not that ``w2``
does too. Any real ``w2`` correlation could only lower the reported error
further, so this is the conservative (upper-bound-on-error) case.

Shapes: default ``(48, 64)`` for both ``w1`` and a low-rank ``w2`` (48x16 @
16x64, ``lokr_dim=16``) -- this fork's actual defaults
(``TrainConfig.lokr_dim=16``, ``lokr_decompose_both=False``) applied to a
3072-wide Linear layer (``factorization(3072, -1) == (48, 64)``, the common
hidden size for the attention projections this fork trains LoKr against; see
``modules/util/lokr_utils.factorization``). Coefficients are equal (``c_i =
1`` for all ``i``) throughout — relative error is scale-invariant in a common
factor on every ``c_i``, so this is the curve for any *uniform*-weight soup.

⚠️ The absolute baseline (rho=0) is shape-sensitive, not just N-sensitive —
see the module-level ``--w1-rows``/``--w1-cols`` flags and the phase report
this script's numbers feed into. Two independent full-rank Gaussian factors
of ambient dimension d concentrate toward "near-orthogonal" as d grows (Gram
-> const*I), which is the *worst case* (spectrum flat across all N
components); small ambient dimension has more Wishart-statistics spread,
which *lowers* the baseline error via finite-size noise, not correlation. The
qualitative trend (higher N -> higher baseline error; error falls
monotonically as rho -> 1) is robust to this; the specific number at rho=0 is
not, and this script's default shape is the fork's real one specifically so
its rho=0 numbers mean something the box can check.

Usage::

    python scripts/util/kron_fit_vs_correlation.py [--trials 40] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_kron_spectrum as bks  # noqa: E402

DEFAULT_NS = (2, 4, 7)
DEFAULT_RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
DEFAULT_TRIALS = 40

# This fork's real LoKr default (TrainConfig.lokr_dim=16, lokr_decompose_both
# =False) applied to a 3072-wide Linear layer's factorization(3072, -1).
DEFAULT_W1_SHAPE = (48, 64)
DEFAULT_W2_DIM = 16  # lokr_dim: w2 = (48,16) @ (16,64), not full-rank


def make_equicorrelated(n: int, shape: tuple[int, int], rho: float, gen: torch.Generator) -> list[torch.Tensor]:
    base = torch.randn(shape, dtype=torch.float64, generator=gen)
    out = []
    for _ in range(n):
        noise = torch.randn(shape, dtype=torch.float64, generator=gen)
        out.append((rho ** 0.5) * base + ((1 - rho) ** 0.5) * noise)
    return out


def make_independent(n: int, shape: tuple[int, int], gen: torch.Generator) -> list[torch.Tensor]:
    return [torch.randn(shape, dtype=torch.float64, generator=gen) for _ in range(n)]


def make_lowrank_independent(
    n: int, out_dim: int, in_dim: int, dim: int, gen: torch.Generator
) -> list[torch.Tensor]:
    """``w2 = a @ b``, matching a decomposed (non-"whole") LoKr factor: the
    fork's actual default for w2 at ``lokr_dim=16``, not a full-rank matrix."""
    out = []
    for _ in range(n):
        a = torch.randn(out_dim, dim, dtype=torch.float64, generator=gen)
        b = torch.randn(dim, in_dim, dtype=torch.float64, generator=gen)
        out.append(a @ b)
    return out


def trial(
    n: int, rho: float, seed: int, w1_shape: tuple[int, int], w2_dim: int | None
) -> tuple[float, float, float]:
    """One draw: ``(rank1_fit_relative_error, top_energy_share, measured_rho)``."""
    gen = torch.Generator().manual_seed(seed)
    w1s = make_equicorrelated(n, w1_shape, rho, gen)
    w2s = (
        make_independent(n, w1_shape, gen)
        if w2_dim is None
        else make_lowrank_independent(n, w1_shape[0], w1_shape[1], w2_dim, gen)
    )
    cs = [1.0] * n
    sigmas = bks.layer_spectrum(w1s, w2s, cs)
    _total, _top, err, share = bks.fit_stats(sigmas)

    vec_a = torch.stack([w1.reshape(-1) for w1 in w1s])
    normed = vec_a / vec_a.norm(dim=1, keepdim=True)
    cos = normed @ normed.T
    off_diag = cos[~torch.eye(n, dtype=torch.bool)]
    measured_rho = float(off_diag.mean())

    return err, share, measured_rho


def run_curve(
    ns: tuple[int, ...],
    rhos: tuple[float, ...],
    n_trials: int,
    w1_shape: tuple[int, int],
    w2_dim: int | None,
) -> dict[tuple[int, float], dict[str, float]]:
    results: dict[tuple[int, float], dict[str, float]] = {}
    for n in ns:
        for rho in rhos:
            errs, shares, mrhos = [], [], []
            for t in range(n_trials):
                err, share, mrho = trial(n, rho, seed=10_000 * n + int(rho * 1000) + t, w1_shape=w1_shape, w2_dim=w2_dim)
                errs.append(err)
                shares.append(share)
                mrhos.append(mrho)
            errs_t = torch.tensor(errs)
            results[(n, rho)] = {
                "rho_measured": sum(mrhos) / len(mrhos),
                "fit_err_mean": errs_t.mean().item(),
                "fit_err_std": errs_t.std().item(),
                "top_energy_share_mean": sum(shares) / len(shares),
            }
    return results


def rho_needed_for(results: dict[tuple[int, float], dict[str, float]], ns, rhos, threshold: float) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for n in ns:
        hit = next((rho for rho in rhos if results[(n, rho)]["fit_err_mean"] < threshold), None)
        out[n] = hit
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="draws averaged per (n, rho) cell")
    parser.add_argument("--w1-rows", type=int, default=DEFAULT_W1_SHAPE[0])
    parser.add_argument("--w1-cols", type=int, default=DEFAULT_W1_SHAPE[1])
    parser.add_argument(
        "--w2-dim", type=int, default=DEFAULT_W2_DIM,
        help="lokr_dim for a decomposed w2 (a @ b); pass 0 for a full-rank w2 instead",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="fit error considered 'acceptable'")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    w1_shape = (args.w1_rows, args.w1_cols)
    w2_dim = args.w2_dim if args.w2_dim > 0 else None

    results = run_curve(DEFAULT_NS, DEFAULT_RHOS, args.trials, w1_shape, w2_dim)
    needed = rho_needed_for(results, DEFAULT_NS, DEFAULT_RHOS, args.threshold)

    if args.json:
        json.dump({
            "w1_shape": list(w1_shape),
            "w2_dim": w2_dim,
            "trials": args.trials,
            "threshold": args.threshold,
            "curve": {
                f"n={n},rho={rho}": v for (n, rho), v in results.items()
            },
            "rho_needed_for_threshold": {str(n): v for n, v in needed.items()},
        }, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"w1 shape {w1_shape}, w2_dim={w2_dim} (None = full-rank), {args.trials} trials/cell\n")
    print(f"{'n':>3} {'rho_target':>10} {'rho_measured':>12} {'fit_err_mean':>13} {'fit_err_std':>12} {'top_energy':>11}")
    for n in DEFAULT_NS:
        for rho in DEFAULT_RHOS:
            r = results[(n, rho)]
            print(f"{n:>3} {rho:>10.2f} {r['rho_measured']:>12.4f} {r['fit_err_mean']:>13.4f} "
                  f"{r['fit_err_std']:>12.4f} {r['top_energy_share_mean']:>11.4f}")
        print()

    print(f"=== rho needed for fit error < {args.threshold} ===")
    for n in DEFAULT_NS:
        rho = needed[n]
        print(f"n={n}: {'not reached by rho=0.99' if rho is None else f'first at rho>={rho:.2f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
