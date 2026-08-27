"""Tests for scripts/util/block_kron_spectrum.py.

The load-bearing claim in this file is the first test: that the Gram-matrix
reduction in ``layer_spectrum`` produces the *same* singular values as an
actual dense Van Loan rearrangement + SVD of the merged delta, for every case
that could plausibly break it (independent factors, near-duplicate factors —
the resumed-lineage case this script exists for — rectangular/Conv2d-shaped
factors, and non-uniform coefficients). Everything else in the module is
built on that identity; if it were wrong, every other number here would be a
confident answer to a different question.

Run with::

    python -m pytest tests/test_block_kron_spectrum.py -q
"""

import importlib.util
import os
import sys

import torch

import pytest
from safetensors.torch import save_file

_here = os.path.dirname(__file__)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


block_groups = _load("block_groups", "../scripts/util/block_groups.py")
lora_soup = _load("lora_soup", "../scripts/util/lora_soup.py")
bks = _load("block_kron_spectrum", "../scripts/util/block_kron_spectrum.py")

# lora_soup's own import puts the repo root on sys.path (see its module
# docstring / test_lora_soup.py) so this resolves without extra path munging.
from modules.util.lokr_utils import make_kron, vl_rearrange  # noqa: E402

KronSpectrumError = bks.KronSpectrumError
SoupError = lora_soup.SoupError
BLOCK_PREFIX = "transformer.transformer_blocks"


# --- brute force reference ---------------------------------------------------

def brute_force_spectrum(w1s, w2s, cs, out_l, out_k, in_m, in_n):
    """Dense reference: actually form sum_i c_i * kron(w1_i, w2_i), rearrange
    it (Van Loan), and SVD the whole thing. Everything layer_spectrum computes
    from two N x N Grams is checked against this, which never sees N x N
    anything -- it materializes the full (out_l*out_k, in_m*in_n) matrix."""
    total = torch.zeros(out_l * out_k, in_m * in_n, dtype=torch.float64)
    for w1, w2, c in zip(w1s, w2s, cs, strict=True):
        total += c * make_kron(w1.to(torch.float64), w2.to(torch.float64))
    r = vl_rearrange(total, out_l, out_k, in_m, in_n)
    return torch.sort(torch.linalg.svdvals(r), descending=True).values


def random_factors(n, shape_a, shape_b, seed, correlated=False, rho=0.0):
    gen = torch.Generator().manual_seed(seed)
    if not correlated:
        w1s = [torch.randn(*shape_a, generator=gen, dtype=torch.float64) for _ in range(n)]
    else:
        base = torch.randn(*shape_a, generator=gen, dtype=torch.float64)
        w1s = []
        for _ in range(n):
            noise = torch.randn(*shape_a, generator=gen, dtype=torch.float64)
            w1s.append((rho ** 0.5) * base + ((1 - rho) ** 0.5) * noise)
    w2s = [torch.randn(*shape_b, generator=gen, dtype=torch.float64) for _ in range(n)]
    return w1s, w2s


@pytest.mark.parametrize("n", [2, 4, 7])
def test_gram_reduction_matches_brute_force_independent_factors(n):
    w1s, w2s = random_factors(n, (5, 4), (6, 3), seed=n)
    cs = [1.0] * n
    brute = brute_force_spectrum(w1s, w2s, cs, 5, 6, 4, 3)
    cheap = bks.layer_spectrum(w1s, w2s, cs)

    k = min(len(brute), len(cheap))
    assert torch.allclose(brute[:k], cheap, atol=1e-9)
    # Nothing beyond rank N should carry real energy — the sum truly is
    # Kronecker rank <= N, and the brute-force SVD has more singular values
    # to offer (min(20,18)=18) than the cheap route ever produces (n).
    if len(brute) > k:
        assert float(brute[k:].abs().max()) < 1e-9


@pytest.mark.parametrize("n", [2, 4, 7])
def test_gram_reduction_matches_brute_force_near_duplicate_factors(n):
    """The resumed-lineage case: w1 factors that are almost the same vector.
    This is where Cholesky would be closest to failing outright (near-singular
    Gram) -- eigh should not even notice."""
    w1s, w2s = random_factors(n, (5, 4), (6, 3), seed=100 + n, correlated=True, rho=0.98)
    cs = [1.0] * n
    brute = brute_force_spectrum(w1s, w2s, cs, 5, 6, 4, 3)
    cheap = bks.layer_spectrum(w1s, w2s, cs)
    k = min(len(brute), len(cheap))
    assert torch.allclose(brute[:k], cheap, atol=1e-8)


def test_gram_reduction_matches_brute_force_rectangular_shapes():
    w1s, w2s = random_factors(5, (3, 7), (8, 2), seed=999)
    cs = [1.0] * 5
    brute = brute_force_spectrum(w1s, w2s, cs, 3, 8, 7, 2)
    cheap = bks.layer_spectrum(w1s, w2s, cs)
    k = min(len(brute), len(cheap))
    assert torch.allclose(brute[:k], cheap, atol=1e-9)


def test_gram_reduction_matches_brute_force_weighted_coefficients():
    w1s, w2s = random_factors(4, (5, 4), (6, 3), seed=42)
    cs = [0.3, -0.7, 1.2, 0.05]
    brute = brute_force_spectrum(w1s, w2s, cs, 5, 6, 4, 3)
    cheap = bks.layer_spectrum(w1s, w2s, cs)
    k = min(len(brute), len(cheap))
    assert torch.allclose(brute[:k], cheap, atol=1e-9)


def test_gram_reduction_matches_brute_force_conv2d_kernel_folded():
    """w2 4-D (out_k, in_n, k1, k2): layer_spectrum folds the kernel into the
    'in' axis via reshape(out_k, -1) before vectorizing. Confirmed against a
    brute-force rearrangement of the *flattened-to-2D* delta -- the same
    reshape LokrLayer.delta() performs on a real Conv2d LoKr layer."""
    n = 3
    gen = torch.Generator().manual_seed(7)
    out_l, out_k, in_m, in_n, k1, k2 = 4, 3, 5, 2, 3, 3
    w1s = [torch.randn(out_l, in_m, generator=gen, dtype=torch.float64) for _ in range(n)]
    w2s = [torch.randn(out_k, in_n, k1, k2, generator=gen, dtype=torch.float64) for _ in range(n)]
    cs = [1.0] * n

    w2s_2d = [w2.reshape(out_k, -1) for w2 in w2s]
    brute = brute_force_spectrum(w1s, w2s_2d, cs, out_l, out_k, in_m, in_n * k1 * k2)
    cheap = bks.layer_spectrum(w1s, w2s, cs)
    k = min(len(brute), len(cheap))
    assert torch.allclose(brute[:k], cheap, atol=1e-9)


# --- fit_stats consistency ---------------------------------------------------

def test_top_energy_share_and_fit_error_are_complementary():
    sigmas = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    total, top, err, share = bks.fit_stats(sigmas)
    assert total == pytest.approx(14.0)
    assert top == pytest.approx(9.0)
    assert share == pytest.approx(9.0 / 14.0)
    assert err == pytest.approx((5.0 / 14.0) ** 0.5)
    assert share == pytest.approx(1.0 - err ** 2)


def test_identical_kron_terms_are_exactly_rank_one():
    """N copies of the same (w1, w2) sum to c*kron(w1,w2) for c = sum(coeffs)
    -- still a single Kronecker product, so the fit is exact regardless of N."""
    gen = torch.Generator().manual_seed(3)
    w1 = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    w2 = torch.randn(6, 3, generator=gen, dtype=torch.float64)
    for n in (2, 5, 7):
        sigmas = bks.layer_spectrum([w1] * n, [w2] * n, [1.0] * n)
        _total, _top, err, share = bks.fit_stats(sigmas)
        assert err == pytest.approx(0.0, abs=1e-9)
        assert share == pytest.approx(1.0, abs=1e-9)
        # and only the leading singular value is non-trivial
        assert float(sigmas[1:].abs().max()) < 1e-9 * float(sigmas[0])


def test_fully_independent_factors_are_far_from_rank_one():
    """A coarse sanity bound, not a precise prediction (see the synthetic
    characterisation script for the real curve): truly independent random
    factors should NOT collapse to a good rank-1 fit. Guards against a
    regression that makes the spectrum spuriously flat-or-peaked."""
    w1s, w2s = random_factors(7, (24, 32), (24, 32), seed=55)
    sigmas = bks.layer_spectrum(w1s, w2s, [1.0] * 7)
    _total, _top, err, _share = bks.fit_stats(sigmas)
    assert err > 0.3


# --- fixtures for the end-to-end kron_spectrum() ----------------------------

def _kaiming(*shape, seed):
    import math
    g = torch.Generator().manual_seed(seed)
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    return (torch.rand(*shape, generator=g) * 2 - 1) * bound


def _trained(*shape, seed, scale=1e-3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def write_lokr_adapter(path, prefixes, out_l=4, out_k=3, in_m=4, in_n=3, dim=4, seed=0, alpha=None):
    """A LoKr file (whole w1, factored w2 — LoKrModule's own default) over the
    given layer prefixes, each with independent random factors."""
    state = {}
    for i, prefix in enumerate(prefixes):
        state[f"{prefix}.lokr_w1"] = _kaiming(out_l, in_m, seed=seed + 10 * i)
        state[f"{prefix}.lokr_w2_a"] = _kaiming(out_k, dim, seed=seed + 10 * i + 1)
        state[f"{prefix}.lokr_w2_b"] = _trained(dim, in_n, seed=seed + 10 * i + 2)
        state[f"{prefix}.alpha"] = torch.tensor(float(alpha if alpha is not None else dim))
    save_file(state, str(path))
    return path


def write_lora_adapter(path, prefixes, rank=4, in_features=16, out_features=12, seed=0):
    state = {}
    for i, prefix in enumerate(prefixes):
        g = torch.Generator().manual_seed(seed + i)
        state[f"{prefix}.lora_down.weight"] = torch.randn(rank, in_features, generator=g)
        state[f"{prefix}.lora_up.weight"] = torch.randn(out_features, rank, generator=g) * 1e-3
        state[f"{prefix}.alpha"] = torch.tensor(float(rank))
    save_file(state, str(path))
    return path


def prefixes(blocks=3, leaves=("attn1.to_q", "attn1.to_v")):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


# --- kron_spectrum() end to end ----------------------------------------------

def test_kron_spectrum_end_to_end_matches_layer_spectrum(tmp_path):
    paths = [
        write_lokr_adapter(tmp_path / f"a{i}.safetensors", prefixes(), seed=i * 100)
        for i in range(3)
    ]
    out = bks.kron_spectrum([(p, 1.0) for p in paths])

    assert out["adapter_count"] == 3
    assert out["lokr_layer_count"] == len(prefixes())
    assert not out["non_lokr_layers"]
    assert not out["dropped_layers"]

    # Recompute one layer directly from the loader and check it against the
    # reported entry -- the end-to-end path must agree with the primitive.
    loaded = [lora_soup.load_lora(p, 1.0) for p in paths]
    prefix = prefixes()[0]
    w1s, w2s, cs = [], [], []
    for entry in loaded:
        layer = entry.layers[prefix]
        w1, w2 = layer.factors()
        w1s.append(w1)
        w2s.append(w2)
        cs.append(1.0 * layer.scale)
    expected_sigmas = bks.layer_spectrum(w1s, w2s, cs)
    entry = next(e for e in out["layers"] if e["layer"] == prefix)
    assert torch.allclose(
        torch.tensor(entry["spectrum"], dtype=torch.float64), expected_sigmas, atol=1e-6
    )


def test_group_energy_is_the_sum_not_the_average(tmp_path):
    """The group-level error is sqrt(sum residual / sum total) over its member
    layers, not mean(per-layer error) -- those differ whenever layers carry
    different total energy, which these fixtures are built to guarantee."""
    ps = prefixes(blocks=2, leaves=("attn1.to_q", "attn1.to_v"))
    paths = [
        write_lokr_adapter(tmp_path / f"a{i}.safetensors", ps, seed=i * 37 + 1)
        for i in range(3)
    ]
    out = bks.kron_spectrum([(p, 1.0) for p in paths], granularity="fine")

    for group, summary in out["groups"].items():
        members = [e for e in out["layers"] if e["group"] == group]
        total = sum(e["total_energy"] for e in members)
        residual = sum(e["total_energy"] * (1.0 - e["top_energy_share"]) for e in members)
        expected_err = (residual / total) ** 0.5 if total > 0 else 0.0
        assert summary["rank1_fit_relative_error"] == pytest.approx(expected_err, abs=1e-6)
        assert summary["total_energy"] == pytest.approx(total, rel=1e-9)


def test_layers_plain_lora_in_one_input_are_named_not_dropped(tmp_path):
    """One layer is LoKr in both inputs (fits normally); a second is LoKr in
    one and plain LoRA in the other (no Kronecker structure in common) and
    must be named in non_lokr_layers rather than silently excluded from
    lokr_layer_count with no trace."""
    lokr_leaf = f"{BLOCK_PREFIX}.0.attn1.to_q"
    mixed_leaf = f"{BLOCK_PREFIX}.0.attn1.to_v"

    state_a = {}
    state_a[f"{lokr_leaf}.lokr_w1"] = _kaiming(4, 4, seed=1)
    state_a[f"{lokr_leaf}.lokr_w2_a"] = _kaiming(3, 2, seed=2)
    state_a[f"{lokr_leaf}.lokr_w2_b"] = _trained(2, 3, seed=3)
    state_a[f"{lokr_leaf}.alpha"] = torch.tensor(2.0)
    state_a[f"{mixed_leaf}.lora_down.weight"] = torch.randn(4, 16)
    state_a[f"{mixed_leaf}.lora_up.weight"] = torch.randn(12, 4) * 1e-3
    state_a[f"{mixed_leaf}.alpha"] = torch.tensor(4.0)

    state_b = {}
    state_b[f"{lokr_leaf}.lokr_w1"] = _kaiming(4, 4, seed=4)
    state_b[f"{lokr_leaf}.lokr_w2_a"] = _kaiming(3, 2, seed=5)
    state_b[f"{lokr_leaf}.lokr_w2_b"] = _trained(2, 3, seed=6)
    state_b[f"{lokr_leaf}.alpha"] = torch.tensor(2.0)
    state_b[f"{mixed_leaf}.lokr_w1"] = _kaiming(3, 4, seed=7)
    state_b[f"{mixed_leaf}.lokr_w2_a"] = _kaiming(4, 2, seed=8)
    state_b[f"{mixed_leaf}.lokr_w2_b"] = _trained(2, 3, seed=9)
    state_b[f"{mixed_leaf}.alpha"] = torch.tensor(2.0)

    pa, pb = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(state_a, str(pa))
    save_file(state_b, str(pb))

    out = bks.kron_spectrum([(pa, 1.0), (pb, 1.0)])
    assert out["non_lokr_layers"] == [mixed_leaf]
    assert out["lokr_layer_count"] == 1
    assert {e["layer"] for e in out["layers"]} == {lokr_leaf}


def test_no_lokr_layer_at_all_is_refused(tmp_path):
    ps = prefixes(blocks=1)
    paths = [
        write_lora_adapter(tmp_path / f"a{i}.safetensors", ps, in_features=4, out_features=4, seed=i)
        for i in range(2)
    ]
    with pytest.raises(KronSpectrumError) as e:
        bks.kron_spectrum([(p, 1.0) for p in paths])
    assert "no layer is LoKr" in str(e.value)


def test_dropped_layers_named_like_block_gram(tmp_path):
    p0 = write_lokr_adapter(tmp_path / "a0.safetensors", prefixes(blocks=3), seed=1)
    p1 = write_lokr_adapter(tmp_path / "a1.safetensors", prefixes(blocks=2), seed=2)
    out = bks.kron_spectrum([(p0, 1.0), (p1, 1.0)])
    assert str(p0) in out["dropped_layers"]
    assert out["shared_layer_count"] == len(prefixes(blocks=2))


def test_differing_kronecker_partition_is_refused(tmp_path):
    """Same layer, same final delta shape, different (out_l,in_m)/(out_k,in_n)
    split -- as if the two files used a different lokr_decompose_factor.
    There is no shared coordinate system, so this must refuse, not average."""
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    state_a = {
        f"{prefix}.lokr_w1": _kaiming(4, 4, seed=1),
        f"{prefix}.lokr_w2_a": _kaiming(3, 2, seed=2),
        f"{prefix}.lokr_w2_b": _trained(2, 3, seed=3),
        f"{prefix}.alpha": torch.tensor(2.0),
    }
    # a 2x8 / 6x2-shaped partition of a differently-factored (but same total
    # weight-shape 12x12) layer
    state_b = {
        f"{prefix}.lokr_w1": _kaiming(2, 2, seed=4),
        f"{prefix}.lokr_w2_a": _kaiming(6, 3, seed=5),
        f"{prefix}.lokr_w2_b": _trained(3, 6, seed=6),
        f"{prefix}.alpha": torch.tensor(3.0),
    }
    pa, pb = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(state_a, str(pa))
    save_file(state_b, str(pb))
    with pytest.raises(KronSpectrumError) as e:
        bks.kron_spectrum([(pa, 1.0), (pb, 1.0)])
    assert "factorized differently" in str(e.value)


def test_foreign_peft_types_are_refused_for_the_same_reasons_as_lora_soup(tmp_path):
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    lokr_path = write_lokr_adapter(tmp_path / "a.safetensors", [prefix], seed=1)
    oft_path = tmp_path / "b.safetensors"
    save_file({f"{prefix}.oft_r": torch.randn(4, 4)}, str(oft_path))

    with pytest.raises(SoupError) as e:
        bks.kron_spectrum([(lokr_path, 1.0), (oft_path, 1.0)])
    assert "OFT" in str(e.value)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_adapters_is_refused(tmp_path, count):
    paths = (
        [write_lokr_adapter(tmp_path / "a0.safetensors", prefixes(blocks=1), seed=1)]
        if count
        else []
    )
    with pytest.raises(KronSpectrumError) as e:
        bks.kron_spectrum([(p, 1.0) for p in paths])
    assert "at least 2" in str(e.value)


# --- parse_adapter_spec -------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_path,expected_coeff",
    [
        ("a.safetensors", "a.safetensors", 1.0),
        ("a.safetensors:0.5", "a.safetensors", 0.5),
        ("a.safetensors:-1.5", "a.safetensors", -1.5),
        ("d:/ai/loras/a.safetensors", "d:/ai/loras/a.safetensors", 1.0),
        ("d:/ai/loras/a.safetensors:0.25", "d:/ai/loras/a.safetensors", 0.25),
    ],
)
def test_parse_adapter_spec(raw, expected_path, expected_coeff):
    from pathlib import Path
    path, coeff = bks.parse_adapter_spec(raw)
    assert path == Path(expected_path)
    assert coeff == pytest.approx(expected_coeff)
