"""Tests for the Van Loan-Pitsianis utilities in modules/util/lokr_utils.py.

These guard the index convention that ties vl_rearrange/nearest_kron_factors to
make_kron and to LoKrModule's factor shapes (w1: (out_l, in_m), w2: (out_k, in_n)).
Pure torch, no training stack: ``python tests/test_lokr_utils.py`` or pytest.
"""

import importlib.util
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "lokr_utils",
    os.path.join(os.path.dirname(__file__), "..", "modules", "util", "lokr_utils.py"),
)
lokr_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lokr_utils)

make_kron = lokr_utils.make_kron
vl_rearrange = lokr_utils.vl_rearrange
nearest_kron_factors = lokr_utils.nearest_kron_factors
factorization = lokr_utils.factorization

SHAPES = [
    (2, 3, 4, 5),    # small, all distinct to catch transposed indices
    (4, 4, 4, 4),    # square factors
    (3, 64, 8, 24),  # asymmetric, closer to real layer factorizations
]


def test_rearrange_of_kron_is_rank_one_outer_product():
    torch.manual_seed(0)
    for out_l, out_k, in_m, in_n in SHAPES:
        w1 = torch.randn(out_l, in_m)
        w2 = torch.randn(out_k, in_n)
        r = vl_rearrange(make_kron(w1, w2), out_l, out_k, in_m, in_n)
        expected = w1.flatten().unsqueeze(1) @ w2.flatten().unsqueeze(0)
        assert torch.allclose(r, expected, atol=1e-6), (out_l, out_k, in_m, in_n)


def test_nearest_kron_recovers_exact_kron_product():
    torch.manual_seed(1)
    for out_l, out_k, in_m, in_n in SHAPES:
        w1 = torch.randn(out_l, in_m)
        w2 = torch.randn(out_k, in_n)
        g = make_kron(w1, w2)
        r1, r2, sigma = nearest_kron_factors(g, out_l, out_k, in_m, in_n)
        # Factors are only determined up to a scalar; the product must match.
        assert torch.allclose(make_kron(r1, r2), g, atol=1e-4)
        assert torch.allclose(sigma, w1.norm() * w2.norm(), atol=1e-4)


def test_nearest_kron_is_best_approximation():
    # For a noisy kron product, the recovered factors must approximate the
    # clean signal better than the noisy matrix's distance to it.
    torch.manual_seed(2)
    out_l, out_k, in_m, in_n = 4, 8, 4, 8
    w1 = torch.randn(out_l, in_m)
    w2 = torch.randn(out_k, in_n)
    clean = make_kron(w1, w2)
    noisy = clean + 0.01 * torch.randn_like(clean)
    r1, r2, _ = nearest_kron_factors(noisy, out_l, out_k, in_m, in_n)
    approx = make_kron(r1, r2)
    assert (approx - noisy).norm() <= (clean - noisy).norm() + 1e-5
    assert (approx - clean).norm() <= 2 * (noisy - clean).norm()


def test_matches_lokr_module_factorization_shapes():
    # Mirror LoKrModule.initialize_weights for a typical Linear layer and check
    # the rearrangement round-trips with the exact shapes the module would use.
    torch.manual_seed(3)
    in_dim, out_dim = 3072, 3072
    in_m, in_n = factorization(in_dim, -1)
    if in_m > in_n:
        in_m, in_n = in_n, in_m
    out_l, out_k = factorization(out_dim, -1)
    if out_l > out_k:
        out_l, out_k = out_k, out_l

    w1 = torch.randn(out_l, in_m)
    w2 = torch.randn(out_k, in_n)
    g = make_kron(w1, w2)
    assert g.shape == (out_dim, in_dim)
    r1, r2, _ = nearest_kron_factors(g, out_l, out_k, in_m, in_n)
    assert r1.shape == (out_l, in_m) and r2.shape == (out_k, in_n)
    assert torch.allclose(make_kron(r1, r2), g, atol=1e-3)


if __name__ == "__main__":
    test_rearrange_of_kron_is_rank_one_outer_product()
    test_nearest_kron_recovers_exact_kron_product()
    test_nearest_kron_is_best_approximation()
    test_matches_lokr_module_factorization_shapes()
    print("all lokr_utils tests passed")
