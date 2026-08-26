"""Tests for LoRAModule.init_from_gradient (LoRA-GA initialization).

The LoRA analogue of test_lokr_grad_init: aligns lora_down with the top-rank
right singular vectors of the estimated weight gradient while keeping lora_up
at zero, so the adapter output stays exactly zero at init but the first
optimizer step moves the delta toward the rank-r truncation of the gradient.
Runs on small float Linears, CPU-only. Run with
``python -m pytest tests/test_lora_grad_init.py``.
"""

import math
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.module.LoRAModule import LoRAModule  # noqa: E402

IN, OUT, RANK = 128, 128, 4


def _make_module(in_dim=IN, out_dim=OUT, rank=RANK, alpha=1.0):
    torch.manual_seed(0)
    linear = nn.Linear(in_dim, out_dim)
    module = LoRAModule("test.linear", linear, rank, alpha)
    module.hook_to_module()
    return linear, module


def _fake_gradient(in_dim=IN, out_dim=OUT, r=RANK):
    """A gradient with a dominant rank-r structure plus noise, so the top-r
    right singular subspace is well defined."""
    torch.manual_seed(7)
    u = torch.linalg.qr(torch.randn(out_dim, r))[0]
    v = torch.linalg.qr(torch.randn(in_dim, r))[0]
    s = torch.linspace(8.0, 2.0, r)
    g = (u * s) @ v.t()
    return g + 0.02 * torch.randn(out_dim, in_dim)


def _right_singular(g, r):
    return torch.linalg.svd(g, full_matrices=False)[2][:r]  # (r, in)


def test_output_stays_zero_after_init():
    linear, module = _make_module()
    grad = _fake_gradient()

    x = torch.randn(3, 7, linear.in_features)
    base_out = nn.functional.linear(x, linear.weight, linear.bias)

    assert module.init_from_gradient(grad) is not None
    out = module.forward(x)
    assert torch.allclose(out, base_out, atol=1e-6), "adapter delta must remain exactly zero"
    assert torch.count_nonzero(module.lora_up.weight) == 0


def test_down_aligned_and_norm_matched():
    linear, module = _make_module()
    old_norm = module.lora_down.weight.detach().norm()
    grad = _fake_gradient()
    vr = _right_singular(grad, module.rank)  # (r, in), orthonormal rows

    assert module.init_from_gradient(grad) is not None
    a = module.lora_down.weight.detach().float()

    assert torch.allclose(a.norm(), old_norm, rtol=1e-4), "norm-matched init"
    # Every row of lora_down must lie in the gradient's top-r right-singular span.
    proj = (a @ vr.t()) @ vr
    assert torch.allclose(proj, a, atol=1e-4), "lora_down must lie in the principal right-singular subspace"


def test_rejects_degenerate_gradient():
    _, module = _make_module()
    assert module.init_from_gradient(torch.zeros(OUT, IN)) is None


def test_first_training_step_moves_toward_gradient_subspace():
    linear, module = _make_module()
    grad = _fake_gradient()
    assert module.init_from_gradient(grad) is not None

    a = module.lora_down.weight.detach().float()  # (r, in)
    scale = (module.alpha / module.rank).item()

    def first_step(down):
        up = torch.zeros(OUT, module.rank, requires_grad=True)
        delta = (up @ down) * scale
        (delta * grad).sum().backward()
        return up.grad @ down  # weight-update direction (lr/sign factored out)

    cos = torch.nn.functional.cosine_similarity(first_step(a).flatten(), grad.flatten(), dim=0)

    rand = torch.empty_like(a)
    nn.init.kaiming_uniform_(rand, a=math.sqrt(5))
    rand_cos = torch.nn.functional.cosine_similarity(first_step(rand).flatten(), grad.flatten(), dim=0)

    assert cos > rand_cos.abs(), f"aligned init must beat random init: {cos} vs {rand_cos}"
    assert cos > 0.5, f"first step should correlate strongly with the gradient, cos={cos}"


def test_factor_replay_matches_gradient_init():
    _, m1 = _make_module()
    grad = _fake_gradient()
    vh = m1.init_from_gradient(grad)
    assert vh is not None and tuple(vh.shape) == (RANK, IN)

    _, m2 = _make_module()  # identical kaiming init -> identical target norm
    assert m2.init_from_factors(vh.clone())
    assert torch.allclose(m1.lora_down.weight, m2.lora_down.weight, atol=1e-5)


def test_factor_replay_rejects_stale_shapes():
    _, module = _make_module()
    assert not module.init_from_factors(torch.randn(RANK, IN + 1)), "wrong in_features must be rejected"


if __name__ == "__main__":
    test_output_stays_zero_after_init()
    test_down_aligned_and_norm_matched()
    test_rejects_degenerate_gradient()
    test_first_training_step_moves_toward_gradient_subspace()
    test_factor_replay_matches_gradient_init()
    test_factor_replay_rejects_stale_shapes()
    print("all lora_grad_init tests passed")
