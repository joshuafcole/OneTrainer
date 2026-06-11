"""Tests for LoKrModule.init_from_gradient (Kron-GA initialization).

Runs against real LoKrModule instances on small Linear layers. The heavy
quantization import chain (diffusers/gguf/bitsandbytes) is stubbed out since
these tests only touch plain float Linears:
``python tests/test_lokr_grad_init.py``.
"""

import os
import sys
import types

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_stub = types.ModuleType("modules.util.quantization_util")
_stub.get_unquantized_weight = lambda m, dtype, device: m.weight.detach().to(dtype)
_stub.get_weight_shape = lambda m: m.weight.shape
sys.modules["modules.util.quantization_util"] = _stub

from modules.module.LoRAModule import LoKrModule  # noqa: E402
from modules.util.lokr_utils import make_kron, nearest_kron_factors  # noqa: E402


def _make_module(in_dim=128, out_dim=128, dim=4, **kwargs):
    torch.manual_seed(0)
    linear = nn.Linear(in_dim, out_dim)
    defaults = {
        "decompose_both": False,
        "decompose_factor": -1,
        "use_tucker": False,
        "weight_decompose": False,
        "dora_on_output": True,
        "full_matrix": False,
        "train_device": torch.device("cpu"),
        "lokr_vec_trick": True,
    }
    defaults.update(kwargs)
    module = LoKrModule("test.linear", linear, dim, 1.0, **defaults)
    module.hook_to_module()
    return linear, module


def _fake_gradient(module):
    torch.manual_seed(7)
    # A gradient with a dominant Kronecker component plus noise, so the
    # principal factor is well defined.
    w1 = torch.randn(module.out_l, module.in_m)
    w2 = torch.randn(module.out_k, module.in_n)
    g = make_kron(w1, w2)
    return g + 0.05 * torch.randn_like(g)


def test_output_stays_zero_after_init():
    linear, module = _make_module()
    grad = _fake_gradient(module)

    x = torch.randn(3, 11, linear.in_features)
    base_out = nn.functional.linear(x, linear.weight, linear.bias)

    assert module.init_from_gradient(grad)
    out = module.forward(x)
    assert torch.allclose(out, base_out, atol=1e-6), "adapter delta must remain exactly zero"
    assert torch.count_nonzero(module.lokr_w2_b) == 0


def test_w1_aligned_and_norm_matched():
    linear, module = _make_module()
    old_w1 = module.lokr_w1.detach().clone()
    grad = _fake_gradient(module)
    w1_t, _, _ = nearest_kron_factors(grad, module.out_l, module.out_k, module.in_m, module.in_n)

    assert module.init_from_gradient(grad)
    new_w1 = module.lokr_w1.detach()

    assert not torch.allclose(new_w1, old_w1)
    assert torch.allclose(new_w1.norm(), old_w1.norm(), rtol=1e-4), "norm-matched init"
    cos = torch.nn.functional.cosine_similarity(new_w1.flatten(), w1_t.flatten(), dim=0)
    assert cos.abs() > 0.999, f"w1 must align with the principal Kronecker factor, cos={cos}"


def test_w2_a_column_space_aligned():
    linear, module = _make_module()
    old_norm = module.lokr_w2_a.detach().norm()
    grad = _fake_gradient(module)
    _, w2_t, _ = nearest_kron_factors(grad, module.out_l, module.out_k, module.in_m, module.in_n)

    assert module.init_from_gradient(grad)
    new_a = module.lokr_w2_a.detach()
    assert torch.allclose(new_a.norm(), old_norm, rtol=1e-4)

    # Columns must span the top-dim left singular subspace of w2_t.
    u2 = torch.linalg.svd(w2_t, full_matrices=False)[0][:, : module.dim]
    proj = u2 @ (u2.T @ new_a)
    assert torch.allclose(proj, new_a, atol=1e-4), "w2_a must lie in the gradient's principal column space"


def test_decompose_both_variant():
    linear, module = _make_module(in_dim=64, out_dim=64, dim=2, decompose_both=True, decompose_factor=8)
    assert not module.use_w1, "test requires the decomposed-w1 branch"
    grad = _fake_gradient(module)

    x = torch.randn(2, 5, linear.in_features)
    base_out = nn.functional.linear(x, linear.weight, linear.bias)
    assert module.init_from_gradient(grad)
    assert torch.allclose(module.forward(x), base_out, atol=1e-6)


def test_full_matrix_variant():
    linear, module = _make_module(full_matrix=True)
    assert module.use_w2
    grad = _fake_gradient(module)

    x = torch.randn(2, 5, linear.in_features)
    base_out = nn.functional.linear(x, linear.weight, linear.bias)
    assert module.init_from_gradient(grad)
    assert torch.allclose(module.forward(x), base_out, atol=1e-6)
    assert torch.count_nonzero(module.lokr_w2) == 0


def test_rejects_degenerate_gradient():
    linear, module = _make_module()
    assert not module.init_from_gradient(torch.zeros(linear.out_features, linear.in_features))


def test_factor_replay_matches_gradient_init():
    # The returned Van Loan pair, replayed onto a fresh identically-configured
    # module via init_from_factors (the cache-hit path), must produce the same
    # parameters as running init_from_gradient directly.
    _, module = _make_module()
    grad = _fake_gradient(module)
    pair = module.init_from_gradient(grad)
    assert pair is not None
    cached = (pair[0].detach().cpu(), pair[1].detach().cpu())

    _, replayed = _make_module()
    assert replayed.init_from_factors(cached[0], cached[1])
    assert torch.allclose(replayed.lokr_w1.detach(), module.lokr_w1.detach(), atol=1e-6)
    assert torch.allclose(replayed.lokr_w2_a.detach(), module.lokr_w2_a.detach(), atol=1e-6)
    assert torch.count_nonzero(replayed.lokr_w2_b) == 0


def test_factor_replay_rejects_stale_shapes():
    # A cache produced under a different decompose factor (or model) has
    # differently-shaped factors; the replay must reject, not misapply.
    _, module = _make_module()
    assert not module.init_from_factors(
        torch.randn(module.out_l + 1, module.in_m), torch.randn(module.out_k, module.in_n)
    )


def test_first_training_step_moves_toward_gradient_subspace():
    # After Kron-GA init, a step on w2_b must produce a delta correlated with
    # the (clean) gradient direction — the point of the whole exercise.
    linear, module = _make_module()
    grad = _fake_gradient(module)
    assert module.init_from_gradient(grad)

    w1, w2a = module.lokr_w1.detach(), module.lokr_w2_a.detach()
    # Simulated first update of w2_b for descending along `grad`:
    # dL/dw2_b = w2_a^T * (second VL factor of grad contribution); use autograd.
    w2_b = torch.zeros_like(module.lokr_w2_b, requires_grad=True)
    delta = make_kron(w1, w2a @ w2_b)
    loss = (delta * grad).sum()  # directional derivative along grad
    loss.backward()
    step = make_kron(w1, w2a @ w2_b.grad)
    cos = torch.nn.functional.cosine_similarity(step.flatten(), grad.flatten(), dim=0)

    # Compare against the same construction with random (Kaiming) factors.
    _, rand_module = _make_module()
    rw1, rw2a = rand_module.lokr_w1.detach(), rand_module.lokr_w2_a.detach()
    rb = torch.zeros_like(rand_module.lokr_w2_b, requires_grad=True)
    (make_kron(rw1, rw2a @ rb) * grad).sum().backward()
    rand_step = make_kron(rw1, rw2a @ rb.grad)
    rand_cos = torch.nn.functional.cosine_similarity(rand_step.flatten(), grad.flatten(), dim=0)

    assert cos > rand_cos.abs(), f"aligned init must beat random init: {cos} vs {rand_cos}"
    assert cos > 0.5, f"first step should correlate strongly with the gradient, cos={cos}"


if __name__ == "__main__":
    test_output_stays_zero_after_init()
    test_w1_aligned_and_norm_matched()
    test_w2_a_column_space_aligned()
    test_decompose_both_variant()
    test_full_matrix_variant()
    test_rejects_degenerate_gradient()
    test_factor_replay_matches_gradient_init()
    test_factor_replay_rejects_stale_shapes()
    test_first_training_step_moves_toward_gradient_subspace()
    print("all lokr_grad_init tests passed")
