"""Tests for modules/util/grad_estimation.py.

Validates the hook-captured dL/dW against autograd's weight.grad in the three
regimes that matter for the Kron-GA init pass: plain forward, gradient
checkpointing (both reentrant modes), and a PeftBase-style replaced forward
with an additive zero adapter branch. Pure torch: ``python tests/test_grad_estimation.py``.
"""

import importlib.util
import os

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "grad_estimation",
    os.path.join(os.path.dirname(__file__), "..", "modules", "util", "grad_estimation.py"),
)
grad_estimation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grad_estimation)
WeightGradientEstimator = grad_estimation.WeightGradientEstimator


def _make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))


def _named_linears(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}


def _reference_grads(model, steps):
    torch.manual_seed(42)
    model.zero_grad()
    for _ in range(steps):
        x = torch.randn(4, 5, 16)
        model(x).square().mean().backward()
    return {name: m.weight.grad.clone() for name, m in _named_linears(model).items()}


def test_matches_autograd_plain():
    model = _make_model()
    expected = _reference_grads(model, steps=3)

    torch.manual_seed(42)
    est = WeightGradientEstimator()
    est.attach(_named_linears(model))
    with est:
        for _ in range(3):
            x = torch.randn(4, 5, 16)
            model(x).square().mean().backward()
            est.count_step()

    for name, ref in expected.items():
        got = est.grads[name]
        assert torch.allclose(got, ref, atol=1e-5), name
        mean = est.mean_gradient(name)
        assert torch.allclose(mean, ref / 3, atol=1e-5), name


def test_matches_autograd_under_checkpointing():
    for reentrant in (True, False):
        model = _make_model()
        expected = _reference_grads(model, steps=2)

        torch.manual_seed(42)
        est = WeightGradientEstimator()
        est.attach(_named_linears(model))
        dummy = torch.zeros(1, requires_grad=True)
        with est:
            for _ in range(2):
                x = torch.randn(4, 5, 16)
                out = torch.utils.checkpoint.checkpoint(
                    lambda d, inp, _m=model: _m(inp), dummy, x, use_reentrant=reentrant
                )
                out.square().mean().backward()
                est.count_step()

        for name, ref in expected.items():
            assert torch.allclose(est.grads[name], ref, atol=1e-5), (name, reentrant)


def test_matches_autograd_with_replaced_forward():
    # Mimic PeftBase.hook_to_module: instance-attribute forward replacement
    # with an additive adapter branch that is zero at init.
    model = _make_model()
    expected = _reference_grads(model, steps=2)

    for m in _named_linears(model).values():
        orig_forward = m.forward
        zero_up = nn.Parameter(torch.zeros(m.out_features, 4))
        down = torch.randn(4, m.in_features)

        def adapter_forward(x, _orig=orig_forward, _up=zero_up, _down=down):
            return _orig(x) + nn.functional.linear(nn.functional.linear(x, _down), _up)

        m.forward = adapter_forward

    torch.manual_seed(42)
    est = WeightGradientEstimator()
    est.attach(_named_linears(model))
    with est:
        for _ in range(2):
            x = torch.randn(4, 5, 16)
            model(x).square().mean().backward()
            est.count_step()

    for name, ref in expected.items():
        assert torch.allclose(est.grads[name], ref, atol=1e-5), name


def test_matches_autograd_with_compiled_module_under_force_eager():
    # dynamo hard-errors on register_hook inside a compiled region
    # ("Compilation of intermediate hooks requires compiled autograd"), so the
    # estimation pass must run under set_stance("force_eager"). Verify the
    # hooks both fire and produce correct gradients in that stance.
    model = _make_model()
    expected = _reference_grads(model, steps=2)

    linears = _named_linears(model)
    try:
        compiled = torch.compile(model)
    except RuntimeError as e:  # e.g. "torch.compile is not supported on Python 3.14+"
        print(f"skipping compiled-module test: {e}")
        return
    torch.manual_seed(42)
    est = WeightGradientEstimator()
    est.attach(linears)
    with est, torch.compiler.set_stance("force_eager"):
        for _ in range(2):
            x = torch.randn(4, 5, 16)
            compiled(x).square().mean().backward()
            est.count_step()

    for name, ref in expected.items():
        assert torch.allclose(est.grads[name], ref, atol=1e-5), name


def test_hooks_removed_on_exit():
    model = _make_model()
    est = WeightGradientEstimator()
    est.attach(_named_linears(model))
    with est:
        pass
    x = torch.randn(2, 16)
    model(x).sum().backward()
    assert est.step_count == 0 and not est.grads


if __name__ == "__main__":
    test_matches_autograd_plain()
    test_matches_autograd_under_checkpointing()
    test_matches_autograd_with_replaced_forward()
    test_matches_autograd_with_compiled_module_under_force_eager()
    test_hooks_removed_on_exit()
    print("all grad_estimation tests passed")
