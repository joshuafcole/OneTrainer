"""Tests for the velocity-space slider objective (ModelSetupSliderMixin).

The decisive test (`test_adapter_learns_guidance_direction`) is an integration
check, not a unit check of one-line math: it wires the real LoRAModule +
set_multiplier into a toy frozen "model" and optimizes the prompt-pair slider
loss for a few hundred steps, then asserts the trained adapter reproduces the
frozen base's guidance direction  eta * ( v(c+) - v(c-) )  at +strength. If this
passes, the objective + multiplier + autograd path is sound end-to-end.

Runs on small float Linears; the heavy quantization import chain is stubbed:
``python tests/test_slider_objective.py`` or pytest.
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

from modules.modelSetup.mixin.ModelSetupSliderMixin import ModelSetupSliderMixin  # noqa: E402
from modules.module.LoRAModule import LoRAModule  # noqa: E402

D = 16


class _Mixin(ModelSetupSliderMixin):
    pass


def _toy_model(rank=8, alpha=8.0):
    """A frozen base Linear with a real LoRA adapter hooked on. run_velocity(cond)
    returns base(cond) + multiplier*adapter(cond); conditionings are just vectors."""
    torch.manual_seed(0)
    base = nn.Linear(D, D, bias=False)
    base.weight.requires_grad_(False)
    lora = LoRAModule("t.l", base, rank, alpha)
    lora.hook_to_module()

    def run_velocity(cond):
        return base(cond)  # hooked: base.weight @ cond + multiplier * adapter(cond)

    return base, lora, run_velocity


def _conds(n_preserve=1):
    torch.manual_seed(7)
    c_t = torch.randn(1, D)
    pos = [torch.randn(1, D) for _ in range(n_preserve)]
    neg = [torch.randn(1, D) for _ in range(n_preserve)]
    return c_t, pos, neg


def _guidance_target(base, c_t, pos, neg, eta):
    with torch.no_grad():
        delta = sum(base(p) - base(n) for p, n in zip(pos, neg)) / len(pos)
        return eta * delta  # the delta the adapter must add on top of base(c_t)


def test_loss_shape_and_no_grad_base():
    mixin = _Mixin()
    base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()
    loss = mixin._slider_prompt_loss(run_velocity, lora.set_multiplier, c_t, pos, neg, eta=3.0)
    assert loss.ndim == 0 and loss.requires_grad
    # base weight is frozen; only adapter params should receive grad
    loss.backward()
    assert base.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora.parameters())


def test_adapter_learns_guidance_direction():
    mixin = _Mixin()
    eta, strength = 3.0, 1.0
    base, lora, run_velocity = _toy_model(rank=8, alpha=8.0)
    c_t, pos, neg = _conds(n_preserve=2)
    want = _guidance_target(base, c_t, pos, neg, eta)  # = eta * mean(v(c+)-v(c-))

    opt = torch.optim.Adam(lora.parameters(), lr=5e-2)
    first = None
    for _ in range(400):
        opt.zero_grad()
        loss = mixin._slider_prompt_loss(
            run_velocity, lora.set_multiplier, c_t, pos, neg,
            eta=eta, strength=strength, symmetric=True,
        )
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    final = loss.item()

    # learned delta at +strength = full(c_t) - base(c_t)
    lora.set_multiplier(0.0)
    base_t = run_velocity(c_t).detach()
    lora.set_multiplier(strength)
    learned = (run_velocity(c_t).detach() - base_t)

    cos = torch.nn.functional.cosine_similarity(learned.flatten(), want.flatten(), dim=0).item()
    rel_err = ((learned - want).norm() / want.norm().clamp_min(1e-8)).item()

    assert final < first * 0.1, f"loss should drop substantially: {first:.4f} -> {final:.4f}"
    assert cos > 0.99, f"learned delta must align with the guidance direction, cos={cos:.4f}"
    assert rel_err < 0.1, f"learned delta must match magnitude too, rel_err={rel_err:.4f}"


def test_negative_multiplier_mirrors_positive():
    """With a symmetric-trained slider, the -strength delta is the negation of the
    +strength delta (the slider is linear around 0)."""
    mixin = _Mixin()
    eta, strength = 4.0, 1.0
    base, lora, run_velocity = _toy_model(rank=8, alpha=8.0)
    c_t, pos, neg = _conds(n_preserve=1)

    opt = torch.optim.Adam(lora.parameters(), lr=5e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = mixin._slider_prompt_loss(
            run_velocity, lora.set_multiplier, c_t, pos, neg,
            eta=eta, strength=strength, symmetric=True,
        )
        loss.backward()
        opt.step()

    lora.set_multiplier(0.0); base_t = run_velocity(c_t).detach()
    lora.set_multiplier(+strength); pos_delta = run_velocity(c_t).detach() - base_t
    lora.set_multiplier(-strength); neg_delta = run_velocity(c_t).detach() - base_t
    cos = torch.nn.functional.cosine_similarity(pos_delta.flatten(), (-neg_delta).flatten(), dim=0).item()
    assert cos > 0.99, f"-strength delta should mirror +strength delta, cos={cos:.4f}"


def test_rejects_mismatched_pairs():
    mixin = _Mixin()
    base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds(n_preserve=2)
    try:
        mixin._slider_prompt_loss(run_velocity, lora.set_multiplier, c_t, pos, neg[:1], eta=3.0)
    except ValueError:
        return
    raise AssertionError("mismatched positive/negative pair counts must raise")


if __name__ == "__main__":
    test_loss_shape_and_no_grad_base()
    test_adapter_learns_guidance_direction()
    test_negative_multiplier_mirrors_positive()
    test_rejects_mismatched_pairs()
    print("all slider_objective tests passed")
