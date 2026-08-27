"""Tests for the velocity-space Concept-Sliders objective (ModelSetupSliderMixin).

The decisive test (``test_adapter_learns_guidance_direction``) is an integration
check, not a unit check of one-line math: it wires a real LoRAModule +
set_multiplier into a toy frozen "model" and optimizes the prompt-pair slider
loss for a few hundred steps, then asserts the trained adapter reproduces the
frozen base's guidance direction ``eta * (v(c+) - v(c-))`` at +strength. If this
passes, the objective + multiplier + autograd path is sound end to end.

Small float Linears only: CPU, no GPU, no model download. Run with
``python -m pytest tests/test_slider_objective.py`` from the repo root.
"""

import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def _guidance_target(base, pos, neg, eta):
    """The delta the adapter must add on top of base(c_t): eta * mean(v(c+)-v(c-))."""
    with torch.no_grad():
        delta = sum(base(p) - base(n) for p, n in zip(pos, neg, strict=True)) / len(pos)
        return eta * delta


def _learned_delta(lora, run_velocity, c_t, multiplier):
    lora.set_multiplier(0.0)
    base_t = run_velocity(c_t).detach()
    lora.set_multiplier(multiplier)
    return run_velocity(c_t).detach() - base_t


def _train(mixin, lora, run_velocity, c_t, pos, neg, eta, strength, symmetric, steps, lr=5e-2):
    opt = torch.optim.Adam(lora.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = mixin._slider_prompt_loss(
            run_velocity, lora.set_multiplier, c_t, pos, neg,
            eta=eta, strength=strength, symmetric=symmetric,
        )
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    return first, loss.item()


def test_loss_shape_and_no_grad_base():
    mixin = _Mixin()
    base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()
    loss = mixin._slider_prompt_loss(run_velocity, lora.set_multiplier, c_t, pos, neg, eta=3.0)
    assert loss.ndim == 0 and loss.requires_grad
    # the base weight is frozen; only adapter params may receive gradient
    loss.backward()
    assert base.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora.parameters())


def test_guidance_target_is_built_with_the_adapter_off():
    """The frozen-base passes must see multiplier 0.

    Recorded per forward, so an implementation that builds the target with the
    adapter live fails here rather than merely training to a subtly wrong place.
    """
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds(n_preserve=2)

    seen = []

    def recording_run_velocity(cond):
        seen.append(lora.multiplier)
        return run_velocity(cond)

    mixin._slider_prompt_loss(
        recording_run_velocity, lora.set_multiplier, c_t, pos, neg,
        eta=3.0, strength=0.75, symmetric=True,
    )
    # 1 base + 2*2 pair forwards at multiplier 0, then one trained forward at each pole
    assert seen[:5] == [0.0] * 5, f"the guidance target must be built at multiplier 0: {seen}"
    assert seen[5:] == [0.75, -0.75], f"the trained passes must run at +/-strength: {seen}"


def test_guidance_target_does_not_backprop_into_the_base():
    """The target is detached, so the loss's gradient cannot reach it.

    A non-detached target lets the optimizer lower the loss by dragging the
    target toward the prediction. With the toy model that shows up as a gradient
    reaching the frozen base's own graph, so assert directly on the target.
    """
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()

    captured = {}
    real_mse = F.mse_loss

    def capturing_loss(predicted, target):
        captured.setdefault("target", target)
        return real_mse(predicted, target)

    mixin._slider_prompt_loss(
        run_velocity, lora.set_multiplier, c_t, pos, neg,
        eta=3.0, loss_fn=capturing_loss,
    )
    assert not captured["target"].requires_grad, "the guidance target must be detached"


def test_adapter_learns_guidance_direction():
    mixin = _Mixin()
    eta, strength = 3.0, 1.0
    base, lora, run_velocity = _toy_model(rank=8, alpha=8.0)
    c_t, pos, neg = _conds(n_preserve=2)
    want = _guidance_target(base, pos, neg, eta)

    first, final = _train(mixin, lora, run_velocity, c_t, pos, neg, eta, strength, True, steps=400)
    learned = _learned_delta(lora, run_velocity, c_t, strength)

    cos = torch.nn.functional.cosine_similarity(learned.flatten(), want.flatten(), dim=0).item()
    rel_err = ((learned - want).norm() / want.norm().clamp_min(1e-8)).item()

    assert final < first * 0.1, f"loss should drop substantially: {first:.4f} -> {final:.4f}"
    assert cos > 0.99, f"learned delta must align with the guidance direction, cos={cos:.4f}"
    assert rel_err < 0.1, f"learned delta must match magnitude too, rel_err={rel_err:.4f}"


def test_negative_multiplier_mirrors_positive_when_symmetric():
    """A symmetric-trained slider is linear around 0: the -strength delta is the
    negation of the +strength delta."""
    mixin = _Mixin()
    eta, strength = 4.0, 1.0
    _base, lora, run_velocity = _toy_model(rank=8, alpha=8.0)
    c_t, pos, neg = _conds(n_preserve=1)

    _train(mixin, lora, run_velocity, c_t, pos, neg, eta, strength, True, steps=300)

    pos_delta = _learned_delta(lora, run_velocity, c_t, +strength)
    neg_delta = _learned_delta(lora, run_velocity, c_t, -strength)
    cos = torch.nn.functional.cosine_similarity(pos_delta.flatten(), (-neg_delta).flatten(), dim=0).item()
    assert cos > 0.99, f"-strength delta should mirror +strength delta, cos={cos:.4f}"


def test_symmetric_off_trains_only_the_positive_pole():
    """Without the symmetric pass only the + pole is supervised.

    A LoRA delta is linear in the multiplier, so the - pole is still the negation
    numerically -- what the symmetric pass buys is that the - pole is *in the
    loss*. Assert on the loss instead: turning it off must halve the number of
    trained forwards and drop the -strength MSE term.
    """
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()

    multipliers = []

    def recording_run_velocity(cond):
        multipliers.append(lora.multiplier)
        return run_velocity(cond)

    mixin._slider_prompt_loss(
        recording_run_velocity, lora.set_multiplier, c_t, pos, neg,
        eta=3.0, strength=1.0, symmetric=False,
    )
    assert -1.0 not in multipliers, f"symmetric=False must not run the -strength pass: {multipliers}"

    multipliers.clear()
    mixin._slider_prompt_loss(
        recording_run_velocity, lora.set_multiplier, c_t, pos, neg,
        eta=3.0, strength=1.0, symmetric=True,
    )
    assert -1.0 in multipliers, f"symmetric=True must run the -strength pass: {multipliers}"


def test_preservation_set_averages_the_pairs():
    """CS Eq. 8: the guidance direction is the MEAN over the preservation set, not
    the first pair and not the sum."""
    mixin = _Mixin()
    base, lora, run_velocity = _toy_model()
    eta = 3.0
    c_t, pos, neg = _conds(n_preserve=3)

    captured = {}
    real_mse = F.mse_loss

    def capturing_loss(predicted, target):
        captured.setdefault("target", target)
        return real_mse(predicted, target)

    mixin._slider_prompt_loss(
        run_velocity, lora.set_multiplier, c_t, pos, neg,
        eta=eta, symmetric=False, loss_fn=capturing_loss,
    )

    lora.set_multiplier(0.0)
    with torch.no_grad():
        v_base = run_velocity(c_t)
        deltas = [run_velocity(p) - run_velocity(n) for p, n in zip(pos, neg, strict=True)]
    want_mean = v_base + eta * (sum(deltas) / len(deltas))
    want_first = v_base + eta * deltas[0]
    want_sum = v_base + eta * sum(deltas)

    got = captured["target"]
    assert torch.allclose(got, want_mean, atol=1e-5), "target must use the mean over the preservation set"
    assert not torch.allclose(got, want_first, atol=1e-4), "target must not be the first pair alone"
    assert not torch.allclose(got, want_sum, atol=1e-4), "target must not be the un-averaged sum"


def test_adapter_is_left_at_the_resting_multiplier():
    """Sampling-during-training runs the model with whatever multiplier the last
    step left set, so the objective must not leave it parked at -strength."""
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()
    mixin._slider_prompt_loss(
        run_velocity, lora.set_multiplier, c_t, pos, neg, eta=3.0, strength=0.5, symmetric=True,
    )
    assert lora.multiplier == 1.0

    # ... and it is restored even when a forward raises partway through.
    def exploding_run_velocity(cond):
        raise RuntimeError("boom")

    lora.set_multiplier(0.25)
    with pytest.raises(RuntimeError, match="boom"):
        mixin._slider_prompt_loss(
            exploding_run_velocity, lora.set_multiplier, c_t, pos, neg, eta=3.0,
        )
    assert lora.multiplier == 1.0


def test_rejects_mismatched_pairs():
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds(n_preserve=2)
    with pytest.raises(ValueError):
        mixin._slider_prompt_loss(run_velocity, lora.set_multiplier, c_t, pos, neg[:1], eta=3.0)
    with pytest.raises(ValueError):
        mixin._slider_prompt_loss(run_velocity, lora.set_multiplier, c_t, [], [], eta=3.0)


def test_rejects_zero_strength():
    """strength 0 disables the adapter entirely: every trained pass would be the
    frozen base and the run would report a plausible, slowly-falling loss while
    learning nothing."""
    mixin = _Mixin()
    _base, lora, run_velocity = _toy_model()
    c_t, pos, neg = _conds()
    with pytest.raises(ValueError):
        mixin._slider_prompt_loss(
            run_velocity, lora.set_multiplier, c_t, pos, neg, eta=3.0, strength=0.0,
        )
