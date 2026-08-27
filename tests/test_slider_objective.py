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


def _base_weight_grad(symmetric, warm_adapter=False, eta=3.0, strength=1.0):
    """||dL/dW|| of the frozen base weight -- the quantity GA init aligns to."""
    base, lora, run_velocity = _toy_model()
    base.weight.requires_grad_(True)
    if warm_adapter:
        with torch.no_grad():
            lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.05)

    c_t, pos, neg = _conds()
    loss = _Mixin()._slider_prompt_loss(
        run_velocity=run_velocity,
        set_multiplier=lora.set_multiplier,
        target_cond=c_t, positive_conds=pos, negative_conds=neg,
        eta=eta, strength=strength, symmetric=symmetric,
    )
    base.weight.grad = None
    loss.backward()
    return base.weight.grad.norm().item()


def test_symmetric_slider_cancels_the_base_weight_gradient():
    """Why GA init is refused for a symmetric slider -- and why that is not a bug.

    GA init re-initializes the adapter factors from a rank truncation of dL/dW of
    the *frozen base weight*. A symmetric slider fits the two poles v_base+eta*d
    and v_base-eta*d, whose residuals are equal and opposite; the base path
    contributes the same Jacobian to both, so the sum cancels. There is nothing to
    align to, and GA init would be aligning to float noise.

    Locked as a test because the skip is otherwise justified only by a comment,
    and because the asymmetric arm below is the evidence for the other half of
    that comment: sliders are not categorically incompatible with GA init, only
    symmetric ones are.
    """
    symmetric = _base_weight_grad(symmetric=True)
    asymmetric = _base_weight_grad(symmetric=False)

    assert asymmetric > 1e-2, "an asymmetric slider must have a real base-weight gradient"
    assert symmetric < asymmetric * 1e-6, (
        f"symmetric ||dL/dW||={symmetric:.3e} should cancel to noise "
        f"against asymmetric {asymmetric:.3e}"
    )


def test_the_cancellation_is_not_a_cold_start_artifact():
    """It does not warm up, so 'run GA init a few steps in' is not a workaround.

    The cancellation is a property of fitting two opposite poles, not of the
    zero-initialized adapter. Distinguishing these matters: the counterexample
    objective's beta calibration *does* recover once the adapter warms, and the
    natural fix there (wait, then measure) does nothing here.
    """
    cold = _base_weight_grad(symmetric=True, warm_adapter=False)
    warm = _base_weight_grad(symmetric=True, warm_adapter=True)
    reference = _base_weight_grad(symmetric=False)

    assert warm < reference * 1e-4, (
        f"warm symmetric ||dL/dW||={warm:.3e} is still noise against {reference:.3e}; "
        "if this fails the cancellation warmed up and the skip needs revisiting"
    )
    assert cold < reference * 1e-4


def test_symmetry_does_not_weaken_the_adapters_own_gradient():
    """The flip side, and the reason the skip message must not say 'no gradient'.

    Only GA's estimand vanishes. The gradient w.r.t. the adapter's own factors is
    exactly *doubled* by symmetry, because the second pole flips the multiplier's
    sign as well as the residual's -- two sign flips, a net addition. A reader who
    takes 'no step-0 gradient' at face value would conclude a symmetric slider
    cannot train, which is the opposite of true.
    """
    def adapter_grad(symmetric):
        base, lora, run_velocity = _toy_model()
        c_t, pos, neg = _conds()
        loss = _Mixin()._slider_prompt_loss(
            run_velocity=run_velocity, set_multiplier=lora.set_multiplier,
            target_cond=c_t, positive_conds=pos, negative_conds=neg,
            eta=3.0, strength=1.0, symmetric=symmetric,
        )
        lora.lora_up.weight.grad = None
        loss.backward()
        return lora.lora_up.weight.grad.norm().item()

    one_pole = adapter_grad(symmetric=False)
    two_poles = adapter_grad(symmetric=True)

    assert one_pole > 1e-3
    assert two_poles == pytest.approx(2.0 * one_pole, rel=1e-4)


# ---------------------------------------------------------------------------
# the IMAGE regime's objective: coordinate-scaled reconstruction
#
# The toy dataset is built so that the *right* answer is known in closed form.
# Each "image" x_i has a coordinate l_i, and its target is the frozen base's
# output plus m_i * (A x_i) for one fixed low-rank A. An adapter that has learned
# A reproduces every sample at once, and -- the claim worth testing -- also at
# coordinates the dataset never contained.
# ---------------------------------------------------------------------------

def _coordinate_model(rank=4):
    torch.manual_seed(0)
    base = nn.Linear(D, D, bias=False)
    base.weight.requires_grad_(False)
    lora = LoRAModule("t.l", base, rank, float(rank))
    lora.hook_to_module()
    return base, lora


def _coordinate_batch(coords, gain=1.0, rank=4, seed=11):
    """(inputs, the closed-form ideal adapter, per-sample multipliers)."""
    torch.manual_seed(seed)
    xs = [torch.randn(1, D) for _ in coords]
    u = torch.randn(D, rank) * 0.5
    v = torch.randn(rank, D) * 0.5
    a_matrix = u @ v

    def apply_a(x):
        return x @ a_matrix.T

    return xs, apply_a, [gain * c for c in coords]


def _coordinate_targets(base, lora, xs, apply_a, multipliers):
    lora.set_multiplier(0.0)
    with torch.no_grad():
        return [(base(x) + m * apply_a(x)).detach() for x, m in zip(xs, multipliers, strict=True)]


def _run_velocity_for(base, xs):
    def run_velocity(indices):
        return base(torch.cat([xs[i] for i in indices], dim=0))
    return run_velocity


def test_coordinate_adapter_learns_a_response_that_scales_with_the_coordinate():
    """The decisive test: what the regime claims is a *calibrated* axis.

    Fitting several coordinates at once is only possible if the adapter's effect
    is linear in the multiplier -- so the check is not "the loss went down" but
    "the trained adapter is right at a coordinate the dataset never contained".
    That is precisely what the user does at inference when they drag the slider
    somewhere between the labels they captioned.
    """
    mixin = _Mixin()
    base, lora = _coordinate_model()
    # Enough images that they span the input space -- otherwise the adapter can
    # fit every training sample while agreeing with nothing off their span, and
    # the generalization claim below would be untested rather than true.
    coords = [-2.0, -1.0, 1.0, 2.0] * 12
    xs, apply_a, multipliers = _coordinate_batch(coords, gain=0.5)
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    run_velocity = _run_velocity_for(base, xs)

    opt = torch.optim.Adam(lora.parameters(), lr=5e-2)
    first = None
    for _ in range(800):
        opt.zero_grad()
        loss = mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 1e-3, f"loss {first:.4f} -> {loss.item():.4f}"

    # an unseen coordinate, and an unseen input
    torch.manual_seed(99)
    probe = torch.randn(1, D)
    lora.set_multiplier(0.0)
    base_out = base(probe).detach()
    for unseen in (0.35, -1.4, 3.0):
        lora.set_multiplier(unseen)
        learned = base(probe).detach() - base_out
        want = unseen * apply_a(probe)
        assert torch.allclose(learned, want, atol=5e-2, rtol=5e-2), (
            f"at multiplier {unseen} the adapter is not on the axis it was calibrated to: "
            f"{learned} vs {want}"
        )


def test_grouping_is_exactly_the_per_sample_loop():
    """Samples sharing a multiplier are batched into one forward. That is an
    optimization, so it has to be arithmetically invisible -- weighting each
    group's mean by its size is what makes it so, and dropping that weight is a
    silent reweighting of the dataset toward whichever coordinate is rarest."""
    mixin = _Mixin()
    base, lora = _coordinate_model()
    coords = [1.0, -1.0, 1.0, 1.0, -1.0]
    xs, apply_a, multipliers = _coordinate_batch(coords)
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    with torch.no_grad():
        lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.1)
    run_velocity = _run_velocity_for(base, xs)

    grouped = mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers)

    one_at_a_time = []
    for i, m in enumerate(multipliers):
        lora.set_multiplier(m)
        one_at_a_time.append(F.mse_loss(run_velocity([i]), targets[i]))
    looped = sum(one_at_a_time) / len(one_at_a_time)

    assert torch.allclose(grouped, looped, atol=1e-6), f"{grouped.item()} != {looped.item()}"


def test_samples_sharing_a_multiplier_run_as_one_forward():
    """The multiplier belongs to the adapter, not to a row, so a batch can only be
    split by distinct multiplier -- but binary poles are the common case and
    collapse to two forwards, not five."""
    mixin = _Mixin()
    base, lora = _coordinate_model()
    coords = [1.0, -1.0, 1.0, 1.0, -1.0]
    xs, apply_a, multipliers = _coordinate_batch(coords)
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)

    seen = []
    inner = _run_velocity_for(base, xs)

    def run_velocity(indices):
        seen.append((lora.multiplier, list(indices)))
        return inner(indices)

    mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers)
    assert seen == [(1.0, [0, 2, 3]), (-1.0, [1, 4])]


def test_the_multiplier_is_set_before_the_forward():
    """Recorded per forward: an implementation that sets the multiplier after
    running the model would train every sample at the previous coordinate and
    still report a falling loss."""
    mixin = _Mixin()
    base, lora = _coordinate_model()
    xs, apply_a, multipliers = _coordinate_batch([2.0, -3.0])
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    inner = _run_velocity_for(base, xs)

    seen = []

    def run_velocity(indices):
        seen.append(lora.multiplier)
        return inner(indices)

    lora.set_multiplier(0.75)
    mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers)
    assert seen == multipliers


def test_a_zero_coordinate_sample_never_reaches_the_model():
    """At multiplier 0 the adapter is disabled, so such a term is a constant with
    respect to every trained parameter: it would cost a forward, add a constant to
    the reported loss, and divide the real gradient down. Measured directly below,
    so this is arithmetic rather than a policy about neutral images."""
    mixin = _Mixin()
    base, lora = _coordinate_model()
    coords = [1.0, 0.0, -1.0]
    xs, apply_a, multipliers = _coordinate_batch(coords)
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    with torch.no_grad():
        lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.1)
    inner = _run_velocity_for(base, xs)

    seen = []

    def run_velocity(indices):
        seen.extend(indices)
        return inner(indices)

    loss = mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers)
    assert seen == [0, 2], "the zero-coordinate sample must not cost a forward"

    # and it is out of the divisor too, not merely out of the numerator
    trained_only = mixin._slider_coordinate_loss(
        _run_velocity_for(base, [xs[0], xs[2]]), lora.set_multiplier,
        [targets[0], targets[2]], [multipliers[0], multipliers[2]],
    )
    assert torch.allclose(loss, trained_only, atol=1e-6)


def test_a_zero_multiplier_really_does_carry_no_gradient():
    """The measurement the drop above rests on. If a future adapter type made the
    multiplier-0 path differentiable, dropping those samples would start throwing
    away real signal and this fails first."""
    base, lora = _coordinate_model()
    xs, _apply_a, _ = _coordinate_batch([1.0])
    with torch.no_grad():
        lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.1)
    target = torch.randn(1, D)

    def grad_at(multiplier):
        for p in lora.parameters():
            p.grad = None
        lora.set_multiplier(multiplier)
        F.mse_loss(base(xs[0]), target).backward()
        return sum(p.grad.abs().sum().item() for p in lora.parameters() if p.grad is not None)

    assert grad_at(0.0) == 0.0
    assert grad_at(0.5) > 1e-3


def test_a_batch_of_only_zero_coordinates_is_refused():
    """The shape a mistyped axis name takes: every caption parses to no coordinate,
    every multiplier is 0, and the run would otherwise train nothing while
    reporting a plausible loss. There is no loss to return here either -- a term
    with no grad_fn fails in backward() with a much worse message."""
    mixin = _Mixin()
    base, lora = _coordinate_model()
    xs, apply_a, multipliers = _coordinate_batch([0.0, 0.0])
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    with pytest.raises(RuntimeError, match="axis name"):
        mixin._slider_coordinate_loss(
            _run_velocity_for(base, xs), lora.set_multiplier, targets, multipliers)


def test_coordinate_loss_rejects_a_mismatched_or_empty_batch():
    mixin = _Mixin()
    base, lora = _coordinate_model()
    xs, apply_a, multipliers = _coordinate_batch([1.0, -1.0])
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
    run_velocity = _run_velocity_for(base, xs)
    with pytest.raises(ValueError):
        mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, targets, multipliers[:1])
    with pytest.raises(ValueError):
        mixin._slider_coordinate_loss(run_velocity, lora.set_multiplier, [], [])


def test_coordinate_loss_leaves_the_adapter_at_the_resting_multiplier():
    mixin = _Mixin()
    base, lora = _coordinate_model()
    xs, apply_a, multipliers = _coordinate_batch([2.0, -3.0])
    targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)

    mixin._slider_coordinate_loss(
        _run_velocity_for(base, xs), lora.set_multiplier, targets, multipliers)
    assert lora.multiplier == 1.0

    def exploding(indices):
        raise RuntimeError("boom")

    lora.set_multiplier(0.25)
    with pytest.raises(RuntimeError, match="boom"):
        mixin._slider_coordinate_loss(exploding, lora.set_multiplier, targets, multipliers)
    assert lora.multiplier == 1.0


def test_a_coordinate_slider_has_a_real_base_weight_gradient():
    """The symmetric prompt-pair slider's dL/dW cancels to zero, which is why GA
    init is refused there. That argument does NOT carry over: a coordinate slider
    fits one residual per image against a real target, so the base-weight gradient
    is an ordinary reconstruction gradient and does not cancel at any coordinate
    spread -- including one symmetric about 0.

    So the GA-init skip for this regime is "not wired up and not validated", not
    "there is nothing to align to", and the message a user sees must say the one
    that is true. This test is the evidence for that distinction; if GA init is
    ever wired to the IMAGE regime, this is what says it was worth doing.
    """
    def coordinate_base_grad(coords):
        base, lora = _coordinate_model()
        base.weight.requires_grad_(True)
        xs, apply_a, multipliers = _coordinate_batch(coords)
        targets = _coordinate_targets(base, lora, xs, apply_a, multipliers)
        loss = _Mixin()._slider_coordinate_loss(
            _run_velocity_for(base, xs), lora.set_multiplier, targets, multipliers)
        base.weight.grad = None
        loss.backward()
        return base.weight.grad.norm().item()

    symmetric_prompt_pair = _base_weight_grad(symmetric=True)
    symmetric_coords = coordinate_base_grad([-2.0, -1.0, 1.0, 2.0])
    asymmetric_coords = coordinate_base_grad([1.0, 2.0])

    assert symmetric_coords > 1e-2, (
        f"coordinate ||dL/dW||={symmetric_coords:.3e} -- if this ever cancels, the "
        "prompt-pair reason for skipping GA init would apply here too"
    )
    assert asymmetric_coords > 1e-2
    assert symmetric_coords > symmetric_prompt_pair * 1e4
