"""Tests for the bounded counterexample objective (ConceptType.COUNTEREXAMPLE).

Two layers. The first is the closed-form arithmetic of
``modules/util/loss/counterexample_loss.py`` -- boundedness, switch-off, and the
exact step-0 value -- because those four properties are the entire argument for
preferring this term over the ``loss_weight = -1`` gradient ascent the config
already accepts.

The second wires a real ``ModelSetupDiffusionLossMixin`` over a synthetic batch
and asserts the *routing*: that a counterexample row's loss is replaced, that a
standard row beside it is untouched, that the concept's own ``loss_weight`` ramp
still lands on top, and -- the one that matters most -- that a missing frozen
reference raises instead of silently training the model toward the wrong image.

``python tests/test_counterexample_objective.py`` or pytest.
"""

import math
import os
import sys
import types

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The quantization import chain is heavy and irrelevant here; the slider test
# stubs it the same way for the same reason.
_stub = types.ModuleType("modules.util.quantization_util")
_stub.get_unquantized_weight = lambda m, dtype, device: m.weight.detach().to(dtype)
_stub.get_weight_shape = lambda m: m.weight.shape
sys.modules["modules.util.quantization_util"] = _stub

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import (  # noqa: E402
    ModelSetupDiffusionLossMixin,
)
from modules.module.LoRAModule import LoRAModule  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.loss.counterexample_loss import (  # noqa: E402
    BETA_BOUNDS,
    BOOTSTRAP_BETA,
    CALIBRATION_TARGET_GATE,
    DEFAULT_CALIBRATION_STEPS,
    MIN_CALIBRATION_STEPS,
    TELEMETRY,
    CounterexampleSchedule,
    CounterexampleTelemetry,
    counterexample_losses,
    counterexample_stats,
    counterexample_weight,
    noise_band_weight,
    ramp_steps,
)

BETA = 4.0


# ---------------------------------------------------------------------------
# the objective itself
# ---------------------------------------------------------------------------


def test_step_zero_is_exactly_half_scale():
    """A LoRA starts at zero, so v_theta == v_ref and delta == 0 exactly. The
    published-value check: L = (2/beta) * log 2, and dL/d(delta) = 1 -- the same
    magnitude a positive row carries, so a counterexample cannot spike step 0."""
    d = torch.tensor([0.3], requires_grad=True)
    d_ref = torch.tensor([0.3])

    loss = counterexample_losses(d, d_ref, BETA)
    assert math.isclose(loss.item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-6)

    loss.sum().backward()
    # dL/dd = -2 * sigmoid(beta * delta) = -1 at delta == 0.
    assert math.isclose(d.grad.item(), -1.0, rel_tol=1e-6)


def test_gradient_pushes_the_distance_up():
    """The sign check. The loss falls as the adapter gets *worse* at reproducing
    the counterexample, which is what "train away from this image" has to mean."""
    d = torch.tensor([0.1, 0.5, 2.0], requires_grad=True)
    d_ref = torch.tensor([0.7, 0.5, 0.1])

    counterexample_losses(d, d_ref, BETA).sum().backward()
    assert torch.all(d.grad < 0)


def test_slope_is_bounded_by_two():
    """Boundedness. However far the adapter has drifted, one counterexample row's
    gradient magnitude never exceeds 2 -- the property `loss_weight = -1` lacks,
    and the reason that configuration collapses."""
    d = torch.linspace(-50.0, 50.0, 201, requires_grad=True)
    d_ref = torch.zeros(201)

    counterexample_losses(d, d_ref, BETA).sum().backward()
    assert torch.all(d.grad.abs() <= 2.0 + 1e-6)


def test_it_switches_itself_off():
    """Once the adapter fits the wrong image *worse* than the base does, the term
    stops pushing. This is the whole difference from gradient ascent."""
    d = torch.tensor([100.0], requires_grad=True)  # delta = -100, deeply saturated
    loss = counterexample_losses(d, torch.tensor([0.0]), BETA)

    assert loss.item() < 1e-6
    loss.sum().backward()
    assert d.grad.abs().item() < 1e-6


def test_loss_is_monotonic_in_delta():
    delta = torch.linspace(-5.0, 5.0, 64)
    losses = counterexample_losses(torch.zeros(64), delta, BETA)
    assert torch.all(losses[1:] - losses[:-1] > 0)


def test_beta_must_be_positive():
    for bad in (0.0, -1.0):
        try:
            counterexample_losses(torch.zeros(1), torch.zeros(1), bad)
        except ValueError:
            continue
        raise AssertionError(f"beta={bad} should have been rejected")


# ---------------------------------------------------------------------------
# telemetry -- the instrument that says whether any of the above ran
# ---------------------------------------------------------------------------


def test_stats_report_the_live_fraction():
    """`gate_mean` is the readout risk 2 exists for: it is the mean per-row
    multiplier on the repulsion gradient, so ~0 means the term was inert."""
    delta = torch.tensor([0.0, -100.0, 100.0, 0.0])
    losses = counterexample_losses(torch.zeros(4), delta, BETA)
    stats = counterexample_stats(delta, losses, BETA)

    assert stats.rows == 4
    # Strictly < 0, so only the -100. The two zeros are the state a cold LoRA is
    # in on step 0 -- counting them would report "all saturated" before the run
    # has done anything. See test_a_cold_start_reports_nothing_saturated.
    assert math.isclose(stats.saturated_fraction, 0.25)
    # gate = sigmoid(beta*delta) -> [0.5, ~0, ~1, 0.5]
    assert math.isclose(stats.gate_mean, 0.5, abs_tol=1e-4)
    assert math.isclose(stats.delta_mean, 0.0, abs_tol=1e-6)


def test_a_cold_start_reports_nothing_saturated():
    """A LoRA starts at zero, so step 0 has `delta == 0` for every row exactly.

    That is the least-informative moment of the run, and `saturated_fraction`
    must not describe it as the most-finished one. Caught on the first real
    training run: `gate_first` correctly read 0.50 while `saturated_first` read
    1.00, two readouts of the same step disagreeing about whether anything had
    happened.
    """
    delta = torch.zeros(4)
    losses = counterexample_losses(torch.zeros(4), delta, BETA)
    stats = counterexample_stats(delta, losses, BETA)

    assert stats.saturated_fraction == 0.0
    assert math.isclose(stats.gate_mean, 0.5)


def test_beta_cannot_ramp_strength_but_the_ramp_can():
    """The correction that motivated `counterexample_ramp` existing at all.

    "Ramp beta up so the term starts gentle" is the intuitive move and it does not
    work: `dL/dDelta = 2*sigmoid(beta*delta)`, so at `delta == 0` -- exactly where
    every cold LoRA starts -- the slope is 1.0 for EVERY beta. Ramping beta is in
    fact an anti-ramp, moving the term from never-switches-off to sharply-bounded.
    """
    for beta in (1.0, 100.0, 1_000.0, 31_500.0, 1_000_000.0):
        d = torch.tensor([1.0], requires_grad=True)
        counterexample_losses(d, torch.tensor([1.0]), beta).sum().backward()
        assert math.isclose(-float(d.grad), 1.0, rel_tol=1e-6)

    # The weight, by contrast, is a strength knob and starts at exactly zero.
    assert counterexample_weight(0, 100, ramp=1.0) == 0.0
    assert counterexample_weight(100, 100, ramp=1.0) == 1.0


def test_the_ramp_arrives_at_full_strength_at_the_end():
    """`ramp = 1.0` means "strongest during the LR anneal" -- the whole request."""
    total = 200
    weights = [counterexample_weight(s, total, ramp=1.0) for s in range(total + 1)]

    assert weights[0] == 0.0
    assert math.isclose(weights[-1], 1.0, abs_tol=1e-9)
    assert weights == sorted(weights)  # monotonic, never overshoots
    assert math.isclose(weights[total // 2], 0.5, abs_tol=1e-6)  # raised cosine
    # Leaves 0 and arrives at 1 with (near) zero slope -- "calm", not linear.
    assert weights[1] < 0.5 / total
    assert (1.0 - weights[-2]) < 0.5 / total


def test_the_ramp_follows_onetrainers_warmup_convention():
    """Fractions in (0, 1], literal steps above 1, 0 disables -- the same rule
    `learning_rate_warmup_steps` already uses, so there is one to learn."""
    assert ramp_steps(0.0, 500) == 0
    assert ramp_steps(0.25, 500) == 125
    assert ramp_steps(1.0, 500) == 500
    assert ramp_steps(50, 500) == 50
    assert counterexample_weight(3, 500, ramp=0.0) == 1.0  # disabled: full at once


def test_an_explicit_beta_is_never_overridden():
    schedule = CounterexampleSchedule()
    schedule.observe(torch.tensor([1e-5, 2e-5]))
    assert schedule.beta(configured=777.0, step=0, total_steps=100) == 777.0


def test_auto_beta_solves_for_this_runs_delta_and_then_freezes():
    """`counterexample_beta = 0` reads the scale off the run instead of a constant.

    Measured motivation: beta=1000 on a real SD 1.5 LoRA gave `beta*|delta| ~
    0.015` and wanted ~31,500, and |delta| grows 21x within a single short run --
    so no published constant can be right for every model and stage.

    Frozen after the window, not tracked: a beta that kept chasing |delta| would
    pin the gate forever and destroy the switch-off the bound exists for.
    """
    schedule = CounterexampleSchedule()
    # Nothing observed yet (step 0 is delta == 0 exactly) -> bootstrap, not a crash.
    assert schedule.beta(configured=0.0, step=99, total_steps=100) == BOOTSTRAP_BETA

    for _ in range(MIN_CALIBRATION_STEPS):
        schedule.observe(torch.tensor([1e-5, -1e-5]))  # |delta| = 1e-5
    beta = schedule.beta(configured=0.0, step=99, total_steps=100)

    # sigmoid(beta * 1e-5) should sit on the calibration target.
    assert math.isclose(1 / (1 + math.exp(-beta * 1e-5)), CALIBRATION_TARGET_GATE, rel_tol=1e-6)

    # Window is over, so later (larger) deltas must not move it.
    schedule.observe(torch.tensor([1.0]))
    assert schedule.beta(configured=0.0, step=99, total_steps=100) == beta


def test_beta_is_stable_through_the_calibration_window():
    """Regression, from a real run: the provisional solve swung 30x mid-window.

    The solve is `logit / mean|delta|`, and a cold LoRA's first deltas are ~0, so
    using it before the window closed produced beta = 15,000,000 at step 6 decaying
    to 400,000 by step 20 -- the objective's own scale thrashing while training.
    It went unnoticed because the ramp had the weight near 0 throughout; with
    `counterexample_ramp = 0` it would have run at full strength.

    So beta must hold ONE value for the whole window and change exactly once.
    """
    schedule = CounterexampleSchedule()
    seen = []
    for i in range(MIN_CALIBRATION_STEPS):
        # Deltas growing from ~0, which is what made the provisional solve explode.
        schedule.observe(torch.tensor([1e-12 * (i + 1)]))
        seen.append(schedule.beta(configured=0.0, step=i, total_steps=1000))

    assert seen == [BOOTSTRAP_BETA] * MIN_CALIBRATION_STEPS
    assert len(set(seen)) == 1


def test_the_calibration_window_does_not_follow_a_long_ramp():
    """Tying beta's window to the ramp degenerates at `counterexample_ramp = 1.0`:
    it would close on the last step, calibrating beta exactly as the run ends. So
    the window is its own fraction of the run -- which at the 0.25 ramp default
    happens to coincide anyway -- and floored for short runs."""
    schedule = CounterexampleSchedule()

    assert schedule.calibration_steps(total_steps=1000) == 250   # 25% of the run
    assert schedule.calibration_steps(total_steps=4) == MIN_CALIBRATION_STEPS
    assert schedule.calibration_steps(total_steps=0) == DEFAULT_CALIBRATION_STEPS


def test_the_window_closes_on_training_steps_not_observed_ones():
    """Regression, from a real run where auto-beta silently never engaged.

    The window was sized in training steps but counted in *observed* ones. Batches
    that hold no counterexample row observe nothing, so with batch 2 and two
    concepts ~25% of steps never counted: `observed` trailed `step` forever, the
    window never closed, and the run trained its whole life on the bootstrap beta
    -- indistinguishable, from outside, from an explicitly configured 1000.
    """
    schedule = CounterexampleSchedule()
    total = 48
    window = schedule.calibration_steps(total)

    # Three quarters of steps carry a row, as in the run that exposed this.
    for step in range(total):
        if step % 4:
            schedule.observe(torch.tensor([1e-5]))
        beta = schedule.beta(configured=0.0, step=step, total_steps=total)

    assert schedule._observed_steps < total     # observations trail the steps...
    assert beta != BOOTSTRAP_BETA               # ...and beta calibrated regardless
    assert math.isclose(1 / (1 + math.exp(-beta * 1e-5)), CALIBRATION_TARGET_GATE, rel_tol=1e-6)

    # It still refuses to solve from nothing: a run with almost no rows holds.
    starved = CounterexampleSchedule()
    starved.observe(torch.tensor([1e-5]))
    assert starved.beta(configured=0.0, step=window + 5, total_steps=total) == BOOTSTRAP_BETA


def test_auto_beta_is_clamped_off_a_denormal_delta():
    """The first steps after zero can produce a |delta| whose reciprocal is not a
    hyperparameter."""
    schedule = CounterexampleSchedule()
    schedule.observe(torch.tensor([1e-30]))
    assert schedule.beta(configured=0.0, step=99, total_steps=10) <= BETA_BOUNDS[1]


def test_the_ramp_scales_the_counterexample_row_and_only_that_row():
    """The point of a separate schedule: counterexample *timing* independent of
    the positives'. At the start of the ramp the wrong image contributes nothing,
    while the positive beside it trains at full strength as usual."""
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.COUNTEREXAMPLE]
    batch, data = _batch_and_data(types_)
    data["counterexample_step"] = 0  # start of the ramp -> weight == 0
    data["counterexample_total_steps"] = 100
    config = _config(counterexample_ramp=1.0)

    losses = _Mixin()._flow_matching_losses(batch, data, config, torch.device("cpu"))

    assert losses[1].item() == 0.0                        # ramped fully out
    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)  # positive untouched

    # ...and at the end of the ramp the same row carries the full step-0 value.
    TELEMETRY.reset()
    batch, data = _batch_and_data(types_)
    data["counterexample_step"] = 100
    data["counterexample_total_steps"] = 100
    losses = _Mixin()._flow_matching_losses(batch, data, config, torch.device("cpu"))
    assert math.isclose(losses[1].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)


def test_telemetry_accumulates_across_micro_steps_and_drains():
    """Gradient accumulation calls the loss several times per optimizer step, so
    the readout must sum the window and then reset -- otherwise the first logged
    step reports one micro-batch and every later one reports the whole run."""
    telemetry = CounterexampleTelemetry()
    delta = torch.tensor([1.0, -1.0])
    losses = counterexample_losses(torch.zeros(2), delta, BETA)

    telemetry.record(counterexample_stats(delta, losses, BETA))
    telemetry.record(counterexample_stats(delta, losses, BETA))

    taken = telemetry.take().stats
    assert taken.rows == 4
    assert math.isclose(taken.saturated_fraction, 0.5)

    assert telemetry.take().stats.rows == 0


# ---------------------------------------------------------------------------
# the noise band
# ---------------------------------------------------------------------------


def _u(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """The coordinate under test, written out independently of the implementation."""
    snr = alphas_cumprod / (1.0 - alphas_cumprod)
    return 1.0 / (1.0 + snr.sqrt())


def test_the_noise_coordinate_is_exactly_sigma_for_a_rectified_flow():
    """The identity the whole design rests on. A flow model's forward process is
    ``x_t = (1 - sigma) x_0 + sigma eps``, so ``SNR = ((1-sigma)/sigma)^2`` and
    ``u = 1/(1+sqrt(SNR))`` collapses to sigma itself. That is what lets the
    flow-matching branch read sigma straight off its own schedule while the
    diffusion branch computes u from alphas_cumprod, and have the two mean the
    same physical thing."""
    sigma = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95])
    snr = ((1.0 - sigma) / sigma) ** 2
    u = 1.0 / (1.0 + snr.sqrt())
    assert torch.allclose(u, sigma, atol=1e-6)


def test_the_noise_coordinate_is_not_the_timestep_fraction():
    """Why this got its own phase instead of reusing ``min_noising_strength``.

    Timestep-index fraction is not comparable across model families. On SD 1.5's
    scaled-linear schedule the tenth of the schedule nearest clean is already a
    QUARTER noise by amplitude, while on a rectified flow t/N = 0.1 is sigma =
    0.1 exactly. A band authored on one model and reused on the other would
    silently cover a different noise range -- and nothing would report it.
    """
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
    u = _u(torch.cumprod(1.0 - betas, dim=0))

    assert 0.25 < u[100].item() < 0.27        # t/N = 0.1 -> u ~ 0.256, not 0.10
    assert 0.75 < u[700].item() < 0.78        # t/N = 0.7 -> u ~ 0.77
    # ...and monotone, so a band in u is still a contiguous range of timesteps.
    assert torch.all(u[1:] >= u[:-1])


def test_no_band_is_exactly_a_no_op():
    """The default has to be provably inert, not merely gentle. A taper nobody
    asked for would change every existing run's dose silently."""
    u = torch.linspace(0.0, 1.0, 51)
    assert torch.equal(noise_band_weight(u, 0.0, 1.0), torch.ones_like(u))


def test_a_band_keeps_a_full_strength_plateau():
    """Edges are a quarter of the width each and combined with `minimum`, so the
    middle half of any band runs at full strength. A product of the two edges
    would instead dip in the centre -- the one place the term should be
    strongest."""
    u = torch.linspace(0.0, 1.0, 1001)
    for low, high in [(0.3, 0.7), (0.0, 0.5), (0.45, 0.55), (0.6, 1.0)]:
        weight = noise_band_weight(u, low, high)
        assert weight.max().item() == 1.0, (low, high)
        mid = noise_band_weight(torch.tensor([(low + high) / 2.0]), low, high)
        assert math.isclose(mid.item(), 1.0, abs_tol=1e-6), (low, high)


def test_a_band_is_zero_outside_and_smooth_at_its_edges():
    u = torch.linspace(0.0, 1.0, 1001)
    weight = noise_band_weight(u, 0.3, 0.7)

    assert weight[u < 0.3].max().item() == 0.0
    assert weight[u > 0.7].max().item() == 0.0
    # A raised cosine, so no step: the largest jump between adjacent samples is
    # far below the 1.0 a hard cutoff would produce.
    assert weight.diff().abs().max().item() < 0.02


def test_an_open_lower_edge_does_not_taper_the_clean_end():
    """`low = 0` means "no lower bound", so u = 0 must pass at full strength. An
    edge applied unconditionally would zero out the cleanest latents -- the
    opposite of what an open bound says."""
    weight = noise_band_weight(torch.tensor([0.0, 0.1]), 0.0, 0.77)
    assert torch.allclose(weight, torch.ones(2))


def test_an_inverted_band_is_rejected():
    u = torch.linspace(0.0, 1.0, 11)
    for low, high in [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)]:
        try:
            noise_band_weight(u, low, high)
        except ValueError:
            continue
        raise AssertionError(f"({low}, {high}) should have been rejected")


def test_beta_calibrates_on_the_rows_the_band_lets_through():
    """|delta| varies strongly with noise level, so a band that mutes part of the
    schedule also moves the delta scale the run optimizes. Calibrating beta on
    rows the band removed would solve for a scale that never trains."""
    # Realistic magnitudes: delta is a difference of element-mean distances, and
    # a large one clamps against BETA_BOUNDS, hiding the effect under test.
    delta = torch.tensor([0.001, 0.1])
    in_band_only = torch.tensor([1.0, 0.0])

    unbanded, banded = CounterexampleSchedule(), CounterexampleSchedule()
    for _ in range(MIN_CALIBRATION_STEPS):
        unbanded.observe(delta)
        banded.observe(delta, in_band_only)

    # mean|delta| is 0.0505 unbanded and 0.001 banded, and beta is
    # logit/mean|delta| -- a factor of 50 in the objective's own scale.
    assert banded.beta(0.0, 999, MIN_CALIBRATION_STEPS) > 40 * unbanded.beta(
        0.0, 999, MIN_CALIBRATION_STEPS
    )


def test_a_step_muted_entirely_by_the_band_is_not_an_observation():
    """Contributing a zero would drag the calibration mean toward zero and blow
    the solved beta up; contributing nothing is the honest reading -- that step
    trained on no counterexample at all."""
    schedule = CounterexampleSchedule()
    for _ in range(MIN_CALIBRATION_STEPS):
        schedule.observe(torch.tensor([2.0]), torch.tensor([0.0]))
    # Enough *calls* to close the window, but no observations behind them.
    assert schedule.beta(0.0, 999, MIN_CALIBRATION_STEPS) == BOOTSTRAP_BETA

    for _ in range(MIN_CALIBRATION_STEPS):
        schedule.observe(torch.tensor([2.0]), torch.tensor([1.0]))
    assert schedule.beta(0.0, 999, MIN_CALIBRATION_STEPS) != BOOTSTRAP_BETA


def test_an_unbanded_run_reports_a_full_dose():
    """`band_pass` is the dose. With no band every row passes, which is 1.0 per
    row -- a default of 0 would report an ordinary run as having delivered
    nothing."""
    delta = torch.tensor([0.5, 0.5])
    stats = counterexample_stats(delta, counterexample_losses(torch.zeros(2), delta, BETA), BETA)
    assert stats.band_mean == 1.0


# ---------------------------------------------------------------------------
# routing through the real loss mixin
# ---------------------------------------------------------------------------


class _Mixin(ModelSetupDiffusionLossMixin):
    pass


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.batch_size = 3
    config.gradient_accumulation_steps = 1
    config.masked_training = False
    config.counterexample_beta = BETA
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _batch_and_data(types_, *, with_reference=True, loss_weight=1.0):
    """Three 1x2x2x2 rows. `predicted` sits at 0, `target` at 1, and the frozen
    reference at 2 -- so the trained model is *closer* to every wrong image than
    the reference is (delta > 0) and the term is unambiguously live."""
    shape = (len(types_), 2, 2, 2)
    batch = {
        "concept_type": [t.value for t in types_],
        "loss_weight": torch.full((len(types_),), loss_weight),
        "latent_mask": torch.ones(shape),
    }
    data = {
        "loss_type": "target",
        "predicted": torch.zeros(shape, requires_grad=True),
        "target": torch.ones(shape),
    }
    if with_reference:
        data["prior_target"] = torch.full(shape, 2.0)
    return batch, data


def test_only_the_counterexample_row_is_replaced():
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.COUNTEREXAMPLE, ConceptType.PRIOR_PREDICTION]
    batch, data = _batch_and_data(types_)
    losses = _Mixin()._flow_matching_losses(batch, data, _config(), torch.device("cpu"))

    # d = mean((0-1)^2) = 1, d_ref = mean((2-1)^2) = 1, so delta = 0 and the
    # counterexample row lands on the step-0 value.
    expected_repulsion = (2.0 / BETA) * math.log(2.0)
    assert math.isclose(losses[1].item(), expected_repulsion, rel_tol=1e-5)

    # Its neighbours keep the ordinary reconstruction loss.
    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(losses[2].item(), 1.0, rel_tol=1e-6)

    stats = TELEMETRY.take().stats
    assert stats.rows == 1


def test_the_epsilon_prediction_path_routes_it_too():
    """There are two insertion points -- `_flow_matching_losses` and
    `_diffusion_losses` -- and a term wired into only one of them would be silently
    absent for every non-flow-matching model."""
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.COUNTEREXAMPLE]
    batch, data = _batch_and_data(types_)
    losses = _Mixin()._diffusion_losses(batch, data, _config(batch_size=2), torch.device("cpu"))

    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(losses[1].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    assert TELEMETRY.take().stats.rows == 1


def test_the_band_mutes_a_counterexample_row_outside_it():
    """Routing, on the flow branch: a row whose noise level falls outside the
    band contributes nothing, while the positive beside it is untouched -- the
    band is a counterexample knob, not a global loss weight."""
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.COUNTEREXAMPLE]
    batch, data = _batch_and_data(types_)
    # sigma = (t+1)/1000, so index 949 is u = 0.95: nearly pure noise.
    data["timestep"] = torch.tensor([949, 949])
    config = _config(counterexample_band_low=0.0, counterexample_band_high=0.5)

    losses = _Mixin()._flow_matching_losses(
        batch, data, config, torch.device("cpu"), sigmas=torch.zeros(1000)
    )

    assert losses[1].item() == 0.0
    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)
    # ...and the readout says the dose was zero rather than reporting a healthy run.
    stats = TELEMETRY.take().stats
    assert stats.band_mean == 0.0
    assert math.isclose(stats.noise_level_mean, 0.95, abs_tol=1e-6)


def test_the_band_passes_a_counterexample_row_inside_it():
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["timestep"] = torch.tensor([499])          # u = 0.5, mid-band
    config = _config(
        batch_size=1, counterexample_band_low=0.3, counterexample_band_high=0.7
    )

    losses = _Mixin()._flow_matching_losses(
        batch, data, config, torch.device("cpu"), sigmas=torch.zeros(1000)
    )
    assert math.isclose(losses[0].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    assert TELEMETRY.take().stats.band_mean == 1.0


def test_both_loss_branches_agree_on_the_noise_coordinate():
    """The cross-family claim, made checkable: a variance-preserving latent and a
    rectified-flow latent at *the same physical noise level* must get the same
    band weight, even though one branch derives u from alphas_cumprod and the
    other reads sigma straight off its schedule.

    u = 0.8 corresponds to alphas_cumprod = 1/17 on a VP schedule and to
    sigma = 0.8 on a flow -- numbers that look nothing alike, which is the point.
    """
    band = {"counterexample_band_low": 0.7, "counterexample_band_high": 0.9}
    expected = (2.0 / BETA) * math.log(2.0)

    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["timestep"] = torch.tensor([799])          # sigma = 0.8
    flow = _Mixin()._flow_matching_losses(
        batch, data, _config(batch_size=1, **band), torch.device("cpu"),
        sigmas=torch.zeros(1000),
    )
    flow_noise = TELEMETRY.take().stats.noise_level_mean

    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["timestep"] = torch.tensor([0])
    diffusion = _Mixin()._diffusion_losses(
        batch, data, _config(batch_size=1, **band), torch.device("cpu"),
        alphas_cumprod_fun=lambda t, dim: torch.full(t.shape, 1.0 / 17.0),
    )
    diffusion_noise = TELEMETRY.take().stats.noise_level_mean

    assert math.isclose(flow_noise, 0.8, abs_tol=1e-6)
    assert math.isclose(diffusion_noise, 0.8, abs_tol=1e-6)
    assert math.isclose(flow.item(), expected, rel_tol=1e-5)
    assert math.isclose(diffusion.item(), expected, rel_tol=1e-5)


def test_a_model_with_no_schedule_simply_gets_no_band():
    """`loss_weight_fn = CONSTANT` with no betas and no alphas_cumprod_fun leaves
    __snr nothing to read. That path has nothing to do with counterexamples and
    must not start raising because one is in the batch."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["timestep"] = torch.tensor([500])
    config = _config(batch_size=1, counterexample_band_low=0.3, counterexample_band_high=0.4)

    losses = _Mixin()._diffusion_losses(batch, data, config, torch.device("cpu"))

    assert math.isclose(losses[0].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    assert TELEMETRY.take().stats.band_mean == 1.0


def test_the_concepts_own_loss_weight_still_ramps_it():
    """`loss_weight` is applied after the substitution, so a counterexample
    concept can still be dialled down without touching beta."""
    TELEMETRY.reset()
    types_ = [ConceptType.COUNTEREXAMPLE]
    batch, data = _batch_and_data(types_, loss_weight=0.25)
    losses = _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))

    assert math.isclose(losses[0].item(), 0.25 * (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    TELEMETRY.take()


def test_a_missing_reference_raises_rather_than_training_toward_the_image():
    """The failure a green run would never reveal: without the frozen forward the
    row would fall through as an ordinary positive and teach the model to
    reproduce the near-miss it was supposed to be pushed away from."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE], with_reference=False)
    try:
        _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))
    except RuntimeError as exc:
        assert "COUNTEREXAMPLE" in str(exc)
        return
    raise AssertionError("a counterexample row without a reference must raise")


def test_the_ga_estimation_pass_can_opt_out_explicitly():
    """kron-GA has no reference forward by design; it says so, and gets the
    ordinary loss (which it has already zeroed by detaching the target)."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE], with_reference=False)
    data["skip_counterexample_repulsion"] = True
    losses = _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))

    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)
    assert TELEMETRY.take().stats.rows == 0


def test_an_identical_reference_gives_delta_exactly_zero():
    """The shared step-seeded generator means both forwards see the same (x_t, t),
    so a zero adapter makes the two predictions identical -- and delta must then
    be exactly 0, not merely small. Anything else is a metric mismatch between
    the two halves of the subtraction."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["prior_target"] = data["predicted"].detach().clone()
    _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))

    stats = TELEMETRY.take().stats
    assert stats.delta_mean == 0.0
    assert math.isclose(stats.gate_mean, 0.5)


def test_masked_training_uses_the_same_metric_on_both_sides():
    """Trap 6: `delta` is meaningless if `d` and `d_ref` come from different
    metrics. Under masked training the mask must apply to both, which a shared
    helper guarantees and two call sites would not.

    Deliberately asymmetric -- d = 1, d_ref = 4 -- because a test where the two
    distances already agree passes whatever the mask does to either side.
    """
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["prior_target"] = torch.full((1, 2, 2, 2), 3.0)  # d_ref = mean((3-1)^2) = 4

    unmasked = _config(batch_size=1)
    _Mixin()._flow_matching_losses(batch, data, unmasked, torch.device("cpu"))
    assert math.isclose(TELEMETRY.take().stats.delta_mean, 3.0, rel_tol=1e-6)

    # Everything outside the mask, weighted 0.5: both distances halve, so delta
    # halves. A mask applied to only one side would have given 3.5 or 2.5.
    batch["latent_mask"] = torch.zeros((1, 2, 2, 2))
    masked = _config(batch_size=1, masked_training=True, unmasked_weight=0.5)
    _Mixin()._flow_matching_losses(batch, data, masked, torch.device("cpu"))
    assert math.isclose(TELEMETRY.take().stats.delta_mean, 1.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# end to end: a real LoRA, trained by this objective, on CPU
# ---------------------------------------------------------------------------


D = 16


def _toy_adapter(rank=8, alpha=8.0):
    """A frozen base Linear with a real LoRA hooked on, plus the base's own output
    captured *before* hooking -- which is exactly what `prior_model()` produces at
    training time by detaching every adapter."""
    torch.manual_seed(0)
    base = nn.Linear(D, D, bias=False)
    base.weight.requires_grad_(False)

    cond = torch.randn(1, D)
    bad_image = torch.randn(1, D)
    reference = base(cond).detach()

    lora = LoRAModule("t.l", base, rank, alpha)
    lora.hook_to_module()
    return base, lora, cond, bad_image, reference


def _train(loss_fn, steps=300, lr=1e-2):
    base, lora, cond, bad_image, reference = _toy_adapter()
    d_ref = ((reference - bad_image) ** 2).mean()
    params = [p for p in lora.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        d = ((base(cond) - bad_image) ** 2).mean()
        loss_fn(d, d_ref).backward()
        optimizer.step()

    with torch.no_grad():
        final = ((base(cond) - bad_image) ** 2).mean().item()
    return d_ref.item(), final


def test_a_real_lora_learns_to_move_away_from_the_counterexample():
    """The decisive check: wire the objective to a real LoRAModule and optimize.
    If the adapter ends up further from the bad image than the frozen base was,
    the objective + hook + autograd path is sound end to end."""
    d_ref, final = _train(lambda d, r: counterexample_losses(d, r, BETA))
    assert final > d_ref


def test_the_bounded_term_stops_where_gradient_ascent_does_not():
    """The straw man, run on purpose. `loss_weight = -1` is naive ascent on an
    unbounded loss and is already expressible in the config today; it does not
    converge, it just keeps going. Same adapter, same steps, same learning rate --
    the only difference is the bound."""
    d_ref, bounded = _train(lambda d, r: counterexample_losses(d, r, BETA))
    _, ascent = _train(lambda d, r: -d)

    assert ascent > 10 * bounded, (
        f"expected unbounded ascent to run away (got {ascent:.3f} vs bounded {bounded:.3f})"
    )
    # And the bounded one settled somewhere finite and only modestly past the
    # reference -- "worse than base on the bad image", not "destroyed".
    assert d_ref < bounded < 20 * d_ref


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
