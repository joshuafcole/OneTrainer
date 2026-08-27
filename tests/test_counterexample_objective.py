"""Tests for the bounded counterexample objective (ConceptType.COUNTEREXAMPLE).

Two layers. The first is the closed-form arithmetic of
``modules/util/loss/counterexample_loss.py`` -- boundedness, switch-off, and the
exact step-0 value -- because those properties are the entire argument for
preferring this term over the ``loss_weight = -1`` gradient ascent the config
already accepts.

The second wires a real ``ModelSetupDiffusionLossMixin`` over a synthetic batch
and asserts the *routing*: that a counterexample row's loss is replaced, that a
standard row beside it is untouched, that both the flow-matching and the
epsilon-prediction entry points route it, that the concept's own ``loss_weight``
still lands on top, and -- the one that matters most -- that a missing frozen
reference raises instead of silently training the model toward the wrong image.
"""

import contextlib
import io
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import (  # noqa: E402
    ModelSetupDiffusionLossMixin,
)
from modules.modelSetup.mixin.ModelSetupNoiseMixin import (  # noqa: E402
    ModelSetupNoiseMixin,
)
from modules.module.LoRAModule import LoRAModule  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.loss.counterexample_loss import (  # noqa: E402
    BETA_BOUNDS,
    BOOTSTRAP_BETA,
    CALIBRATION_TARGET_GATE,
    DEFAULT_CALIBRATION_STEPS,
    GRADED_BAND,
    MIN_CALIBRATION_STEPS,
    SCHEDULE,
    TELEMETRY,
    CounterexampleSchedule,
    CounterexampleTelemetry,
    band_dose,
    counterexample_losses,
    counterexample_stats,
    counterexample_weight,
    noise_band_weight,
    noise_level_from_snr,
    ramp_steps,
)

import torch  # noqa: E402
from torch import nn  # noqa: E402

import pytest  # noqa: E402

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
    """The runtime half of a two-layer validation: the UI's check_range keeps a
    typed-in beta positive, and this keeps a config assembled any other way --
    a hand-edited json, the CLI, a test -- from dividing by zero silently."""
    for bad in (0.0, -1.0):
        try:
            counterexample_losses(torch.zeros(1), torch.zeros(1), bad)
        except ValueError:
            continue
        raise AssertionError(f"beta={bad} should have been rejected")


def test_the_shipped_defaults_ask_for_auto_calibration():
    """The default beta is 0 -- "solve it from this run's own delta" -- because the
    borrowed 1000 did not survive a real run: it gave `beta*|delta| ~ 0.015` where
    ~31,500 was wanted, i.e. a gate pinned at 0.5 and a term that never switched
    off. Zero is a *legal* value of this field for exactly that reason, so this
    also pins that the objective itself still refuses it (the schedule, not the
    config, is what turns 0 into a usable beta).

    The config tuple's shape (name, default, type, nullable) makes the types easy
    to get wrong, so those are asserted too.
    """
    config = TrainConfig.default_values()
    assert isinstance(config.counterexample_beta, float)
    assert config.counterexample_beta == 0.0
    assert isinstance(config.counterexample_ramp, float)
    assert config.counterexample_ramp == 0.25
    # 0 / 1 is "no band", and the band function proves it is exactly a no-op.
    assert config.counterexample_band_low == 0.0
    assert config.counterexample_band_high == 1.0

    # 0 never reaches the objective: SCHEDULE.beta() substitutes a real one.
    assert SCHEDULE.beta(config.counterexample_beta, 0, 100) > 0.0
    try:
        counterexample_losses(torch.zeros(1), torch.zeros(1), config.counterexample_beta)
    except ValueError:
        return
    raise AssertionError("the objective must still refuse a beta of 0 directly")


# ---------------------------------------------------------------------------
# telemetry -- the instrument that says whether any of the above ran
# ---------------------------------------------------------------------------


def test_stats_report_the_live_fraction():
    """`gate_mean` is the readout that says whether the term is live at all: it
    is the mean per-row multiplier on the repulsion gradient."""
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


def test_the_gate_is_measured_in_beta_scaled_delta():
    """`gate_mean` is `sigmoid(beta * delta)`, and beta is the whole point of it:
    the documented way to choose beta is to run briefly and read this number, so a
    gate computed on raw delta would misdirect every tuning decision while looking
    entirely healthy.

    Pinned at a delta in the sensitive region, because the obvious probe values
    do not discriminate -- at delta 0 the gate is 0.5 for every beta, and far from
    0 it is saturated for every beta.
    """
    delta = torch.tensor([0.25])
    losses = counterexample_losses(torch.zeros(1), delta, BETA)

    scaled = counterexample_stats(delta, losses, BETA).gate_mean
    assert math.isclose(scaled, 1.0 / (1.0 + math.exp(-BETA * 0.25)), rel_tol=1e-6)

    # And it genuinely moves with beta, rather than tracking delta alone.
    unscaled = 1.0 / (1.0 + math.exp(-0.25))
    assert abs(scaled - unscaled) > 0.1
    assert counterexample_stats(delta, losses, 1.0).gate_mean < scaled


def test_a_cold_start_reports_nothing_saturated():
    """A LoRA starts at zero, so step 0 has `delta == 0` for every row exactly.

    That is the least-informative moment of the run, and `saturated_fraction`
    must not describe it as the most-finished one. Caught on a real training run:
    `gate_mean` correctly read 0.50 while `saturated_fraction` read 1.00, two
    readouts of the same step disagreeing about whether anything had happened.
    """
    delta = torch.zeros(4)
    losses = counterexample_losses(torch.zeros(4), delta, BETA)
    stats = counterexample_stats(delta, losses, BETA)

    assert stats.saturated_fraction == 0.0
    assert math.isclose(stats.gate_mean, 0.5)


# ---------------------------------------------------------------------------
# the ramp, and beta calibrated from the run instead of guessed
# ---------------------------------------------------------------------------


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


def test_a_second_run_in_one_process_does_not_inherit_the_first_beta():
    """`SCHEDULE` is a process-global singleton and a process can train twice --
    the GUI runs the trainer on a thread and stays alive across Start presses,
    which is exactly the shape a back-to-back A/B bake-off has.

    A carried-over frozen beta is the worst kind of wrong: it was calibrated
    against a different model, resolution or band, and it is *indistinguishable
    from a configured one* in the telemetry, so the second run reads healthy.
    `GenericTrainer.start()` calls `reset()`; this pins what that has to restore.
    """
    schedule = CounterexampleSchedule()
    for _ in range(MIN_CALIBRATION_STEPS):
        schedule.observe(torch.tensor([1e-5, -1e-5]))
    first = schedule.beta(configured=0.0, step=99, total_steps=100)

    # Run 2, an order of magnitude coarser -- without the reset it would be
    # handed run 1's beta and never solve at all.
    schedule.reset()
    for _ in range(MIN_CALIBRATION_STEPS):
        schedule.observe(torch.tensor([1e-4, -1e-4]))
    second = schedule.beta(configured=0.0, step=99, total_steps=100)

    assert math.isclose(second, first / 10.0, rel_tol=1e-6)


def test_the_trainer_actually_resets_the_schedule_between_runs():
    """`reset()` shipped with no callers at all, and that -- not the method -- was
    the defect: a method that restores the invariant but is never reached restores
    nothing, and a carried-over beta says nothing about itself. The test above
    would pass with the call site missing, so this one asserts the *wiring*.

    Read as source rather than imported, because importing GenericTrainer pulls
    in the whole training stack for one grep. Scoped to `start`'s own body so a
    reset called from anywhere else does not satisfy it: `start` is what every
    entry point (scripts/train.py, MultiTrainer, the GUI's Start button on its
    long-lived thread) actually calls.
    """
    source = pathlib.Path(__file__).resolve().parents[1] / "modules/trainer/GenericTrainer.py"
    body = source.read_text().split("\n    def start(self):\n", 1)[1].split("\n    def ", 1)[0]

    assert "counterexample_schedule.reset()" in body
    assert "counterexample_telemetry.reset()" in body


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
    the window is not the ramp's -- it is its own, absolute, and capped on short
    runs so it still closes with training left to use the result."""
    schedule = CounterexampleSchedule()

    assert schedule.calibration_steps(total_steps=1000) == DEFAULT_CALIBRATION_STEPS
    assert schedule.calibration_steps(total_steps=4) == MIN_CALIBRATION_STEPS
    assert schedule.calibration_steps(total_steps=0) == DEFAULT_CALIBRATION_STEPS
    # capped to half the run, so a 40-step run still gets 20 steps of training
    # after beta freezes rather than freezing it on the finish line
    assert schedule.calibration_steps(total_steps=40) == 20


def test_beta_does_not_depend_on_how_long_the_run_is():
    """The defect G2 measured: two runs, same model and data, 24 vs 96 steps,
    froze beta 6.4x apart (11,018,508 vs 1,713,595).

    Cause: the window was 25% of the run and `|delta|` grows steeply while it is
    open, so a longer window averaged in larger deltas and solved a *smaller*
    beta. Run length is a scheduling choice; it must not silently rescale the
    objective. Any run long enough to complete the window now solves the same
    beta from the same deltas.
    """
    def freeze_beta(total_steps, deltas):
        schedule = CounterexampleSchedule()
        window = schedule.calibration_steps(total_steps)
        for step in range(total_steps):
            schedule.observe(torch.tensor([deltas(step)]))
            beta = schedule.beta(configured=0.0, step=step, total_steps=total_steps)
            if schedule._frozen_beta is not None:
                return window, beta
        return window, None

    # |delta| growing the way a real run's does, identical across both runs
    def growth(step):
        return 1.0e-7 * 21.0 * (1.0 - math.exp(-(step + 1.0) / 25.0))

    long_runs = [freeze_beta(t, growth)[1] for t in (64, 96, 200, 1000, 5000)]
    assert all(b is not None for b in long_runs), "the window must close on every real run"
    assert max(long_runs) - min(long_runs) < 1e-6 * max(long_runs), (
        f"run length changed the objective: {long_runs}")

    # Below 2x the window the cap bites and runs of different lengths still
    # differ -- irreducibly, a 24-step run cannot observe 32 steps. Bounded, not
    # eliminated: the old rule spanned 6.1x across this whole range.
    short = [freeze_beta(t, growth)[1] for t in (24, 40, 48)]
    assert max(short) / min(short) < 2.5


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


def test_the_coordinate_has_exactly_one_implementation():
    """Everything that reasons about where on the schedule the term acts must not
    be able to disagree about what u is. They call the same function; this pins
    that it is the published one."""
    snr = torch.tensor([((1 - s) / s) ** 2 for s in (0.1, 0.35, 0.5, 0.8)])
    assert torch.allclose(
        noise_level_from_snr(snr), torch.tensor([0.1, 0.35, 0.5, 0.8]), atol=1e-6
    )


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
    """Why the band got its own coordinate instead of reusing
    ``min_noising_strength``.

    Timestep-index fraction is not comparable across model families. On SD 1.5's
    scaled-linear schedule the tenth of the schedule nearest clean is already a
    QUARTER noise by amplitude, while on a rectified flow t/N = 0.1 is sigma =
    0.1 exactly. A band authored on one model and reused on the other would
    silently cover a different noise range -- and nothing would report it.
    """
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
    u = _u(torch.cumprod(1.0 - betas, dim=0))

    assert 0.25 < u[100].item() < 0.27        # t/N = 0.1 -> u ~ 0.26, not 0.10
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


def test_the_dose_is_the_mean_band_weight_over_the_sampled_schedule():
    """`band_dose` is the number the forecast reports and the number
    `counterexample/band_pass` measures. Checked against a schedule whose answer
    is known in closed form: for a rectified flow u IS sigma, so a uniform draw
    over the schedule puts a [0, 0.5] band's dose at the band function's own mean
    over [0, 1]."""
    u = torch.linspace(0.0, 1.0, 100_001)
    dose = band_dose(u, 0.0, 0.5)
    assert math.isclose(dose, noise_band_weight(u, 0.0, 0.5).mean().item(), rel_tol=1e-9)
    # 0.375 at full strength + a raised-cosine edge over [0.375, 0.5], which
    # integrates to half its width.
    assert math.isclose(dose, 0.375 + 0.5 * 0.125, abs_tol=1e-3)


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


class _Mixin(ModelSetupDiffusionLossMixin, ModelSetupNoiseMixin):
    """Both mixins, because every real ``Base*Setup`` has both and the dose
    forecast genuinely reaches across: it draws through the noise mixin's
    ``_get_timestep_discrete`` rather than modelling the sampler. A harness with
    only the loss half would be testing a class that does not exist."""


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
    reference at 2 -- so both distances are 1 and delta is exactly 0, which is
    the step-0 state every cold LoRA is actually in."""
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
    `_diffusion_losses` -- and a term wired into only one of them would be
    silently absent for every non-flow-matching model."""
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.COUNTEREXAMPLE]
    batch, data = _batch_and_data(types_)
    losses = _Mixin()._diffusion_losses(batch, data, _config(batch_size=2), torch.device("cpu"))

    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(losses[1].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    assert TELEMETRY.take().stats.rows == 1


def _run_banded(*, low, high, timestep=499, mixin=None):
    """One flow-matching step with a counterexample row, returning the mixin (so a
    caller can reuse it for a second step) and whatever the run printed."""
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["timestep"] = torch.tensor([timestep])
    mixin = mixin or _Mixin()
    config = _config(
        batch_size=1, counterexample_band_low=low, counterexample_band_high=high
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mixin._flow_matching_losses(
            batch, data, config, torch.device("cpu"), sigmas=torch.zeros(1000)
        )
    return mixin, out.getvalue()


def test_the_dose_forecast_fires_once_and_only_for_a_banded_run():
    """The band is model-agnostic; the DOSE is not, so a run has to be told what
    its band will actually deliver *before* the numbers come back. But only when
    there is a band -- an unbanded run passes everything and has nothing to say."""
    TELEMETRY.reset()
    _, quiet = _run_banded(low=0.0, high=1.0)
    assert quiet == ""

    TELEMETRY.reset()
    mixin, first = _run_banded(low=0.0, high=0.5)
    assert "expected dose" in first
    assert "loss_weight by" in first        # the remedy, not just the diagnosis

    # ...and exactly once, not every step for the rest of the run.
    TELEMETRY.reset()
    _, second = _run_banded(low=0.0, high=0.5, mixin=mixin)
    assert second == ""


def test_a_starving_band_is_called_starvation_not_reported_as_a_dose():
    """The failure the forecast exists for: a band this narrow leaves the rows
    that *do* pass behaving perfectly normally, so nothing downstream reports
    it."""
    TELEMETRY.reset()
    _, out = _run_banded(low=0.0, high=0.02, timestep=5)
    assert "WARNING" in out and "starvation" in out


def test_the_dose_forecast_does_not_disturb_the_training_generator():
    """A diagnostic that draws 50,000 samples from the run's own generator would
    advance its state and change every later noise draw -- making runs silently
    unreproducible in exchange for a log line. It uses a fresh, fixed-seed one."""
    config = _config(batch_size=1, counterexample_band_low=0.0, counterexample_band_high=0.5)

    def draw(mixin, generator):
        return mixin._get_timestep_discrete(1000, False, generator, 4, config).tolist()

    baseline_gen = torch.Generator(device="cpu")
    baseline_gen.manual_seed(1234)
    baseline = draw(_Mixin(), baseline_gen)

    mixin = _Mixin()
    live_gen = torch.Generator(device="cpu")
    live_gen.manual_seed(1234)
    TELEMETRY.reset()
    _run_banded(low=0.0, high=0.5, mixin=mixin)
    assert draw(mixin, live_gen) == baseline


def test_the_band_reaches_the_beta_calibration_and_not_only_the_loss():
    """Found by mutation: `SCHEDULE.observe(delta, band)` -> `observe(delta)` broke
    nothing.

    Every visible number survives it -- the losses are identical, so is the whole
    telemetry window -- and the only casualty is beta, calibrated against a delta
    scale drawn from rows the band removed from training. `test_beta_calibrates_
    on_the_rows_the_band_lets_through` pins the schedule's behaviour and cannot
    see the call site, so this watches the call itself.
    """
    TELEMETRY.reset()
    seen = []
    SCHEDULE.observe = lambda delta, band_weight=None: seen.append(band_weight)
    try:
        _run_banded(low=0.0, high=0.5, timestep=949)   # u = 0.95, fully out of band
    finally:
        del SCHEDULE.observe
    TELEMETRY.reset()

    assert len(seen) == 1
    assert seen[0] is not None                  # the band was handed over...
    assert float(seen[0].sum().item()) == 0.0   # ...and it is the muting one


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

    assert losses[1].item() == 0.0                            # ramped fully out
    assert math.isclose(losses[0].item(), 1.0, rel_tol=1e-6)  # positive untouched
    # The readout separates the two: the objective's own state is unchanged (a
    # cold gate of 0.5), and the ramp reports itself as the reason nothing moved.
    window = TELEMETRY.take()
    assert window.weight == 0.0
    assert math.isclose(window.stats.gate_mean, 0.5)

    # ...and at the end of the ramp the same row carries the full step-0 value.
    TELEMETRY.reset()
    batch, data = _batch_and_data(types_)
    data["counterexample_step"] = 100
    data["counterexample_total_steps"] = 100
    losses = _Mixin()._flow_matching_losses(batch, data, config, torch.device("cpu"))
    assert math.isclose(losses[1].item(), (2.0 / BETA) * math.log(2.0), rel_tol=1e-5)
    assert TELEMETRY.take().weight == 1.0


def test_the_concepts_own_loss_weight_still_scales_it():
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


def test_a_batch_with_no_counterexample_row_is_left_completely_alone():
    """The inertness claim, as a test: every existing config has no concept of
    this type, so the loss must come out bit-identical and the telemetry empty."""
    TELEMETRY.reset()
    types_ = [ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION]
    batch, data = _batch_and_data(types_)
    losses = _Mixin()._flow_matching_losses(batch, data, _config(batch_size=2), torch.device("cpu"))

    assert torch.allclose(losses, torch.ones(2))
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
    """`delta` is meaningless if `d` and `d_ref` come from different metrics.
    Under masked training the mask must apply to both, which a shared helper
    guarantees and two call sites would not.

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


# ---------------------------------------------------------------------------
# Composition with gradient-aligned (GA) initialization.
#
# These features are separate upstream PRs and neither diff can carry the code
# under test: upstream has no GA init, so the counterexample PR cannot name its
# estimation pass; upstream has no counterexamples, so the GA PR has nothing to
# exclude. The coupling lands only where both are present, and these tests are
# the only thing holding it.
#
# The coupling has two ends and they fail differently. The GA pass zeroes these
# rows' targets (GenericTrainer) *and* opts out of the repulsion (the flag read
# below). Dropping the opt-out is loud -- the run crashes on the missing
# reference, as the control here shows. Dropping the target substitution is
# silent: the pass completes and estimates the initialization toward the very
# images the run is meant to be pushed away from.
# ---------------------------------------------------------------------------

def test_the_ga_estimation_pass_opts_out_of_the_repulsion():
    """The GA pass runs the model without the frozen reference, and says so."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE], with_reference=False)
    data["skip_counterexample_repulsion"] = True

    losses = _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))

    assert torch.isfinite(losses).all()
    assert TELEMETRY.take().stats.rows == 0, "an opted-out pass must not report repulsion telemetry"


def test_the_opt_out_did_not_weaken_the_missing_reference_guard():
    """The control. Same batch, flag absent -- the guard must still fire.

    Without it the opt-out would be a general 'skip the loss' switch, and a
    counterexample row reaching the loss with no reference would fall through
    as an ordinary positive: the one failure of this feature a green run never
    reveals.
    """
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE], with_reference=False)
    try:
        _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))
    except RuntimeError as exc:
        assert "COUNTEREXAMPLE" in str(exc)
        return
    raise AssertionError("the flag must not suppress the guard when it is unset")


def test_the_opt_out_is_scoped_to_counterexample_rows():
    """A batch with no counterexample row is untouched either way, so the flag
    can never become a switch that disables ordinary training rows."""
    TELEMETRY.reset()
    cfg = _config(batch_size=2)
    baseline = None
    for flag in (False, True):
        batch, data = _batch_and_data([ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION])
        data["skip_counterexample_repulsion"] = flag
        losses = _Mixin()._flow_matching_losses(batch, data, cfg, torch.device("cpu"))
        if baseline is None:
            baseline = losses.detach().clone()
        else:
            assert torch.allclose(losses, baseline)


def test_the_ga_pass_treats_counterexample_rows_as_inert():
    """The silent half of the coupling, made testable.

    The pass itself needs a model and a dataloader, so this pins the predicate
    instead. A counterexample row left out of this list keeps its ordinary
    reconstruction loss during gradient estimation, and the initialization is
    then aligned toward the near-miss image -- with nothing raised and nothing
    logged.
    """
    from modules.trainer.GenericTrainer import GenericTrainer

    types_ = [
        ConceptType.STANDARD.value,
        ConceptType.COUNTEREXAMPLE.value,
        ConceptType.PRIOR_PREDICTION.value,
        ConceptType.VALIDATION.value,
    ]

    assert GenericTrainer.inert_gradient_init_indices(types_, 4) == [1, 2]


def test_graded_fraction_separates_a_shaped_batch_from_a_sign_test():
    """The number that makes the frozen-scale decision reviewable later.

    Once `|delta|` outgrows the frozen beta the gate stops grading and resolves
    by the sign of `delta` alone -- accepted behaviour, but invisible in
    `gate_mean`, which reads ~0.5 both for a batch genuinely being shaped and for
    a half-and-half batch of saturated rows. `graded_fraction` tells them apart.
    """
    # deltas straddling zero at a scale beta actually grades
    delta = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

    shaped = counterexample_stats(delta, torch.zeros(5), beta=1.0)
    sign_test = counterexample_stats(delta, torch.zeros(5), beta=1_000.0)

    # gate_mean cannot distinguish them: symmetric deltas average to 0.5 either way
    assert shaped.gate_mean == pytest.approx(0.5, abs=1e-6)
    assert sign_test.gate_mean == pytest.approx(0.5, abs=1e-6)

    # graded_fraction can
    assert shaped.graded_fraction > 0.5, "a batch at beta's own scale is being shaped"
    assert sign_test.graded_fraction == pytest.approx(0.2), (
        "only the delta == 0 row (gate exactly 0.5) is still inside the band")


def test_the_graded_band_contains_the_gate_calibration_aims_for():
    """A row sitting exactly where beta was solved to put it is, by definition,
    a row the bound is grading. An upper edge at CALIBRATION_TARGET_GATE would
    score it as saturated -- found by mutating the band's comparison from `<` to
    `<=` and watching nothing fail."""
    assert GRADED_BAND[1] > CALIBRATION_TARGET_GATE
    assert GRADED_BAND[0] < 1.0 - CALIBRATION_TARGET_GATE

    # a row exactly at the calibration target, built the way the solver does
    beta = 1000.0
    delta = math.log(CALIBRATION_TARGET_GATE / (1.0 - CALIBRATION_TARGET_GATE)) / beta
    at_target = counterexample_stats(
        torch.tensor([delta]), torch.zeros(1), beta=beta)
    assert at_target.gate_mean == pytest.approx(CALIBRATION_TARGET_GATE, abs=1e-6)
    assert at_target.graded_fraction == 1.0, "the calibration target must count as graded"


def test_graded_survives_merging_two_batches():
    """`__add__` accumulates per-step stats into TELEMETRY. A field left out of
    it reads plausibly -- just always low -- rather than failing."""
    a = counterexample_stats(torch.tensor([0.0, 0.0]), torch.zeros(2), beta=1.0)
    b = counterexample_stats(torch.tensor([0.0, 0.0]), torch.zeros(2), beta=1.0)
    assert a.graded == 2 and b.graded == 2
    assert (a + b).graded == 4, "graded must accumulate like every other counter"
    assert (a + b).graded_fraction == 1.0


def test_graded_fraction_is_telemetry_only():
    """GRADED_BAND must never reach the objective. If a future edit branches the
    loss on it, the band stops being an observation and becomes a third regime
    nobody specified."""
    delta = torch.linspace(-1.0, 1.0, 32)
    for beta in (1.0, 1_000.0, 1_000_000.0):
        losses = counterexample_losses(
            distance=torch.zeros(32), reference_distance=delta, beta=beta)
        expected = (2.0 / beta) * torch.nn.functional.softplus(beta * delta)
        assert torch.allclose(losses, expected, atol=1e-6), (
            f"the loss at beta={beta} is not the published formula")

    assert GRADED_BAND == (0.02, 0.98)
