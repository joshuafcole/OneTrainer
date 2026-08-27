"""Bounded, reference-anchored repulsion for counterexample concepts.

A counterexample is a **close-but-wrong** image: a near-miss the model already
produces, which a human has marked as bad. Positive-only training cannot address
it, because "not this" is not expressible as an image the model should
reproduce.

The naive expression -- ``concept.loss_weight = -1``, which the config already
accepts because ``loss_weight`` is an unclamped float -- is gradient ascent on
an unbounded loss. It diverges at linear speed, which the unlearning literature
calls catastrophic collapse. This module is the bounded alternative: the shape
`NPO <https://arxiv.org/pdf/2404.05868>`_ publishes, transplanted into
velocity/epsilon space the same way Diffusion-DPO transplants the DPO objective.

.. code-block::

    d      = || v_theta(x_t, c, t) - v ||^2     trained  (adapter on)
    d_ref  = || v_ref(x_t, c, t)   - v ||^2     frozen   (adapter off)
    delta  = d_ref - d       # > 0  <=>  the adapter fits the WRONG image
                             #           BETTER than the base model does
    L      = (2 / beta) * softplus(beta * delta)

Four properties, and each one is a test in
``tests/test_counterexample_objective.py``:

* **It switches itself off.** ``dL/d(delta) = 2 * sigmoid(beta * delta)``, so
  once the adapter is meaningfully worse than the reference on the bad image the
  gradient vanishes. This is the entire difference from ``loss_weight = -1``,
  which keeps pushing forever.
* **It is bounded in slope.** ``|dL/d(d)| <= 2`` for every input, so one bad row
  cannot dominate a step no matter how far the adapter has drifted.
* **Step 0 is well conditioned.** A LoRA starts at zero, so ``v_theta == v_ref``,
  ``delta == 0`` exactly, ``L == (2/beta) * log 2``, and the gradient is exactly
  half scale -- the same order of magnitude a positive row carries. No spike.
* **No pairing, no KL baseline.** ``beta`` sets the scale; ``counterexample_ramp``
  sets the timing. The concept's own ``loss_weight`` survives as a multiplier on
  top, unchanged.

**beta is a SCALE knob, not a STRENGTH knob.** This is the thing to get right
before reaching for it. ``dL/d(delta) = 2 * sigmoid(beta * delta)``, so at
``delta == 0`` -- which is *exactly* where every cold LoRA starts -- the slope is
``1.0`` for **every** value of beta. beta cannot make the term start gentle; all
it moves is the ``delta`` scale at which the bound engages. Ramping beta up over
training is in fact an *anti*-ramp: small beta early is the regime where the gate
sits at 0.5 and never switches off (constant-rate repulsion, i.e. the unbounded
ascent this module exists to replace), and large beta late is where it finally
bounds. Use :func:`counterexample_weight` to start gentle.

**Choosing beta.** ``delta`` is a difference of *element-mean* distances, so it
is small, and the switch-off only means anything when ``beta * |delta|`` is order
1. The hundreds-to-thousands range quoted from Diffusion-DPO's ``beta_T``
convention **did not survive contact with a real run**: measured on SD 1.5 LoRA,
beta=1000 gave ``beta*|delta| ~ 0.015`` and wanted ~31,500. Worse, ``|delta|``
*grows* as the adapter trains (21x within one 48-step run), so no single
published constant can be right for every model, resolution and stage.

So do not guess it at all: leave ``counterexample_beta`` at its default of ``0``
and :class:`CounterexampleSchedule` measures ``|delta|`` over the calibration
window, solves for the beta that puts *this* run's delta at
:data:`CALIBRATION_TARGET_GATE`, and freezes it. The window pays for itself
twice -- it is both the gentle start and the calibration sample.

**Reading the gate.** :data:`TELEMETRY` reports ``gate_mean`` -- the mean of
``sigmoid(beta * delta)``, i.e. the fraction of full strength the term is
actually running at.

``~1.0`` does **not** mean beta is too small. A gate near 1 requires
``beta * delta >= 2.197``, so it *proves* the bound engaged; what it says is that
``delta`` stayed positive and the term has not yet won. (Beta too small is a gate
pinned at ``0.5`` -- the constant-rate regime that never switches off.)

``~0.0`` means the term is not currently pushing, and that has **two opposite
causes** which the number alone cannot tell apart: the adapter has already been
driven past the reference on every counterexample row (working as designed, the
job is done), or the term never engaged at all -- most commonly because the run
is a warm start, where ``prior_model()`` detaches *every* adapter and so the
"reference" is the foundation model rather than the checkpoint being resumed.
Only the trajectory over training separates them: a gate that started near 0.5
and fell has done its work; one that was near 0 from the first logged step never
started.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# beta is solved so the run's own observed |delta| lands at this gate. 0.9 rather
# than something closer to 1 because the point is a bound that *engages*, not one
# that is already saturated the moment it turns on.
CALIBRATION_TARGET_GATE = 0.9
# Used only before the calibration window closes -- step 0 has delta == 0 exactly,
# so there is nothing to solve from yet and the ramp weight is ~0 anyway.
BOOTSTRAP_BETA = 1000.0
# Calibration window when the run's length is unknown. Enough optimizer steps
# carrying counterexample rows for the mean to mean something, small against any
# real run.
DEFAULT_CALIBRATION_STEPS = 32
# Fraction of the run beta is calibrated over. Deliberately NOT "the ramp window":
# at `counterexample_ramp = 1.0` that would only close on the last step, so beta
# would calibrate exactly when the run ends -- useless. At the 0.25 ramp default
# the two coincide anyway.
CALIBRATION_FRACTION = 0.25
# Floor on the calibration window, so a very short ramp cannot freeze beta off a
# single observation.
MIN_CALIBRATION_STEPS = 8
# A solved beta is clamped into this range: |delta| can be denormal-small on the
# first steps after zero, and 1/tiny is not a hyperparameter.
BETA_BOUNDS = (1.0, 1.0e9)
# Each edge of the noise band is a raised cosine this fraction of the band's
# width. A quarter on each side leaves the middle half of the band at full
# strength, so a band always has a plateau and never degenerates into a spike.
BAND_EDGE_FRACTION = 0.25
# Below this dose the band has narrowed the term nearly out of existence.
# Nothing downstream can see it -- the rows that do pass behave perfectly
# normally -- so the forecast has to say it out loud.
STARVED_DOSE = 0.1
# Timesteps drawn to estimate the dose. Large enough that the estimate's own
# error (~0.2% on a fraction near 0.5) is far below any decision it informs, and
# cheap enough to pay once per run.
DOSE_SAMPLES = 50_000


def counterexample_losses(distance: Tensor, reference_distance: Tensor, beta: float) -> Tensor:
    """Per-sample bounded repulsion, given the two distances and ``beta``.

    Both distances must come from the *same* metric -- see
    ``ModelSetupDiffusionLossMixin._prediction_distance``, which is the reason
    that helper exists at all. Subtracting an MSE from a Huber would produce a
    number with no meaning and no error.
    """
    if beta <= 0.0:
        raise ValueError(f"counterexample_beta must be positive, got {beta}")
    delta = reference_distance - distance
    return (2.0 / beta) * F.softplus(beta * delta)


def ramp_steps(ramp: float, total_steps: int) -> int:
    """Resolve ``counterexample_ramp`` to a step count.

    Mirrors OneTrainer's existing ``learning_rate_warmup_steps`` convention so
    there is one rule to learn, not two: ``0`` disables the ramp, a value in
    ``(0, 1]`` is a **fraction of the whole run**, and anything above 1 is a
    literal optimizer-step count.

    The fraction form is the one that answers "ramp it up towards the end, while
    the LR is annealing": ``counterexample_ramp = 1.0`` reaches full strength
    exactly at the last step, so the term is weakest when the adapter knows
    nothing and strongest when it has something worth correcting.
    """
    if ramp <= 0.0 or total_steps <= 0:
        return 0
    if ramp <= 1.0:
        return int(ramp * total_steps)
    return int(ramp)


def counterexample_weight(step: int, total_steps: int, ramp: float) -> float:
    """Strength multiplier on the repulsion at ``step``, in ``[0, 1]``.

    A raised cosine, not a line: it leaves 0 and arrives at 1 with zero slope, so
    the term neither jolts on at the start of the ramp nor stops changing abruptly
    at the end of it. "Calm" is the requirement -- a linear ramp has a slope
    discontinuity at both ends.

    This is the knob that makes the counterexample term's *timing* independent of
    the positives'. beta cannot do it (see the module docstring): at ``delta == 0``
    the slope is 1.0 for every beta.
    """
    window = ramp_steps(ramp, total_steps)
    if window <= 0:
        return 1.0
    if step >= window:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * max(0, step) / window))


def noise_level_from_snr(snr: Tensor) -> Tensor:
    """``u = 1 / (1 + sqrt(SNR))`` -- the fraction of the noised latent's
    amplitude that is noise, in ``[0, 1]``.

    The one coordinate a noise band means the same thing in on every model
    family. For a variance-preserving schedule
    ``x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) eps``, so it is exactly
    ``sqrt(1 - a_bar) / (sqrt(a_bar) + sqrt(1 - a_bar))``. For a rectified flow
    ``x_t = (1 - sigma) x_0 + sigma eps`` gives ``SNR = ((1-sigma)/sigma)^2``, and
    the same expression collapses to ``sigma`` itself.

    Lives here rather than on the loss mixin so that everything reasoning about
    *where on the schedule* the term acts shares one implementation: a second
    copy of this formula would keep reporting the old answer after the first one
    changed.
    """
    return 1.0 / (1.0 + snr.clamp(min=0.0).sqrt())


def band_dose(noise_level: Tensor, low: float, high: float) -> float:
    """Expected :func:`noise_band_weight` over a sample of noise levels.

    The band's **dose**: the fraction of the repulsion a run will actually
    deliver. Estimated by pushing a large sample of timesteps -- drawn through
    the run's *real* sampler, not a model of it -- through the same band
    function the loss uses.
    """
    return float(noise_band_weight(noise_level, low, high).mean().item())


def _smooth_edge(x: Tensor) -> Tensor:
    """Raised-cosine 0 -> 1 over ``x`` in ``[0, 1]``, flat outside it."""
    clamped = x.clamp(0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * clamped))


def noise_band_weight(noise_level: Tensor, low: float, high: float) -> Tensor:
    """Per-row weight in ``[0, 1]`` restricting repulsion to a noise band.

    ``noise_level`` is the model-agnostic coordinate ``u = 1 / (1 + sqrt(SNR))``
    -- the fraction of the noised latent's amplitude that is noise. ``u = 0`` is
    a clean latent and ``u = 1`` is pure noise, on *every* model family:
    ``ModelSetupDiffusionLossMixin.__noise_level_from_snr`` derives it once from
    the SNR each branch already computes, and for a rectified-flow model it comes
    out exactly equal to sigma.

    That is the whole reason the band is not expressed in timestep indices, the
    coordinate OneTrainer's own ``min_noising_strength`` uses. Index fraction is
    not comparable across model families: on SD 1.5's scaled-linear schedule
    ``t/N = 0.1`` is ``u = 0.26``, while on a rectified-flow model ``t/N = 0.1``
    is ``u = 0.10`` exactly. A band authored on one model and reused on another
    would silently mean a different physical noise range -- and nothing would
    report it.

    ``low <= 0 and high >= 1`` is "no band" and returns all ones, so the default
    is provably a no-op rather than a taper nobody asked for.
    """
    if low <= 0.0 and high >= 1.0:
        return torch.ones_like(noise_level)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(
            f"counterexample noise band must satisfy 0 <= low < high <= 1, got ({low}, {high})"
        )

    edge = BAND_EDGE_FRACTION * (high - low)
    weight = torch.ones_like(noise_level)
    if low > 0.0:
        weight = torch.minimum(weight, _smooth_edge((noise_level - low) / edge))
    if high < 1.0:
        weight = torch.minimum(weight, _smooth_edge((high - noise_level) / edge))
    # minimum, not a product: with edges a quarter of the width each, the two
    # never both bite, so the band keeps a full-strength plateau across its
    # middle half. A product would dip in the centre of a narrow band -- the one
    # place the term is supposed to be strongest.
    return weight


class CounterexampleSchedule:
    """Process-global auto-calibration for ``beta``, alongside :data:`TELEMETRY`.

    Solves ``sigmoid(beta * |delta|) == CALIBRATION_TARGET_GATE`` for beta using
    the ``|delta|`` this run actually produces, over the calibration window, then
    **freezes** it. Frozen rather than tracked, because a beta that keeps chasing
    ``|delta|`` would hold the gate at a constant value forever and destroy the
    one property the bound is for: switching itself off once the adapter is worse
    than the reference.

    Inert when ``counterexample_beta > 0`` -- an explicit beta is always obeyed.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._abs_delta_sum = 0.0
        self._observed_steps = 0
        self._frozen_beta: float | None = None

    def observe(self, delta: Tensor, band_weight: Tensor | None = None) -> None:
        """Record one step's counterexample rows for the calibration mean.

        ``band_weight`` matters more than it looks: beta is solved so that *the
        rows that actually train* land at the target gate. A band that mutes the
        high-noise rows also removes them from the objective, so calibrating on
        them would solve for a delta scale the run never optimizes -- and since
        |delta| varies strongly with noise level, that is not a rounding error.
        A step whose rows are entirely out of band contributes nothing at all,
        rather than contributing a zero that would drag the mean down.
        """
        if delta.numel() == 0:
            return
        magnitude = delta.detach().abs().to(dtype=torch.float32)
        if band_weight is None:
            self._abs_delta_sum += float(magnitude.mean().item())
        else:
            weights = band_weight.detach().to(dtype=torch.float32)
            total = float(weights.sum().item())
            if total <= 0.0:
                return
            self._abs_delta_sum += float((magnitude * weights).sum().item()) / total
        self._observed_steps += 1

    def calibration_steps(self, total_steps: int) -> int:
        """How many *training* steps beta is measured over before it freezes."""
        if total_steps <= 0:
            return DEFAULT_CALIBRATION_STEPS
        return max(int(CALIBRATION_FRACTION * total_steps), MIN_CALIBRATION_STEPS)

    def beta(self, configured: float, step: int, total_steps: int) -> float:
        """The beta to use this step: ``configured`` when > 0, else the solved one.

        :data:`BOOTSTRAP_BETA` holds for the *whole* calibration window, and the
        solved value takes over only once the window closes. The obvious
        alternative -- solve from whatever has accumulated so far and use that
        provisionally -- was tried and is unusable: the solve is
        ``logit / mean|delta|``, and a cold LoRA's first deltas are ~0, so on a
        real run it produced beta = 15,000,000 at step 6 decaying to 400,000 by
        step 20. A 30x swing in the objective's own scale across the window. That
        run survived only because the ramp had the weight at ~0.03 while it
        happened; with ``counterexample_ramp = 0`` it would have run at full
        strength.

        One clean discontinuity when the window closes, rather than a chaotic
        sequence of them throughout it.
        """
        if configured > 0.0:
            return configured
        if self._frozen_beta is not None:
            return self._frozen_beta
        # The window closes on a TRAINING step count, not an observed-delta count.
        # Sizing it in one and counting it in the other is how the window fails to
        # close at all: with batch 2 and two concepts, ~25% of steps carry no
        # counterexample row, so `observed` trails `step` forever and beta stays on
        # the bootstrap value for the whole run -- silently, since a bootstrap beta
        # is indistinguishable from a configured one. Observed on a real run.
        if step < self.calibration_steps(total_steps):
            return BOOTSTRAP_BETA
        # ...but still require enough actual observations to solve from.
        if self._observed_steps < MIN_CALIBRATION_STEPS:
            return BOOTSTRAP_BETA
        solved = self._solve()
        if solved is None:
            return BOOTSTRAP_BETA
        self._frozen_beta = solved
        return solved

    def _solve(self) -> float | None:
        if self._observed_steps == 0:
            return None
        mean_abs_delta = self._abs_delta_sum / self._observed_steps
        if mean_abs_delta <= 0.0:
            return None
        logit = math.log(CALIBRATION_TARGET_GATE / (1.0 - CALIBRATION_TARGET_GATE))
        return min(max(logit / mean_abs_delta, BETA_BOUNDS[0]), BETA_BOUNDS[1])


SCHEDULE = CounterexampleSchedule()


@dataclass(frozen=True)
class CounterexampleStats:
    """One step's readout, aggregated over the counterexample rows in the batch.

    ``gate_mean`` is the load-bearing one (see the module docstring): it is the
    mean of ``sigmoid(beta * delta)``, the per-row multiplier on the repulsion
    gradient, so it says directly how much of the term is live -- and near-zero
    is ambiguous between "finished" and "never started". ``saturated`` is its
    blunt companion: the fraction of rows strictly past the reference.

    Strictly past, and the strictness is load-bearing: a LoRA starts at zero, so
    on the first step of every cold run ``delta`` is *exactly* 0 for every row.
    Counting that as saturated would make ``saturated_fraction`` read 1.0 -- "all
    done" -- at the one moment nothing has happened yet, on every run, regardless
    of outcome. A row sitting exactly on the reference has not passed it; its
    gate says so too, at 0.5.
    """

    rows: int
    delta_sum: float
    gate_sum: float
    saturated: int
    loss_sum: float
    noise_level_sum: float = 0.0
    """Sum of the rows' noise coordinates ``u``, so the mean says *where on the
    schedule* the term was operating -- the number a band is set against."""
    band_sum: float = 0.0
    """Sum of the rows' band weights. Its mean is the term's **dose**: a band
    that passes 30% of rows delivers 30% of the repulsion, and an A/B that does
    not match it across arms is comparing two different treatments."""

    @property
    def delta_mean(self) -> float:
        return self.delta_sum / self.rows if self.rows else 0.0

    @property
    def gate_mean(self) -> float:
        return self.gate_sum / self.rows if self.rows else 0.0

    @property
    def saturated_fraction(self) -> float:
        return self.saturated / self.rows if self.rows else 0.0

    @property
    def loss_mean(self) -> float:
        return self.loss_sum / self.rows if self.rows else 0.0

    @property
    def noise_level_mean(self) -> float:
        return self.noise_level_sum / self.rows if self.rows else 0.0

    @property
    def band_mean(self) -> float:
        return self.band_sum / self.rows if self.rows else 0.0

    def __add__(self, other: "CounterexampleStats") -> "CounterexampleStats":
        return CounterexampleStats(
            rows=self.rows + other.rows,
            delta_sum=self.delta_sum + other.delta_sum,
            gate_sum=self.gate_sum + other.gate_sum,
            saturated=self.saturated + other.saturated,
            loss_sum=self.loss_sum + other.loss_sum,
            noise_level_sum=self.noise_level_sum + other.noise_level_sum,
            band_sum=self.band_sum + other.band_sum,
        )


EMPTY_STATS = CounterexampleStats(rows=0, delta_sum=0.0, gate_sum=0.0, saturated=0, loss_sum=0.0)


def counterexample_stats(
    delta: Tensor,
    losses: Tensor,
    beta: float,
    noise_level: Tensor | None = None,
    band_weight: Tensor | None = None,
) -> CounterexampleStats:
    """Summarize one batch's counterexample rows. Every tensor is already
    restricted to those rows by the caller."""
    if delta.numel() == 0:
        return EMPTY_STATS
    detached = delta.detach().to(dtype=torch.float32)
    rows = int(detached.numel())
    return CounterexampleStats(
        rows=rows,
        delta_sum=float(detached.sum().item()),
        gate_sum=float(torch.sigmoid(beta * detached).sum().item()),
        saturated=int((detached < 0).sum().item()),
        loss_sum=float(losses.detach().to(dtype=torch.float32).sum().item()),
        noise_level_sum=(
            float(noise_level.detach().to(dtype=torch.float32).sum().item())
            if noise_level is not None
            else 0.0
        ),
        # No band means every row passes, which is a dose of 1.0 per row -- not
        # a dose of zero. Defaulting this to 0 would report an unbanded run as
        # having delivered nothing.
        band_sum=(
            float(band_weight.detach().to(dtype=torch.float32).sum().item())
            if band_weight is not None
            else float(rows)
        ),
    )


@dataclass(frozen=True)
class CounterexampleWindow:
    """One optimizer step's drained readout: what happened, and under what settings."""

    stats: CounterexampleStats
    weight: float
    """The ramp's strength multiplier in force for the step, in [0, 1]."""
    beta: float
    """The beta actually used -- the configured one, or the value solved from this
    run's own delta when ``counterexample_beta = 0``."""


class CounterexampleTelemetry:
    """Process-global accumulator between the loss and the trainer's logger.

    A module-level singleton rather than state on the model setup, because the
    two ends are a mixin and a trainer that share no type: ``calculate_loss`` is
    reached through :class:`~modules.modelSetup.BaseModelSetup.BaseModelSetup`,
    which knows nothing about diffusion losses, and reaching across with
    ``getattr`` would make the readout optional -- exactly the property a gate
    against "the term was inert all along" must not have.

    Accumulates across gradient-accumulation micro-steps and is drained once per
    optimizer step by :meth:`take`.
    """

    def __init__(self) -> None:
        self.reset()

    def record(
        self,
        stats: CounterexampleStats,
        *,
        weight: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Accumulate one micro-step.

        ``weight`` and ``beta`` are the *settings* in force, not sums: they are
        constant within an optimizer step, so the last one seen is the one that
        described the whole window.
        """
        self._accumulated = self._accumulated + stats
        self._weight = weight
        self._beta = beta

    def take(self) -> CounterexampleWindow:
        """Return what has accumulated since the last call, and reset.

        A record rather than a bare tuple: ``weight`` and ``beta`` are settings in
        force, not sums, and reading them off a positional tuple at the one call
        site that matters is how they end up transposed.
        """
        taken = CounterexampleWindow(
            stats=self._accumulated, weight=self._weight, beta=self._beta
        )
        self.reset()
        return taken

    def reset(self) -> None:
        self._accumulated = EMPTY_STATS
        self._weight = 1.0
        self._beta = 0.0


TELEMETRY = CounterexampleTelemetry()
