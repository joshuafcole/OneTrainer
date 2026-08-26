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

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import (  # noqa: E402
    ModelSetupDiffusionLossMixin,
)
from modules.module.LoRAModule import LoRAModule  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.loss.counterexample_loss import (  # noqa: E402
    TELEMETRY,
    CounterexampleTelemetry,
    counterexample_losses,
    counterexample_stats,
)

import torch  # noqa: E402
from torch import nn  # noqa: E402

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


def test_the_shipped_default_beta_is_a_positive_float():
    """The default is a starting convention borrowed from Diffusion-DPO's beta_T,
    not a measured optimum -- but it must at least be a float the objective
    accepts, which the config tuple's shape (name, default, type, nullable) makes
    easy to get wrong."""
    config = TrainConfig.default_values()
    assert isinstance(config.counterexample_beta, float)
    assert config.counterexample_beta == 1000.0
    counterexample_losses(torch.zeros(1), torch.zeros(1), config.counterexample_beta)


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


def test_telemetry_accumulates_across_micro_steps_and_drains():
    """Gradient accumulation calls the loss several times per optimizer step, so
    the readout must sum the window and then reset -- otherwise the first logged
    step reports one micro-batch and every later one reports the whole run."""
    telemetry = CounterexampleTelemetry()
    delta = torch.tensor([1.0, -1.0])
    losses = counterexample_losses(torch.zeros(2), delta, BETA)

    telemetry.record(counterexample_stats(delta, losses, BETA))
    telemetry.record(counterexample_stats(delta, losses, BETA))

    taken = telemetry.take()
    assert taken.rows == 4
    assert math.isclose(taken.saturated_fraction, 0.5)

    assert telemetry.take().rows == 0


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

    stats = TELEMETRY.take()
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
    assert TELEMETRY.take().rows == 1


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
    assert TELEMETRY.take().rows == 0


def test_an_identical_reference_gives_delta_exactly_zero():
    """The shared step-seeded generator means both forwards see the same (x_t, t),
    so a zero adapter makes the two predictions identical -- and delta must then
    be exactly 0, not merely small. Anything else is a metric mismatch between
    the two halves of the subtraction."""
    TELEMETRY.reset()
    batch, data = _batch_and_data([ConceptType.COUNTEREXAMPLE])
    data["prior_target"] = data["predicted"].detach().clone()
    _Mixin()._flow_matching_losses(batch, data, _config(batch_size=1), torch.device("cpu"))

    stats = TELEMETRY.take()
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
    assert math.isclose(TELEMETRY.take().delta_mean, 3.0, rel_tol=1e-6)

    # Everything outside the mask, weighted 0.5: both distances halve, so delta
    # halves. A mask applied to only one side would have given 3.5 or 2.5.
    batch["latent_mask"] = torch.zeros((1, 2, 2, 2))
    masked = _config(batch_size=1, masked_training=True, unmasked_weight=0.5)
    _Mixin()._flow_matching_losses(batch, data, masked, torch.device("cpu"))
    assert math.isclose(TELEMETRY.take().delta_mean, 1.5, rel_tol=1e-6)


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
    assert TELEMETRY.take().rows == 0, "an opted-out pass must not report repulsion telemetry"


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
