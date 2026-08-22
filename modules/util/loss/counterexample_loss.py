"""Bounded, reference-anchored repulsion for counterexample concepts.

A counterexample is a **close-but-wrong** image: a near-miss the last run
produced and a human marked bad. Positive-only training cannot address it,
because "not this" is not expressible as an image the model should reproduce.

The naive expression -- ``concept.loss_weight = -1``, which the config already
accepts because ``loss_weight`` is an unclamped float -- is gradient ascent on
an unbounded loss. It diverges at linear speed, which the unlearning literature
calls catastrophic collapse. This module is the bounded alternative, the shape
`NPO <https://arxiv.org/pdf/2404.05868>`_ publishes, transplanted into
velocity/epsilon space the same way Diffusion-DPO transplants the DPO objective:

.. code-block::

    d      = || v_theta(x_t, c, t) - v ||^2     trained  (adapter on)
    d_ref  = || v_ref(x_t, c, t)   - v ||^2     frozen   (adapter off)
    delta  = d_ref - d       # > 0  <=>  the adapter fits the WRONG image
                             #           BETTER than the base model does
    L      = (2 / beta) * softplus(beta * delta)

Four properties, and each one is a test in ``tests/test_counterexample_objective.py``:

* **It switches itself off.** ``dL/d(delta) = 2 * sigmoid(beta * delta)``, so once
  the adapter is meaningfully worse than the reference on the bad image the
  gradient vanishes. This is the entire difference from ``loss_weight = -1``,
  which keeps pushing forever.
* **It is bounded in slope.** ``|dL/d(d)| <= 2`` for every input, so one bad row
  cannot dominate a step no matter how far the adapter has drifted.
* **Step 0 is well conditioned.** A LoRA starts at zero, so ``v_theta == v_ref``,
  ``delta == 0`` exactly, ``L == (2/beta) * log 2``, and the gradient is exactly
  half scale -- the same magnitude a positive row carries. No spike.
* **No pairing, no KL baseline, one knob.** ``beta``. The concept's own
  ``loss_weight`` survives as the ramp on top, unchanged.

**Choosing beta.** ``delta`` is a difference of *element-mean* distances, so it
is small -- order 1e-3 to 1e-1 on latents. The switch-off only means anything
when ``beta * |delta|`` is order 1, which puts beta in the hundreds-to-thousands
range, matching Diffusion-DPO's ``beta_T`` for the same element-mean convention.
Do not guess it twice: :data:`TELEMETRY` reports ``gate_mean`` -- the mean of
``sigmoid(beta * delta)``, i.e. the fraction of full strength the term is
actually running at -- and a first short run reads it straight off. ~1.0 means
beta is too small to ever switch off; ~0.0 means the term is inert and the run
learned nothing from its counterexamples.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def counterexample_losses(distance: Tensor, reference_distance: Tensor, beta: float) -> Tensor:
    """Per-sample bounded repulsion, given the two distances and ``beta``.

    Both distances must come from the *same* metric -- see
    ``ModelSetupDiffusionLossMixin._prediction_distance``, which is why that
    helper exists at all. Subtracting an MSE from a Huber would produce a number
    with no meaning and no error.
    """
    if beta <= 0.0:
        raise ValueError(f"counterexample_beta must be positive, got {beta}")
    delta = reference_distance - distance
    return (2.0 / beta) * F.softplus(beta * delta)


@dataclass(frozen=True)
class CounterexampleStats:
    """One step's readout, aggregated over the counterexample rows in the batch.

    ``gate_mean`` is the load-bearing one (see the module docstring): it is the
    mean of ``sigmoid(beta * delta)``, the per-row multiplier on the repulsion
    gradient, so it says directly how much of the term is live. ``saturated``
    is its blunt companion -- the fraction of rows strictly past the reference,
    which is what "the job is done" and "the reference was never the right
    anchor" both look like from the outside.

    Strictly past, and the strictness is load-bearing: a LoRA starts at zero, so
    on the first step of every cold run ``delta`` is *exactly* 0 for every row.
    Counting that as saturated would make ``saturated_fraction`` read 1.0 -- "all
    done" -- at the one moment nothing has happened yet, on every run, regardless
    of outcome. A row sitting exactly on the reference is not past it; its gate
    says so too, at 0.5.
    """

    rows: int
    delta_sum: float
    gate_sum: float
    saturated: int
    loss_sum: float

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

    def __add__(self, other: "CounterexampleStats") -> "CounterexampleStats":
        return CounterexampleStats(
            rows=self.rows + other.rows,
            delta_sum=self.delta_sum + other.delta_sum,
            gate_sum=self.gate_sum + other.gate_sum,
            saturated=self.saturated + other.saturated,
            loss_sum=self.loss_sum + other.loss_sum,
        )


EMPTY_STATS = CounterexampleStats(rows=0, delta_sum=0.0, gate_sum=0.0, saturated=0, loss_sum=0.0)


def counterexample_stats(delta: Tensor, losses: Tensor, beta: float) -> CounterexampleStats:
    """Summarize one batch's counterexample rows. ``delta``/``losses`` are already
    restricted to those rows by the caller."""
    if delta.numel() == 0:
        return EMPTY_STATS
    detached = delta.detach().to(dtype=torch.float32)
    return CounterexampleStats(
        rows=int(detached.numel()),
        delta_sum=float(detached.sum().item()),
        gate_sum=float(torch.sigmoid(beta * detached).sum().item()),
        saturated=int((detached < 0).sum().item()),
        loss_sum=float(losses.detach().to(dtype=torch.float32).sum().item()),
    )


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
        self._accumulated = EMPTY_STATS

    def record(self, stats: CounterexampleStats) -> None:
        self._accumulated = self._accumulated + stats

    def take(self) -> CounterexampleStats:
        """Return what has accumulated since the last call, and reset."""
        taken = self._accumulated
        self._accumulated = EMPTY_STATS
        return taken

    def reset(self) -> None:
        self._accumulated = EMPTY_STATS


TELEMETRY = CounterexampleTelemetry()
