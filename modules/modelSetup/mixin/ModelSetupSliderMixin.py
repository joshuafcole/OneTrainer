"""The model-agnostic Concept-Sliders training objective.

Implements the velocity / flow-matching form of the Concept Sliders objective
(Gandikota et al., ECCV 2024):

    v*(x_t, c_t, t)  =  v(c_t)  +  eta * mean_p( v(c+,p) - v(c-,p) )

The frozen base supplies the guidance direction; only the adapter trains. The
mixin is decoupled from any specific model -- the host setup passes in two
callables:

  * ``run_velocity(conditioning) -> Tensor`` : ONE model forward at the adapter's
    current multiplier. The host builds the noised latent and timestep once and
    closes over them, so only the conditioning varies per call.
  * ``set_multiplier(float) -> None``        : sets the adapter's signed delta
    scale (``LoRAModuleWrapper.set_multiplier``); 0.0 disables it.

so the same objective serves Anima, SDXL, Flux, ... The orchestration is the
load-bearing part: run the base at multiplier 0 under ``no_grad`` to build a
*detached* target, then the trained pass(es) at +/- strength to fit it.

**Epsilon-parameterized models are covered by the same objective.** At a fixed
``(x_t, t)`` the velocity and the noise prediction are related by an affine map
whose coefficients depend only on ``t``: ``v = a(t)*x_t + b(t)*eps``. So
``v(c+) - v(c-) = b(t) * (eps(c+) - eps(c-))``, and fitting
``eps(c_t) + eta*delta_eps`` in epsilon space is the same problem as fitting
``v(c_t) + eta*delta_v`` in velocity space, up to the constant factor ``b(t)^2``
on the MSE. An epsilon host therefore returns its noise prediction from
``run_velocity`` and needs no other change -- which is also the form the paper
itself is written in.

``eta`` is the *training-time* guidance scale. The user-facing slider strength at
inference is the adapter multiplier and is independent of it (the CS alpha/eta
decoupling).
"""

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
from torch import Tensor
from torch.nn import functional as F

# The host's conditioning type. Opaque to the objective -- it is only ever handed
# straight back to run_velocity -- but it is one concrete type per host, so the
# hosts stay type-checked (Anima: a Tensor; SDXL: a SliderConditioning pair).
Conditioning = TypeVar("Conditioning")

# Where the adapter is left once a step is done. Sampling-during-training and the
# saver both expect the adapter at its nominal strength, and the next step re-zeros
# it before the base passes anyway.
_RESTING_MULTIPLIER = 1.0


class ModelSetupSliderMixin:
    def _slider_prompt_loss(
        self,
        run_velocity: Callable[[Conditioning], Tensor],
        set_multiplier: Callable[[float], None],
        target_cond: Conditioning,
        positive_conds: Sequence[Conditioning],
        negative_conds: Sequence[Conditioning],
        eta: float,
        strength: float = 1.0,
        symmetric: bool = True,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Prompt-pair (textual) slider loss at one (x_t, t).

        Args:
            run_velocity: one forward at the current multiplier, given a conditioning.
            set_multiplier: sets the adapter delta scale (0 disables it).
            target_cond: conditioning for the neutral target concept c_t.
            positive_conds / negative_conds: paired c+/c- conditionings. With a
                preservation set P (disentanglement, CS Eq. 8) these are the
                preservation-augmented pairs and the guidance direction is their
                mean. For a bare slider (CS Eq. 7) pass length-1 sequences.
            eta: training-time guidance scale.
            strength: adapter multiplier magnitude used for the trained passes.
            symmetric: also train the -strength pole toward v(c_t) - eta*delta.
                Both poles learned keeps the slider linear around 0.

        Returns the scalar loss (the + direction MSE, plus the - direction's if
        symmetric).
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        if len(positive_conds) != len(negative_conds) or not positive_conds:
            raise ValueError("positive_conds and negative_conds must be non-empty and equal length")
        if strength == 0.0:
            raise ValueError("strength must be non-zero (0 disables the adapter entirely)")

        try:
            # ---- frozen-base guidance direction (adapter disabled) -------------
            # delta = mean over the preservation set of ( v(c+,p) - v(c-,p) ),
            # computed by the base model with the adapter off. no_grad + detach:
            # this is the *target*, not something to backprop through. Without the
            # detach the optimizer can lower the loss by dragging the target toward
            # the prediction, which is the frozen base's whole job to prevent.
            set_multiplier(0.0)
            with torch.no_grad():
                v_base = run_velocity(target_cond)
                delta = None
                for c_plus, c_minus in zip(positive_conds, negative_conds, strict=True):
                    d = run_velocity(c_plus) - run_velocity(c_minus)
                    delta = d if delta is None else delta + d
                delta = delta / len(positive_conds)

            target_pos = (v_base + eta * delta).detach()

            # ---- trained pass at +strength -------------------------------------
            set_multiplier(+strength)
            loss = loss_fn(run_velocity(target_cond), target_pos)

            # ---- symmetric trained pass at -strength ---------------------------
            if symmetric:
                target_neg = (v_base - eta * delta).detach()
                set_multiplier(-strength)
                loss = loss + loss_fn(run_velocity(target_cond), target_neg)

            return loss
        finally:
            # Restore unconditionally: a raise partway through would otherwise leave
            # the adapter parked at -strength for whatever runs next.
            set_multiplier(_RESTING_MULTIPLIER)
