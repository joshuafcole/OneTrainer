"""Model-agnostic Concept-Sliders training objective, in velocity / flow-matching
space.

This implements the velocity-space form of the Concept Sliders objective
(Gandikota et al., ECCV 2024) derived for Anima in docs/slider_lora.md S2:

    v*(x_t, c_t, t)  =  v(c_t)  +  eta * mean_p( v(c+,p) - v(c-,p) )

The frozen base supplies the guidance direction; only the adapter trains. The
mixin is deliberately decoupled from any specific model: the host setup passes
in two callables --

  * ``run_velocity(conditioning) -> Tensor`` : ONE model forward at the adapter's
    current multiplier (the host builds the noised latent / timestep once and
    closes over them; only the conditioning varies per call).
  * ``set_multiplier(float) -> None``        : sets the adapter's signed delta
    scale (PeftBase.set_multiplier / LoRAModuleWrapper.set_multiplier).

so the same objective serves SDXL, Flux, Anima, etc. The orchestration here is
the load-bearing part: run the base (multiplier 0, no_grad) to build a detached
target, then the trained pass(es) at +/- strength to fit it.

The eta used here is the *training-time* guidance scale; the user-facing slider
strength at inference is the adapter multiplier and is independent of it (the CS
alpha/eta decoupling).
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


class ModelSetupSliderMixin:
    def _slider_prompt_loss(
        self,
        run_velocity: Callable[[Any], Tensor],
        set_multiplier: Callable[[float], None],
        target_cond: Any,
        positive_conds: Sequence[Any],
        negative_conds: Sequence[Any],
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
                preservation-augmented pairs; the guidance direction is their mean.
                For a bare slider (CS Eq. 7) pass length-1 lists.
            eta: training-time guidance scale (paper uses ~3-4).
            strength: adapter multiplier magnitude used for the trained passes.
            symmetric: also train the -strength direction toward v(c_t) - eta*delta.
                Improves the slider's linearity around 0 (both poles learned).

        Returns the scalar loss (sum of the + and, if symmetric, - direction MSEs).
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        if len(positive_conds) != len(negative_conds) or not positive_conds:
            raise ValueError("positive_conds and negative_conds must be non-empty and equal length")
        if strength == 0.0:
            raise ValueError("strength must be non-zero (0 disables the adapter entirely)")

        # ---- frozen-base guidance direction (adapter disabled) -----------------
        # delta = mean over the preservation set of ( v(c+,p) - v(c-,p) ), computed
        # by the base model with the adapter off. Detached: it is the *target*, not
        # something we backprop through.
        set_multiplier(0.0)
        with torch.no_grad():
            v_base = run_velocity(target_cond)
            delta = None
            for c_plus, c_minus in zip(positive_conds, negative_conds, strict=True):
                d = run_velocity(c_plus) - run_velocity(c_minus)
                delta = d if delta is None else delta + d
            delta = delta / len(positive_conds)

        target_pos = (v_base + eta * delta).detach()

        # ---- trained pass at +strength -----------------------------------------
        set_multiplier(+strength)
        loss = loss_fn(run_velocity(target_cond), target_pos)

        # ---- symmetric trained pass at -strength --------------------------------
        if symmetric:
            target_neg = (v_base - eta * delta).detach()
            set_multiplier(-strength)
            loss = loss + loss_fn(run_velocity(target_cond), target_neg)

        # Leave the adapter at full positive strength; the next step re-zeros it
        # before the base passes anyway, and this is a sane default for any
        # interleaved sampling.
        set_multiplier(1.0)
        return loss

    def _slider_image_pair_loss(
        self,
        run_velocity_for_sample: Callable[[int, float], Tensor],
        set_multiplier: Callable[[float], None],
        targets: Sequence[Tensor],
        strength: float = 1.0,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Image-pair (visual) slider loss, CS Eq. 9.

        The negative-scaled adapter must reconstruct the "before" sample and the
        positive-scaled adapter the "after" sample, both under an empty prompt, so
        the slider's +/- directions align with the visual A->B effect.

        Args:
            run_velocity_for_sample: ``(sample_index, multiplier) -> velocity`` --
                the host sets the multiplier, builds sample ``i``'s noised latent
                (empty prompt) and returns the predicted velocity.
            targets: per-sample target velocity (the flow-matching target, e.g.
                noise - image), index-aligned with run_velocity_for_sample.
            The convention is index 0 = "before"/A (negative side), 1 = "after"/B
                (positive side); additional pairs follow as (A, B, A, B, ...).
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        if len(targets) < 2 or len(targets) % 2 != 0:
            raise ValueError("targets must contain an even number (>=2) of A,B,... samples")

        loss = None
        for i, target in enumerate(targets):
            mult = -strength if (i % 2 == 0) else +strength  # even -> A (neg), odd -> B (pos)
            set_multiplier(mult)
            term = loss_fn(run_velocity_for_sample(i, mult), target)
            loss = term if loss is None else loss + term
        set_multiplier(1.0)
        return loss / len(targets)
