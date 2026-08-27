"""The model-agnostic Concept-Sliders training objectives.

Two of them, one per SliderRegime, sharing the multiplier plumbing and nothing
else. ``_slider_coordinate_loss`` (IMAGE) is documented at its own definition;
what follows is ``_slider_prompt_loss`` (PROMPT_PAIR), the velocity /
flow-matching form of the Concept Sliders objective (Gandikota et al., ECCV
2024):

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

from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType

import torch
from torch import Tensor
from torch.nn import functional as F

# The host's conditioning type. Opaque to the objective -- it is only ever handed
# straight back to run_velocity -- but it is one concrete type per host, so the
# hosts stay type-checked (Anima: a Tensor; SDXL: a SliderConditioning pair).
Conditioning = TypeVar("Conditioning")

# Where the adapter is left once a step is done.
#
# Not `strength`, and not 0.0. The saver is indifferent -- a LoRA file stores the
# factors, and the multiplier is not one of them -- so the only thing this decides
# is what sampling-during-training shows. 1.0 is the multiplier every inference
# stack applies by default, so a training sample at 1.0 previews the file the user
# is about to load. The next step re-zeros it before the base passes anyway.
_RESTING_MULTIPLIER = 1.0


class ModelSetupSliderMixin:
    @staticmethod
    def _check_slider_peft_type(config) -> None:
        """Refuse a PEFT type whose adapter has no signed multiplier.

        A slider *is* the multiplier: the objective needs the adapter off (0.0)
        for the frozen-base passes and at +/-strength for the trained ones, and
        the file is only worth training because the user can dial it afterwards.
        DoRA, OFT and weight-decomposed LoKr recompose the base weight instead of
        adding a scaled delta, so a signed scale is not defined for them and
        LoRAModule raises `NotImplementedError` for any multiplier but 1.0.

        It raises in the *forward*, though -- so without this check a slider run
        configured that way loads the whole model, starts training, and only then
        dies. Same shape of trap as `supported_output_formats()`: a config
        mistake that costs a model load to discover. Checked here at setup time
        instead, once, with a message that names the control to change.

        SVDQuant is the fourth way into the same forward raise and is not a PEFT
        type at all: `quantization.svd_dtype` makes the *base* linears
        `BaseLinearSVD`, whose `forward_with_lora` also refuses a multiplier
        other than 1.0. Gated on the dtype alone, without also proving some part
        is quantized -- an SVD dtype set with nothing quantized builds no SVD
        linears, so refusing it is a false positive, but the only config it
        rejects is one where the setting was already doing nothing, and the
        message names the control that is set.
        """
        if config.quantization.svd_dtype != DataType.NONE:
            raise RuntimeError(
                "Slider training needs an adapter with a signed multiplier, and SVDQuant "
                "('SVD dtype' under quantization) merges the adapter into the base linear, "
                "which has no signed delta scale. Set SVD dtype to NONE to train a slider."
            )

        peft_type = config.peft_type
        if peft_type == PeftType.OFT_2:
            reason = "OFT applies an orthogonal rotation, which has no signed delta scale"
        elif peft_type == PeftType.LORA and config.lora_decompose:
            reason = "DoRA ('Decompose Weights' on the LoRA tab) scales a recomposed weight, not a delta"
        elif peft_type == PeftType.LOKR and config.lokr_weight_decompose:
            reason = "weight-decomposed LoKr ('Decompose Weights' on the LoRA tab) is DoRA-shaped"
        else:
            return
        raise RuntimeError(
            f"Slider training needs an adapter with a signed multiplier, and {reason}. "
            f"Pick LoRA, LoHa, or LoKr without weight decomposition on the LoRA tab."
        )

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


    def _slider_coordinate_loss(
        self,
        run_velocity: Callable[[Sequence[int]], Tensor],
        set_multiplier: Callable[[float], None],
        targets: Sequence[Tensor],
        multipliers: Sequence[float],
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        axis: str | None = None,
    ) -> Tensor:
        """Coordinate-scaled reconstruction loss for the IMAGE regime.

        Each sample carries its own signed multiplier ``m_i = gain_k *
        coordinate_i``, read from that image's caption. The adapter at ``m_i``
        must reconstruct that image's flow-matching target, so what the slider
        learns is a calibrated response *along* the axis rather than two poles:
        with coordinates spread over a range, the only way to fit all of them at
        once is for the adapter's effect to scale with the multiplier, which is
        exactly what the user dials at inference. Binary poles
        (``coordinate in {-1, +1}``) are the special case, and are why an explicit
        image-pair regime is not a separate thing.

        There is no frozen-base pass and no eta here, unlike ``_slider_prompt_loss``.
        The real image IS the supervision; nothing has to be synthesized, so
        nothing has to be guided.

        Args:
            run_velocity: ``(sample_indices) -> velocity`` for those samples as one
                batched forward, rows in the order given. The host has already
                built each sample's noised latent and conditioning; the multiplier
                is set here, before the call.
            set_multiplier: sets the adapter delta scale.
            targets: per-sample flow-matching target, each with a leading batch
                dim of 1 so a group concatenates.
            multipliers: per-sample multiplier, index-aligned with ``targets``.
            loss_fn: mean-reduction elementwise loss (the group weighting below
                assumes a mean).

        Samples sharing a multiplier run as ONE forward. The multiplier is a
        property of the adapter, not of a row, so a batch can only be split by
        distinct multiplier -- but binary poles collapse a whole batch to two
        forwards, and that is the common case. Weighting each group's mean by its
        size makes the result identical to looping one sample at a time, which
        ``test_grouping_is_exactly_the_per_sample_loop`` pins.

        A sample whose multiplier is 0 is dropped before the forward. At
        multiplier 0 the adapter is disabled, so such a term is a constant with
        respect to every trained parameter (measured: sum|dL/dtheta| exactly 0.0
        at m=0, 6.2 at m=0.5) -- keeping it would spend a forward to add a
        constant to the reported loss and divide the real gradient down. Dropping
        it is not a policy about neutral images; it is arithmetic.
        """
        if loss_fn is None:
            loss_fn = F.mse_loss
        if len(targets) != len(multipliers):
            raise ValueError("targets and multipliers must be the same length")
        if not targets:
            raise ValueError("targets must be non-empty")

        # dict preserves insertion order, so the multiplier sequence a test sees
        # is the order the coordinates appeared in the batch.
        groups: dict[float, list[int]] = {}
        for index, multiplier in enumerate(multipliers):
            multiplier = float(multiplier)
            if multiplier != 0.0:
                groups.setdefault(multiplier, []).append(index)

        trained = sum(len(indices) for indices in groups.values())
        if trained == 0:
            # Name the axis that was actually declared. The overwhelmingly common
            # cause is a typo in it, and two nearly-identical strings are exactly
            # what a user cannot diff by eye -- so quote the one we were given and
            # show the literal caption token it would have to match, rather than
            # offering an unrelated example and leaving them to spot the
            # difference themselves.
            if axis:
                raise RuntimeError(
                    f"Every sample in this batch has a slider coordinate of 0, so the batch "
                    f"trains nothing: at multiplier 0 the adapter is disabled and receives no "
                    f"gradient. Not one caption in this batch declared a coordinate on the "
                    f"axis you asked for, {axis!r}, so for these images the axis is absent "
                    f"rather than zero. A caption opts in by spelling that name exactly, as "
                    f"in '({axis}:-2)'. Check the axis name on the Slider tab character by "
                    f"character against a caption -- if the captions say something else, it "
                    f"is the two spellings that disagree, not the data."
                )
            raise RuntimeError(
                "Every sample in this batch has a slider coordinate of 0, so the batch trains "
                "nothing: at multiplier 0 the adapter is disabled and receives no gradient. The "
                "usual cause is a declared axis name that does not match the captions -- check "
                "the axis name on the Slider tab against a caption, e.g. '(distance:-2)'."
            )

        try:
            loss = None
            for multiplier, indices in groups.items():
                set_multiplier(multiplier)
                term = loss_fn(
                    run_velocity(indices),
                    torch.cat([targets[i] for i in indices], dim=0),
                ) * len(indices)
                loss = term if loss is None else loss + term
            return loss / trained
        finally:
            # Same reason as _slider_prompt_loss: a raise partway through must not
            # leave the adapter parked at whatever coordinate it happened to reach.
            set_multiplier(_RESTING_MULTIPLIER)

    # ------------------------------------------------------------------------
    # Host-neutral plumbing.
    #
    # Everything below turns TrainConfig into the objective's arguments and
    # touches no model API at all, so both hosts share it verbatim and it is
    # testable without a model. The one host-specific step -- turning a prompt
    # string into a conditioning -- is passed in as `encode`.
    # ------------------------------------------------------------------------

    def _slider_triples(self, config) -> list:
        """The enabled prompt triples, or a UI-actionable error."""
        triples = [t for t in config.slider_prompts if t.enabled]
        if not triples:
            raise RuntimeError(
                "Slider training needs at least one enabled prompt pair (see the Slider tab)."
            )
        return triples

    def _choose_triple(self, triples: list, rand):
        """Pick one triple, weighted. A weight of 0 excludes a triple without the
        user having to delete it; all-zero weights fall back to uniform rather
        than raising, since that is what an all-zero list plainly means."""
        weights = [max(0.0, t.weight) for t in triples]
        total = sum(weights)
        if total <= 0.0:
            return rand.choice(triples)
        r = rand.random() * total
        acc = 0.0
        for triple, w in zip(triples, weights, strict=True):
            acc += w
            if r <= acc:
                return triple
        return triples[-1]

    @staticmethod
    def _slider_preservation_contexts(config) -> list[str]:
        """The disentanglement set P, parsed from the free-text field.

        Split on newlines *and* pipes: a multi-line text box and a single-line
        entry are the same field on two toolkits, and a user who typed one
        separator should not silently get a single context containing the other.
        """
        raw = config.slider_preservation_prompts.replace("\n", "|")
        return [part.strip() for part in raw.split("|") if part.strip()]

    def _slider_prompt_pairs(self, triple, config) -> tuple[list[str], list[str]]:
        """The c+/c- *prompt strings* whose velocity difference is the guidance
        direction.

        Bare pair (CS Eq. 7) when no preservation set is configured. Otherwise the
        attribute poles are re-stated in each preservation context and the
        objective averages the per-context delta (the disentanglement mean, CS
        Eq. 8). The bare pair is always included, so adding a context widens the
        average rather than replacing it.
        """
        contexts = self._slider_preservation_contexts(config)
        if not contexts:
            return [triple.positive], [triple.negative]

        positive, negative = [], []
        for ctx in [None, *contexts]:
            positive.append(triple.positive if ctx is None else f"{triple.positive}, {ctx}")
            negative.append(triple.negative if ctx is None else f"{triple.negative}, {ctx}")
        return positive, negative

    def _slider_cached_conditioning(
        self,
        text: str,
        encode: Callable[[str], Conditioning],
    ) -> Conditioning:
        """Encode ``text`` once per run.

        The slider prompts are fixed for the whole run and the text encoders are
        frozen, so re-encoding them every step is pure cost -- on Anima that is a
        Qwen3 forward per prompt per step. The cache lives on the setup instance,
        which is created per run.
        """
        cache = getattr(self, "_slider_cond_cache", None)
        if cache is None:
            cache = {}
            self._slider_cond_cache = cache
        if text not in cache:
            cache[text] = encode(text)
        return cache[text]

    @staticmethod
    def _slider_sample_noise_level(config, rand) -> float:
        """A noise level in [0, 1] drawn uniformly from the configured range.

        Bounds are order-insensitive: a user who swaps min and max gets the range
        they clearly meant, not an empty one.
        """
        lo, hi = float(config.slider_sigma_min), float(config.slider_sigma_max)
        lo, hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi))
        return lo + (hi - lo) * rand.random()

    @staticmethod
    def _slider_resolution(config) -> tuple[int, int]:
        """(height, width) in pixels for the synthetic latent.

        Prompt-pair training has no dataset, so there is no bucketing to read the
        resolution from -- it comes from the same `resolution` field the rest of
        the trainer uses. A multi-resolution list trains one slider at one size;
        the first entry wins.
        """
        token = config.resolution.split(",")[0].strip().lower()
        if "x" in token:
            h_str, w_str = token.split("x", 1)
            return int(h_str), int(w_str)
        size = int(token)
        return size, size
