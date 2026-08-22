from abc import ABCMeta
from collections.abc import Callable

from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.counterexample_loss import (
    DOSE_SAMPLES,
    SCHEDULE,
    STARVED_DOSE,
    TELEMETRY,
    band_dose,
    counterexample_losses,
    counterexample_stats,
    counterexample_weight,
    noise_band_weight,
    noise_level_from_snr,
)
from modules.util.loss.masked_loss import masked_losses, masked_losses_with_prior
from modules.util.loss.vb_loss import vb_losses

import torch
import torch.nn.functional as F
from torch import Tensor


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
    __sigmas: Tensor | None

    def __init__(self):
        super().__init__()
        self.__coefficients = None
        self.__alphas_cumprod_fun = None
        self.__sigmas = None
        self.__band_forecast_done = False

    def __log_cosh_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> Tensor:
        diff = pred - target
        loss = diff + torch.nn.functional.softplus(-2.0*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        return loss

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += masked_losses_with_prior(
                losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ),
                prior_losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32)
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ),
                prior_losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.huber_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.vb_loss_strength

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += F.huber_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none',
                delta=config.huber_delta,
            ).mean(mean_dim) * config.huber_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean(mean_dim) * config.vb_loss_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(mean_dim)
            losses /= mask_mean

        return losses

    def _prediction_distance(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            predicted: Tensor,
    ) -> Tensor:
        """The configured per-sample distance between an arbitrary ``predicted``
        and the batch's target.

        Exists so a counterexample's ``d`` and ``d_ref`` are computed by the
        *same* code path (mse/mae/huber/log-cosh mixture, masking, area
        normalization and all), because ``delta = d_ref - d`` is meaningless the
        moment the two halves are different metrics -- and nothing would report
        that they were.

        Two terms are deliberately dropped from the probe:

        * ``prior_target`` -- masked prior preservation compares the prediction
          against the frozen model's output, which for the reference forward *is*
          the prediction, so the term would be identically zero on one side of
          the subtraction and not the other.
        * ``predicted_var_values`` -- the variational bound is a property of the
          trained head, not a reconstruction distance, and has no reference twin.
        """
        probe = {
            key: value
            for key, value in data.items()
            if key not in ("prior_target", "predicted_var_values")
        }
        probe["predicted"] = predicted
        if config.masked_training and not config.model_type.has_conditioning_image_input():
            return self.__masked_losses(batch, probe, config)
        return self.__unmasked_losses(batch, probe, config)

    def _apply_counterexample_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            losses: Tensor,
            noise_level: Tensor | None = None,
    ) -> Tensor:
        """Replace every ``COUNTEREXAMPLE`` row's loss with the bounded repulsion.

        Substituted *before* the loss scaler and ``loss_weight``, so a
        counterexample concept's own ramp still applies on top exactly as it does
        for a positive concept.

        Raises rather than degrading when the frozen reference is missing: a
        counterexample row whose repulsion silently did not apply is a row that
        trained the model **toward** the wrong image, which is the one failure
        mode of this feature that a green run would never reveal. The kron-GA
        estimation pass, which legitimately has no reference forward, opts out
        explicitly via ``data['skip_counterexample_repulsion']``.
        """
        concept_types = batch.get("concept_type")
        if concept_types is None:
            return losses
        indices = [
            i
            for i in range(len(concept_types))
            if ConceptType(concept_types[i]) == ConceptType.COUNTEREXAMPLE
        ]
        if not indices:
            return losses
        if data.get("skip_counterexample_repulsion", False):
            return losses
        if "prior_target" not in data:
            raise RuntimeError(
                "A COUNTEREXAMPLE concept is in the batch but no frozen reference prediction "
                "was computed. Counterexample training needs the prior-model forward, which is "
                "LoRA-only (see BaseModelSetup.prior_model) and is armed in GenericTrainer."
            )

        # `beta` may be auto-calibrated from this run's own delta, so it is asked
        # for per step rather than read straight off the config -- see
        # CounterexampleSchedule. An explicit positive beta is returned unchanged.
        step = int(data.get("counterexample_step", 0))
        total_steps = int(data.get("counterexample_total_steps", 0))
        ramp = config.counterexample_ramp
        beta = SCHEDULE.beta(config.counterexample_beta, step, total_steps)

        index = torch.tensor(indices, device=losses.device, dtype=torch.long)
        distance = self._prediction_distance(batch, data, config, data["predicted"])
        reference_distance = self._prediction_distance(
            batch, data, config, data["prior_target"].detach()
        )
        repulsion = counterexample_losses(distance, reference_distance, beta)
        delta = (reference_distance - distance)[index]

        # The noise band restricts the repulsion to the part of the schedule
        # where a close-but-wrong image actually differs from a right one. It is
        # a *reweighting*, not a resampling, deliberately: the timestep is drawn
        # per sample before concept type is ever consulted, so narrowing
        # `min_noising_strength`/`max_noising_strength` instead would move the
        # positives' schedule too.
        row_noise = None if noise_level is None else noise_level.to(device=losses.device)[index]
        band = (
            None
            if row_noise is None
            else noise_band_weight(
                row_noise, config.counterexample_band_low, config.counterexample_band_high
            )
        )
        # Forecast the dose here rather than before the loop: this is the first
        # moment the schedule AND the resolved timestep shift both exist. It costs
        # one 50k draw, once, and only on a run that has both a band and a
        # counterexample concept -- the only run it could tell anything.
        if band is not None and not self.__band_forecast_done:
            self.__band_forecast_done = True
            if config.counterexample_band_low > 0.0 or config.counterexample_band_high < 1.0:
                self.__forecast_band_dose(config, losses.device)
        SCHEDULE.observe(delta, band)

        # Recorded BEFORE the ramp weight, deliberately: `gate_mean` has to keep
        # describing the objective's own state (is beta scaled for this run's
        # delta?) and not get mixed up with how much of it is currently switched
        # on. The weight is its own scalar instead.
        weight = counterexample_weight(step, total_steps, ramp)
        TELEMETRY.record(
            counterexample_stats(
                delta=delta,
                losses=repulsion[index],
                beta=beta,
                noise_level=row_noise,
                band_weight=band,
            ),
            weight=weight,
            beta=beta,
        )

        losses = losses.clone()
        losses[index] = weight * band * repulsion[index] if band is not None else weight * repulsion[index]
        return losses

    def __noise_level_from_snr(self, timesteps: Tensor, device: torch.device) -> Tensor:
        """``u = 1 / (1 + sqrt(SNR))`` -- the fraction of the noised latent's
        amplitude that is noise, in ``[0, 1]``.

        The one coordinate in which a noise band means the same thing on every
        model family. For a variance-preserving schedule
        ``x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) eps``, so this is exactly
        ``sqrt(1 - a_bar) / (sqrt(a_bar) + sqrt(1 - a_bar))``. For a rectified
        flow ``x_t = (1 - sigma) x_0 + sigma eps`` gives ``SNR = ((1-sigma)/sigma)^2``
        and the same expression collapses to ``sigma`` itself -- which is why the
        flow-matching branch can read sigma straight off its own schedule and be
        speaking the same language.

        Prediction type is deliberately not consulted: ``u`` describes ``x_t``,
        not how the network is asked to parameterize it, so unlike
        ``__min_snr_weight`` there is no v-prediction correction here.
        """
        return noise_level_from_snr(self.__snr(timesteps, device))

    def __noise_level_for(self, timesteps: Tensor, device: torch.device) -> Tensor | None:
        """``u`` for arbitrary timesteps, from whichever schedule this setup was
        given.

        Not a new branch: ``__sigmas`` is set only by ``_flow_matching_losses``
        and ``__coefficients`` only by ``_diffusion_losses``, so "which one is
        populated" *is* how the run already identifies its own family. Returns
        ``None`` when neither is -- a continuous-timestep model (Wuerstchen
        supplies an ``alphas_cumprod_fun`` and samples with
        ``_get_timestep_continuous``) has no discrete schedule to sample over,
        and a forecast built on a made-up one would be worse than no forecast.
        """
        if self.__sigmas is not None:
            return self.__sigmas[timesteps].to(device=device)
        if self.__coefficients is not None:
            return self.__noise_level_from_snr(timesteps, device)
        return None

    def __forecast_band_dose(self, config: TrainConfig, device: torch.device) -> None:
        """Say, once, how much of the repulsion this band will actually deliver.

        The band is model-agnostic; **the dose is not**. ``timestep_shift`` skews
        where samples land, so an identical band delivers 0.57 of the term on
        SD 1.5 and 0.22 on a flow model at shift 7.51 -- a 3x spread that also
        varies *within* a family with resolution, since
        ``model.calculate_timestep_shift`` reads the training size. So this is
        derived per run rather than tabulated per family, the same way that
        method is.

        Estimated by drawing through ``_get_timestep_discrete`` itself -- the run's
        real sampler, with its real distribution and its real shift -- rather than
        by modelling it. A model of the sampler would be a second implementation
        of eight distribution branches, and it would go stale silently.
        """
        # A *discrete* schedule is what makes the forecast possible: it is what
        # `_get_timestep_discrete` samples over. A model that has only an
        # `alphas_cumprod_fun` (Wuerstchen) samples with
        # `_get_timestep_continuous` instead, so there is nothing to draw from
        # and no length to draw it over. The band itself still works there --
        # `__noise_level_from_snr` handles that path fine -- only the forecast is
        # skipped, and it is skipped rather than approximated.
        if self.__sigmas is not None:
            length = self.__sigmas.shape[0]
        elif self.__coefficients is not None:
            length = self.__coefficients.sqrt_alphas_cumprod.shape[0]
        else:
            return
        # A FRESH generator, never the training one: drawing 50k samples from the
        # run's own generator would advance its state and change every subsequent
        # noise draw, so a diagnostic would silently make runs unreproducible.
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        timesteps = self._get_timestep_discrete(
            num_train_timesteps=length,
            deterministic=False,
            generator=generator,
            batch_size=DOSE_SAMPLES,
            config=config,
            shift=self._last_timestep_shift,
        )
        noise_level = self.__noise_level_for(timesteps.long(), device)
        if noise_level is None:
            return

        low, high = config.counterexample_band_low, config.counterexample_band_high
        dose = band_dose(noise_level, low, high)
        shift = self._last_timestep_shift
        print(
            f"counterexample: band [{low:.2f}, {high:.2f}] in u, timestep_shift {shift:.2f}"
            f" -> expected dose (band_pass) ~{dose:.2f}"
        )
        if dose <= 0.0:
            print(
                "counterexample: WARNING this band passes no timesteps at all -- the "
                "repulsion will not train anything. Widen it."
            )
        elif dose < STARVED_DOSE:
            print(
                f"counterexample: WARNING a dose of {dose:.2f} is starvation -- the rows "
                "that do pass will look perfectly healthy and the gate will not report it."
            )
        else:
            print(
                f"counterexample: to match an unbanded arm's dose, multiply the concept's "
                f"loss_weight by {1.0 / dose:.2f}"
            )

    def __snr(self, timesteps: Tensor, device: torch.device) -> Tensor:
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                       self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr

    def __min_snr_weight(
            self,
            timesteps: Tensor,
            gamma: float,
            v_prediction: bool,
            device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    def __debiased_estimation_weight(
        self,
        timesteps: Tensor,
        v_prediction: bool,
        device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        weight = snr
        # The line below is a departure from the original paper.
        # This is to match the Kohya implementation, see: https://github.com/kohya-ss/sd-scripts/pull/889
        # In addition, it helps avoid numerical instability.
        torch.clip(weight, max=1.0e3, out=weight)
        if v_prediction:
            weight += 1.0
        torch.rsqrt(weight, out=weight)
        return weight

    def __p2_loss_weight(
        self,
        timesteps: Tensor,
        gamma: float,
        v_prediction: bool,
        device: torch.device,
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        if v_prediction:
            snr += 1.0
        return (1.0 + snr) ** -gamma

    def __sigma_loss_weight(
        self,
        timesteps: Tensor,
        device: torch.device,
    ) -> Tensor:
        return self.__sigmas[timesteps].to(device=device)

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            betas: Tensor | None = None,
            alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas.to(train_device))

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

            # Guarded on the schedule, not just on the key: with
            # `loss_weight_fn = CONSTANT` and no betas passed, __snr has nothing
            # to read and neither source is guaranteed to exist. A model whose
            # schedule is unavailable simply gets no band rather than a crash on
            # a path that has nothing to do with counterexamples.
            has_schedule = self.__coefficients is not None or self.__alphas_cumprod_fun is not None
            noise_level = (
                self.__noise_level_from_snr(data['timestep'], losses.device)
                if 'timestep' in data and has_schedule
                else None
            )
            losses = self._apply_counterexample_losses(batch, data, config, losses, noise_level)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(batch_size=config.batch_size, accumulation_steps=config.gradient_accumulation_steps)

        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            v_pred = data.get('prediction_type', '') == 'v_prediction'
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.MIN_SNR_GAMMA:
                    losses *= self.__min_snr_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data['timestep'], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for diffusion models")

        return losses

    def _flow_matching_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            sigmas: Tensor | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=train_device)
            self.__sigmas = all_timesteps / num_timesteps

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

            # A flow model's sigma *is* the noise coordinate `u` -- see
            # __noise_level_from_snr, where the general expression collapses to
            # exactly this for a rectified flow.
            noise_level = (
                self.__sigmas[data['timestep']].to(device=losses.device)
                if 'timestep' in data and self.__sigmas is not None
                else None
            )
            losses = self._apply_counterexample_losses(batch, data, config, losses, noise_level)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.SIGMA:
                    losses *= self.__sigma_loss_weight(data['timestep'], losses.device)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for flow matching models")

        return losses
