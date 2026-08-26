import torch
from torch import Tensor


def _non_batch_dims(tensor: Tensor) -> tuple[int, ...]:
    """Every dimension except the batch dimension.

    The masked-area normalizer has to divide by the mask's mean over its whole
    area. Hardcoding `(1, 2, 3)` covers a 4D `(B, C, H, W)` latent but silently
    omits `W` on a 5D `(B, C, T, H, W)` one, which yields a wrong scale rather
    than an error. `ModelSetupDiffusionLossMixin` already reduces the predicted
    latent generically via `list(range(1, ndim))`; this follows that convention.

    For 4D inputs the result is exactly `(1, 2, 3)`, so nothing changes for any
    image model.
    """
    return tuple(range(1, tensor.dim()))


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=_non_batch_dims(clamped_mask), keepdim=True)

    return losses


def masked_losses_with_prior(
        losses: Tensor,
        prior_losses: Tensor | None,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        masked_prior_preservation_weight: float,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=_non_batch_dims(clamped_mask), keepdim=True)

    if masked_prior_preservation_weight == 0 or prior_losses is None:
        return losses

    clamped_mask = (1 - clamped_mask)
    prior_losses *= clamped_mask * masked_prior_preservation_weight

    if normalize_masked_area_loss:
        prior_losses = prior_losses / clamped_mask.mean(dim=_non_batch_dims(clamped_mask), keepdim=True)

    return losses + prior_losses
