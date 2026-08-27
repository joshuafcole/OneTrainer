"""Masked-area loss normalization over the whole non-batch area.

`normalize_masked_area_loss` divides by the mask's mean. Taking that mean over a
hardcoded `(1, 2, 3)` is correct for a 4D `(B, C, H, W)` latent and wrong for a
5D `(B, C, T, H, W)` one, where it omits `W` -- a wrong scale, not a crash.
Upstream reaches the loss in 5D for `vae_frame_dim` models such as HunyuanVideo.
"""

import pytest
import torch

from modules.util.loss.masked_loss import masked_losses, masked_losses_with_prior


def _mask(shape: tuple[int, ...], covered: float) -> torch.Tensor:
    """A mask whose mean over the non-batch area is exactly `covered`.

    Coverage is laid out along the LAST dim on purpose. A mask that is uniform
    along `W` is normalized identically by the correct reduction and by the
    hardcoded `(1, 2, 3)` one, so it cannot tell them apart -- these tests would
    pass against the very defect they exist to catch.
    """
    width = shape[-1]
    covered_width = round(width * covered)
    assert covered_width / width == covered, "coverage must be exact for this width"

    mask = torch.zeros(shape)
    mask[..., :covered_width] = 1.0
    return mask


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (2, 4, 3, 8, 8)])
def test_normalizer_divides_by_the_whole_area_mean(shape: tuple[int, ...]):
    covered = 0.25
    mask = _mask(shape, covered)
    losses = torch.ones(shape)

    out = masked_losses(losses.clone(), mask, unmasked_weight=0.0, normalize_masked_area_loss=True)

    # Inside the mask the clamped weight is 1, so the normalized loss is 1/covered.
    inside = out[mask == 1]
    assert torch.allclose(inside, torch.full_like(inside, 1.0 / covered))


def test_4d_result_is_unchanged_by_the_generalization():
    """The 4D path must be bit-identical to the old hardcoded (1, 2, 3) reduction."""
    shape = (2, 4, 8, 8)
    mask = _mask(shape, 0.375)
    losses = torch.rand(shape)

    expected = losses * torch.clamp(mask, 0.1, 1)
    expected = expected / torch.clamp(mask, 0.1, 1).mean(dim=(1, 2, 3), keepdim=True)

    actual = masked_losses(losses.clone(), mask, unmasked_weight=0.1, normalize_masked_area_loss=True)

    assert torch.equal(actual, expected)


def test_5d_reduction_over_1_2_3_leaves_w_unreduced():
    """Guard the defect directly.

    On a 5D tensor, `mean(dim=(1, 2, 3), keepdim=True)` does not produce a
    per-sample scalar at all -- it leaves `W` intact and yields a `[B,1,1,1,W]`
    tensor that then broadcasts back across the loss. The correct reduction is
    a `[B,1,1,1,1]` scalar per sample.
    """
    shape = (2, 4, 3, 8, 8)
    mask = torch.zeros(shape)
    mask[..., :2] = 1.0  # non-uniform along W, so the two reductions disagree

    whole_area = mask.mean(dim=(1, 2, 3, 4), keepdim=True)
    omitting_w = mask.mean(dim=(1, 2, 3), keepdim=True)

    assert tuple(whole_area.shape) == (2, 1, 1, 1, 1)
    assert tuple(omitting_w.shape) == (2, 1, 1, 1, 8)
    assert whole_area.flatten()[0].item() == pytest.approx(0.25)

    out = masked_losses(torch.ones(shape), mask, unmasked_weight=0.0, normalize_masked_area_loss=True)
    inside = out[mask == 1]
    assert torch.allclose(inside, torch.full_like(inside, 1.0 / 0.25))


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (2, 4, 3, 8, 8)])
def test_prior_branch_normalizes_over_the_same_dims(shape: tuple[int, ...]):
    mask = _mask(shape, 0.25)
    out = masked_losses_with_prior(
        torch.ones(shape),
        torch.ones(shape),
        mask,
        unmasked_weight=0.0,
        normalize_masked_area_loss=True,
        masked_prior_preservation_weight=1.0,
    )
    assert torch.isfinite(out).all()
    # Prior weight lands on the complement, whose non-batch mean is 1 - covered.
    outside = out[mask == 0]
    assert torch.allclose(outside, torch.full_like(outside, 1.0 / 0.75))
