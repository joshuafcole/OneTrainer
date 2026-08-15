"""Tests for scripts/util/lora_soup.py.

These pin the merge engine the whole preference-soup arc reuses, so they pin the
*decisions* rather than the surface: the alpha/rank scale, that coefficients are
applied as given, that the merge happens in delta space rather than factor
space, that an absent layer means zero, and every refusal.

Pure torch + safetensors, no training stack and no model code -- every fixture
is a synthetic state dict built here. Run with::

    python -m pytest tests/test_lora_soup.py -q
"""

import importlib.util
import itertools
import json
import math
import os
import sys

import torch

import pytest
from safetensors.torch import load_file, safe_open, save_file

_spec = importlib.util.spec_from_file_location(
    "lora_soup",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "util", "lora_soup.py"),
)
lora_soup = importlib.util.module_from_spec(_spec)
# Register before executing: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["lora_soup"] = lora_soup
_spec.loader.exec_module(lora_soup)

# Importing lora_soup puts the repo root on sys.path (it reaches modules/ for
# the LoKr Kronecker math), so this resolves. The LoKr reference deltas below
# deliberately go through the *same* make_kron/rebuild_tucker the engine uses:
# a second copy of the Kronecker index convention here would test torch.kron,
# not the loader. What the references pin is everything around that call --
# which factors get assembled, the reshape, the dim inference, the scale.
from modules.util.lokr_utils import make_kron, rebuild_tucker  # noqa: E402

SoupError = lora_soup.SoupError

# fp16 round-trip tolerance. The factors are stored at half precision on both
# sides, so a delta reconstructed from them carries a relative error on the
# order of the half-precision epsilon (2**-11 ~ 4.9e-4) accumulated over the
# rank-r inner sum. Measured on these fixtures it is 2.9e-4; 2e-3 leaves ~7x
# headroom without being a precision lottery, while a wrong scale (off by a
# factor, not a few ulps) fails it instantly. bf16 storage would land at
# ~2.3e-3 -- these fixtures are fp16 on purpose.
FP16_REL_TOL = 2e-3
# fp32 storage plus an fp32 SVD: only the LAPACK error remains.
FP32_REL_TOL = 1e-5


def rel_err(actual, expected):
    """Relative Frobenius error. Scale-free, so one tolerance covers all layers."""
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def make_layer(out_features, in_features, rank, alpha=None, kernel=None, seed=0):
    """One synthetic (A, B, alpha) triple, Linear or Conv2d shaped."""
    g = torch.Generator().manual_seed(seed)
    if kernel is None:
        down = torch.randn(rank, in_features, generator=g)
        up = torch.randn(out_features, rank, generator=g)
    else:
        down = torch.randn(rank, in_features, *kernel, generator=g)
        up = torch.randn(out_features, rank, 1, 1, generator=g)
    return down, up, float(rank if alpha is None else alpha)


def write_lora(path, layers, bundle=None, header=None, dtype=torch.float16):
    """Serialize {prefix: (A, B, alpha)} the way AnimaLoRASaver does."""
    state_dict = {}
    for prefix, (down, up, alpha) in layers.items():
        # clone: safetensors refuses aliased storages, and a transposed view of
        # a size-1 dimension still counts as contiguous
        state_dict[prefix + ".lora_down.weight"] = down.to(dtype).contiguous().clone()
        state_dict[prefix + ".lora_up.weight"] = up.to(dtype).contiguous().clone()
        state_dict[prefix + ".alpha"] = torch.tensor(alpha, dtype=dtype)
    for key, tensor in (bundle or {}).items():
        state_dict[key] = tensor.to(dtype).contiguous().clone()
    save_file(state_dict, str(path), header)
    return path


def delta_of(state_dict, prefix):
    """(alpha/rank)*B@A, computed independently of the module under test."""
    a = state_dict[prefix + ".lora_down.weight"].float()
    b = state_dict[prefix + ".lora_up.weight"].float()
    alpha = float(state_dict[prefix + ".alpha"].item())
    rank = a.shape[0]
    return (b.reshape(b.shape[0], -1) @ a.reshape(rank, -1)) * (alpha / rank)


def rank_of(state_dict, prefix):
    return state_dict[prefix + ".lora_down.weight"].shape[0]


_soup_counter = itertools.count()


def soup_to_dict(tmp_path, specs, **kwargs):
    """Run a merge and hand back (state_dict, header) as written and re-read.

    Each call gets its own output path: ``load_file`` memory-maps, so reusing
    one filename would silently rewrite an earlier result under the caller.
    """
    out = tmp_path / f"soup{next(_soup_counter)}.safetensors"
    state_dict, header = lora_soup.soup_files(specs, log=lambda _m: None, **kwargs)
    save_file(state_dict, str(out), header)
    with safe_open(str(out), framework="pt") as f:
        read_header = dict(f.metadata() or {})
    return load_file(str(out)), read_header


P1 = "transformer.transformer_blocks.0.attn1.to_q"
P2 = "transformer.transformer_blocks.0.attn2.to_k"
P3 = "transformer.transformer_blocks.1.ff.net.0.proj"


# --------------------------------------------------------------------------
# round trip and the alpha convention
# --------------------------------------------------------------------------

def test_single_input_at_one_round_trips(tmp_path):
    """One file at coefficient 1.0 reproduces its own delta.

    alpha is deliberately rank/2, so a merge that dropped the (alpha/rank)
    scale would be off by 2x and could not hide inside the tolerance.
    """
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=1)}
    a = write_lora(tmp_path / "a.safetensors", layers)
    reference = delta_of(load_file(str(a)), P1)

    merged, _ = soup_to_dict(tmp_path, [(a, 1.0)])

    assert rel_err(delta_of(merged, P1), reference) < FP16_REL_TOL


def test_output_alpha_equals_output_rank(tmp_path):
    """The stated convention: alpha == rank, so the loader's scale is 1.0 and
    the factors alone are the delta."""
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=1)})
    merged, _ = soup_to_dict(tmp_path, [(a, 1.0)])

    rank = rank_of(merged, P1)
    assert rank == 4
    assert float(merged[P1 + ".alpha"].item()) == float(rank)

    bare = merged[P1 + ".lora_up.weight"].float() @ merged[P1 + ".lora_down.weight"].float()
    assert rel_err(bare, delta_of(merged, P1)) < FP32_REL_TOL


def test_two_identical_files_at_half_each_equals_either(tmp_path):
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=2)}
    a = write_lora(tmp_path / "a.safetensors", layers)
    b = write_lora(tmp_path / "b.safetensors", layers)
    reference = delta_of(load_file(str(a)), P1)

    merged, _ = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)])

    assert rel_err(delta_of(merged, P1), reference) < FP16_REL_TOL


def test_coefficients_are_used_as_given_not_normalized(tmp_path):
    """A single input at 1.5 is a 1.5x rescale, not a normalized no-op."""
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=3)})
    reference = delta_of(load_file(str(a)), P1)

    merged, _ = soup_to_dict(tmp_path, [(a, 1.5)])

    assert rel_err(delta_of(merged, P1), reference * 1.5) < FP16_REL_TOL


def test_delta_is_computed_in_float32_regardless_of_storage(tmp_path):
    """Half-precision storage must not become half-precision arithmetic.

    The factors here are large enough that an fp16 matmul overflows to inf --
    the honest failure. The quiet failure is the ordinary case, where an fp16
    accumulation is merely a few ulps worse and hides inside any reconstruction
    tolerance, so the dtype itself is asserted too.
    """
    down = torch.full((2, 4), 200.0, dtype=torch.float16)
    up = torch.full((3, 2), 200.0, dtype=torch.float16)
    layer = lora_soup.LoraLayer(down=down, up=up, alpha=2.0)

    delta = layer.delta()
    assert delta.dtype == torch.float32
    assert torch.isfinite(delta).all(), "fp16 arithmetic overflows here; the promotion is what prevents it"
    assert delta[0, 0].item() == pytest.approx(200.0 * 200.0 * 2 * (2.0 / 2))

    # ...and end to end, through the file path
    a = write_lora(tmp_path / "a.safetensors", {P1: (down, up, 2.0)}, dtype=torch.float16)
    merged, _ = soup_to_dict(tmp_path, [(a, 1.0)], dtype=torch.float32)
    assert torch.isfinite(merged[P1 + ".lora_down.weight"]).all()
    assert rel_err(delta_of(merged, P1), delta) < FP16_REL_TOL


# --------------------------------------------------------------------------
# svd vs concat, and factor averaging
# --------------------------------------------------------------------------

def test_svd_and_concat_agree_on_a_real_pair(tmp_path):
    """Both re-factoring paths must produce the same delta.

    This is the test that catches an alpha/rank convention error in either
    path: concat folds alpha_i/rank_i into its B blocks, svd folds it into the
    delta before factoring, and only agreement proves both spell it the same.
    Target rank 8 is the sum of the input ranks, so the SVD is not truncating.
    """
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=4)}, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 12, 4, alpha=8.0, seed=5)}, dtype=torch.float32)
    specs = [(a, 0.3), (b, 0.7)]

    by_svd, _ = soup_to_dict(tmp_path, specs, rank=8)
    by_concat, _ = soup_to_dict(tmp_path, specs, method="concat")

    assert rank_of(by_concat, P1) == 8
    assert rel_err(delta_of(by_svd, P1), delta_of(by_concat, P1)) < FP32_REL_TOL


def test_concat_survives_a_small_coefficient_in_fp16(tmp_path):
    """A small coefficient must not be parked entirely on one stored tensor.

    456 searches coefficients bounded at [-0.25, 1.5] and will visit values near
    zero. Folding the whole per-input factor into B drives fp16 B entries
    subnormal, so the candidate the search evaluates is not the candidate it
    asked for -- a degradation that reads as a bad merge coefficient rather than
    as a storage artifact.

    **The magnitudes matter and `make_layer` does not have them.** ``lora_up``
    is initialized to *zero* and grows small during training, so a trained
    adapter's B entries are ~1e-3 while ``randn`` gives ~1. With randn-sized
    factors this test passes against the unbalanced implementation -- measured,
    not assumed. At 1e-3 the two diverge sharply: whole-into-B gives 3.4e-3 at
    c=0.01 and 2.9e-2 at c=0.001, against 3.0e-4 and 7.1e-4 for the sqrt split.
    """
    generator = torch.Generator().manual_seed(21)
    small = {
        P1: (
            torch.randn(4, 12, generator=generator),
            torch.randn(16, 4, generator=generator) * 1e-3,  # a *trained* B
            2.0,
        )
    }
    a = write_lora(tmp_path / "a.safetensors", small, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", small, dtype=torch.float32)

    for coefficient in (0.01, 0.001):
        specs = [(a, coefficient), (b, coefficient)]
        reference, _ = soup_to_dict(tmp_path, specs, dtype=torch.float32, method="concat")
        stored, _ = soup_to_dict(tmp_path, specs, dtype=torch.float16, method="concat")
        error = rel_err(delta_of(stored, P1), delta_of(reference, P1))
        assert error < FP16_REL_TOL, f"c={coefficient} lost precision in storage: {error}"


def test_concat_handles_a_negative_coefficient(tmp_path):
    """Negative coefficients are legal (bounded extrapolation), and the sqrt
    split has to put the sign somewhere -- it rides on B. Without that, the root
    of a negative factor is either a crash or a dropped sign, and a dropped sign
    is the silent one."""
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=13)}, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 12, 4, alpha=6.0, seed=14)}, dtype=torch.float32)
    specs = [(a, 1.0), (b, -0.25)]

    by_concat, _ = soup_to_dict(tmp_path, specs, dtype=torch.float32, method="concat")
    by_svd, _ = soup_to_dict(tmp_path, specs, dtype=torch.float32, rank=8)
    assert rel_err(delta_of(by_concat, P1), delta_of(by_svd, P1)) < FP32_REL_TOL


def test_merges_deltas_not_factors(tmp_path):
    """The average of factorizations is not a factorization of the average.

    A = e1,B = e1 and A = e2,B = e2 have deltas E11 and E22, whose average is
    diag(.5,.5) -- while the average of the factors is the constant matrix
    0.25. A naive implementation that averaged (B, A) would produce the latter,
    so this test fails loudly for it rather than drifting a few ulps.
    """
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0]])
    a = write_lora(tmp_path / "a.safetensors", {P1: (e1, e1.T, 1.0)}, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", {P1: (e2, e2.T, 1.0)}, dtype=torch.float32)

    delta_average = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    factor_average = torch.full((2, 2), 0.25)

    for kwargs in ({"rank": 2}, {"method": "concat"}):
        merged, _ = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)], **kwargs)
        got = delta_of(merged, P1)
        assert rel_err(got, delta_average) < FP32_REL_TOL, kwargs
        assert rel_err(got, factor_average) > 0.5, kwargs


# --------------------------------------------------------------------------
# shapes: conv, differing ranks, disagreement
# --------------------------------------------------------------------------

def test_conv_shaped_layer_merges(tmp_path):
    """4-D factors fold the kernel exactly as PeftBase.make_weight does."""
    layers_a = {P1: make_layer(8, 6, 4, alpha=2.0, kernel=(3, 3), seed=6)}
    layers_b = {P1: make_layer(8, 6, 4, alpha=2.0, kernel=(3, 3), seed=7)}
    a = write_lora(tmp_path / "a.safetensors", layers_a, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", layers_b, dtype=torch.float32)
    expected = 0.5 * delta_of(load_file(str(a)), P1) + 0.5 * delta_of(load_file(str(b)), P1)

    merged, _ = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)], rank=8)

    assert merged[P1 + ".lora_down.weight"].shape == (8, 6, 3, 3)
    assert merged[P1 + ".lora_up.weight"].shape == (8, 8, 1, 1)
    assert rel_err(delta_of(merged, P1), expected) < FP32_REL_TOL


def test_differing_ranks_merge(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=8)}, dtype=torch.float32)
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 12, 8, alpha=8.0, seed=9)}, dtype=torch.float32)
    specs = [(a, 0.5), (b, 0.5)]
    expected = 0.5 * delta_of(load_file(str(a)), P1) + 0.5 * delta_of(load_file(str(b)), P1)

    # default target rank is the largest input rank for that layer
    default_rank, _ = soup_to_dict(tmp_path, specs)
    assert rank_of(default_rank, P1) == 8

    # ...and at the full rank of the sum, the merge is exact
    exact, _ = soup_to_dict(tmp_path, specs, rank=12)
    assert rel_err(delta_of(exact, P1), expected) < FP32_REL_TOL

    by_concat, _ = soup_to_dict(tmp_path, specs, method="concat")
    assert rank_of(by_concat, P1) == 12
    assert rel_err(delta_of(by_concat, P1), expected) < FP32_REL_TOL


def test_explicit_rank_overrides_the_default(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 8, seed=10)}, dtype=torch.float32)
    merged, _ = soup_to_dict(tmp_path, [(a, 1.0)], rank=2)
    assert rank_of(merged, P1) == 2


def test_shape_disagreement_is_a_hard_error(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, seed=11)})
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 20, 4, seed=12)})

    with pytest.raises(SoupError) as excinfo:
        lora_soup.soup_files([(a, 0.5), (b, 0.5)], log=lambda _m: None)
    assert P1 in str(excinfo.value)
    assert "disagreeing shapes" in str(excinfo.value)


# --------------------------------------------------------------------------
# layer-set policy
# --------------------------------------------------------------------------

def test_absent_layer_counts_as_zero_and_is_reported(tmp_path):
    """A layer one adapter never trained contributes nothing -- but the count of
    such layers reaches the operator."""
    a = write_lora(
        tmp_path / "a.safetensors",
        {P1: make_layer(16, 12, 4, alpha=2.0, seed=13), P2: make_layer(16, 12, 4, alpha=2.0, seed=14)},
        dtype=torch.float32,
    )
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 12, 4, alpha=2.0, seed=15)}, dtype=torch.float32)

    notes = []
    state_dict, _ = lora_soup.soup_files([(a, 0.5), (b, 0.5)], rank=8, log=notes.append)

    assert P2 + ".alpha" in state_dict
    only_a = 0.5 * delta_of(load_file(str(a)), P2)
    assert rel_err(delta_of(state_dict, P2), only_a) < FP32_REL_TOL
    assert any("1 of 2 layer(s) are absent" in note for note in notes)


def test_concat_refuses_a_partial_key_set(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, seed=16), P2: make_layer(16, 12, 4, seed=17)})
    b = write_lora(tmp_path / "b.safetensors", {P1: make_layer(16, 12, 4, seed=18)})

    with pytest.raises(SoupError, match="same key set"):
        lora_soup.soup_files([(a, 0.5), (b, 0.5)], method="concat", log=lambda _m: None)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("transformer.transformer_blocks.0.attn1.to_q.hada_w1_a", "LoHa"),
        ("transformer.transformer_blocks.0.attn1.to_q.oft_R.oft_blocks", "OFT"),
        ("transformer.transformer_blocks.0.attn1.to_q.dora_scale", "DoRA"),
    ],
)
def test_non_lora_peft_files_are_refused_by_key_name(tmp_path, key, kind):
    """A file this script cannot decompose is refused, naming the key -- never
    merged approximately.

    ``lokr_*`` used to be in this list and no longer is: LoKr has a closed-form
    additive delta and merges in delta space like everything else. The LoKr
    section below is what replaced this row.
    """
    path = tmp_path / "foreign.safetensors"
    down, up, alpha = make_layer(16, 12, 4, seed=19)
    state_dict = {
        P1 + ".lora_down.weight": down,
        P1 + ".lora_up.weight": up,
        P1 + ".alpha": torch.tensor(alpha),
        key: torch.randn(4, 4),
    }
    save_file(state_dict, str(path))

    with pytest.raises(SoupError) as excinfo:
        lora_soup.load_lora(path, 1.0)
    message = str(excinfo.value)
    assert key in message
    assert kind in message


@pytest.mark.parametrize(
    ("key", "kind", "reason_fragment"),
    [
        ("transformer.transformer_blocks.0.attn1.to_q.oft_R.oft_blocks", "OFT", "multiplicative"),
        ("transformer.transformer_blocks.0.attn1.to_q.dora_scale", "DoRA", "renormalizes"),
    ],
)
def test_oft_and_dora_are_refused_for_being_non_additive(tmp_path, key, kind, reason_fragment):
    """The refusal must give the *real* reason, now that LoKr is merged here.

    "not plain LoRA" was the old reason and it distinguishes nothing any more:
    LoKr is not plain LoRA either and merges fine. What rules OFT and DoRA out is
    that neither contributes an additive delta -- OFT rotates the base weight,
    DoRA renormalizes the combined one -- so there is no dW to sum. A reader who
    takes the old wording at face value goes on to fix the wrong thing.
    """
    path = tmp_path / "foreign.safetensors"
    down, up, alpha = make_layer(16, 12, 4, seed=19)
    save_file(
        {
            P1 + ".lora_down.weight": down,
            P1 + ".lora_up.weight": up,
            P1 + ".alpha": torch.tensor(alpha),
            key: torch.randn(4, 4),
        },
        str(path),
    )

    with pytest.raises(SoupError) as excinfo:
        lora_soup.load_lora(path, 1.0)
    message = str(excinfo.value)
    assert f"looks like {kind}" in message
    assert reason_fragment in message
    # ...and it must not still be lumping LoKr in with them: LoKr appears in
    # this message only in the list of types that *are* understood.
    assert "LoKr, which" not in message
    assert "LoKr (lokr_*)" in message


def test_loha_is_refused_as_a_gap_not_an_impossibility(tmp_path):
    """LoHa's delta *is* additive and closed-form (``(W1 * W2) * alpha/rank``,
    per ``LoHaModule.forward``), so it could be carried here exactly as LoKr now
    is. It isn't, and the message has to say that rather than imply the algebra
    forbids it."""
    path = tmp_path / "loha.safetensors"
    down, up, alpha = make_layer(16, 12, 4, seed=21)
    save_file(
        {
            P1 + ".lora_down.weight": down,
            P1 + ".lora_up.weight": up,
            P1 + ".alpha": torch.tensor(alpha),
            P2 + ".hada_w1_a": torch.randn(4, 4),
        },
        str(path),
    )

    with pytest.raises(SoupError) as excinfo:
        lora_soup.load_lora(path, 1.0)
    message = str(excinfo.value)
    assert "LoHa" in message
    assert "closed-form" in message
    assert "does not, yet" in message


def test_layer_missing_a_factor_is_refused(tmp_path):
    path = tmp_path / "half.safetensors"
    down, _up, _alpha = make_layer(16, 12, 4, seed=20)
    save_file({P1 + ".lora_down.weight": down}, str(path))

    with pytest.raises(SoupError, match="lora_up.weight"):
        lora_soup.load_lora(path, 1.0)


def test_a_non_1x1_up_kernel_is_refused_and_says_why(tmp_path):
    """OneTrainer emits a 1x1 ``lora_up`` for conv layers, and ``delta()``'s
    flattening depends on it. A file whose up carries a real kernel must be
    refused -- and the message must name *that*, not report an inconsistent
    rank, which is what the shape check would otherwise conclude."""
    path = tmp_path / "kerneled.safetensors"
    save_file(
        {
            P1 + ".lora_down.weight": torch.randn(4, 6, 3, 3),
            P1 + ".lora_up.weight": torch.randn(8, 4, 3, 3),  # rank agrees; kernel doesn't
            P1 + ".alpha": torch.tensor(4.0),
        },
        str(path),
    )

    with pytest.raises(SoupError, match="non-1x1 lora_up kernel"):
        lora_soup.load_lora(path, 1.0)


# --------------------------------------------------------------------------
# bundled TI vectors
# --------------------------------------------------------------------------

def test_bundle_emb_is_carried_verbatim_from_the_anchor(tmp_path):
    """Byte-identical from the highest-coefficient input, never averaged and
    never even dtype-converted."""
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=21)}
    g = torch.Generator().manual_seed(99)
    anchor_bundle = {
        "bundle_emb.tok.qwen": torch.randn(2, 8, generator=g),
        "bundle_emb.tok.qwen_out": torch.randn(2, 8, generator=g),
        "bundle_emb.tok.t5": torch.randn(2, 6, generator=g),
    }
    other_bundle = {key: value + 1.0 for key, value in anchor_bundle.items()}

    low = write_lora(tmp_path / "low.safetensors", layers, bundle=other_bundle)
    high = write_lora(tmp_path / "high.safetensors", layers, bundle=anchor_bundle)

    merged, _ = soup_to_dict(tmp_path, [(low, 0.25), (high, 0.75)], dtype=torch.float32)

    high_read = load_file(str(high))
    low_read = load_file(str(low))
    for key in anchor_bundle:
        assert merged[key].dtype == torch.float16, "verbatim means the anchor's dtype, not the output dtype"
        assert torch.equal(merged[key], high_read[key])
        average = (high_read[key].float() + low_read[key].float()) / 2
        assert not torch.allclose(merged[key].float(), average)


# --------------------------------------------------------------------------
# block scales (455's hook)
# --------------------------------------------------------------------------

def test_block_scale_zero_removes_only_matching_layers(tmp_path):
    layers = {
        P1: make_layer(16, 12, 4, alpha=2.0, seed=22),  # attn1 -- matched
        P2: make_layer(16, 12, 4, alpha=2.0, seed=23),  # attn2 -- not matched
        P3: make_layer(16, 12, 4, alpha=2.0, seed=24),  # ff    -- not matched
    }
    a = write_lora(tmp_path / "a.safetensors", layers, dtype=torch.float32)

    plain, _ = soup_to_dict(tmp_path, [(a, 1.0)])
    scaled, _ = soup_to_dict(tmp_path, [(a, 1.0)], block_scales=[("*attn1*", 0.0)])

    assert delta_of(scaled, P1).abs().max().item() == pytest.approx(0.0, abs=1e-6)
    for prefix in (P2, P3):
        assert torch.equal(scaled[prefix + ".lora_down.weight"], plain[prefix + ".lora_down.weight"])
        assert torch.equal(scaled[prefix + ".lora_up.weight"], plain[prefix + ".lora_up.weight"])


def test_block_scale_of_one_everywhere_is_the_identity(tmp_path):
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=25), P2: make_layer(16, 12, 4, alpha=2.0, seed=26)}
    a = write_lora(tmp_path / "a.safetensors", layers, dtype=torch.float32)

    plain, _ = soup_to_dict(tmp_path, [(a, 1.0)])
    scaled, _ = soup_to_dict(tmp_path, [(a, 1.0)], block_scales=[("*attn1*", 1.0), ("*", 1.0)])

    assert set(plain) == set(scaled)
    for key in plain:
        assert torch.equal(plain[key], scaled[key]), key


def test_block_scale_applies_in_both_methods(tmp_path):
    """The scale is on the delta, so it must survive the concat path too."""
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=27)}
    a = write_lora(tmp_path / "a.safetensors", layers, dtype=torch.float32)
    reference = delta_of(load_file(str(a)), P1)

    for kwargs in ({"rank": 4}, {"method": "concat"}):
        merged, _ = soup_to_dict(tmp_path, [(a, 1.0)], block_scales=[("*attn1*", 0.25)], **kwargs)
        assert rel_err(delta_of(merged, P1), reference * 0.25) < FP32_REL_TOL, kwargs


def test_block_scale_pattern_is_a_glob_over_the_whole_prefix():
    assert lora_soup.block_scale_for(P1, [("*attn1*", 0.5)]) == 0.5
    # a bare substring is NOT a match -- the documented rule is one rule
    assert lora_soup.block_scale_for(P1, [("attn1", 0.5)]) == 1.0
    # overlapping patterns compose by multiplication
    assert lora_soup.block_scale_for(P1, [("*attn1*", 0.5), ("*blocks.0.*", 0.5)]) == 0.25


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_header_stamps_the_soup_block_and_preserves_ot_config(tmp_path):
    layers = {P1: make_layer(16, 12, 4, alpha=2.0, seed=28)}
    config = json.dumps({"lora_rank": 4, "model_type": "ANIMA"})
    a = write_lora(
        tmp_path / "a.safetensors", layers,
        header={"ot_config": config, "ot_branch": "wt/454-soup", "hash_sha256": "0xdeadbeef", "modelspec.title": "x"},
    )
    b = write_lora(tmp_path / "b.safetensors", layers, header={"ot_config": "{}", "hash_sha256": "0xfeed"})

    merged, header = soup_to_dict(tmp_path, [(b, 0.25), (a, 0.75)])

    assert header["ot_config"] == config, "the anchor's config must survive -- a warm start needs it"
    assert header["modelspec.title"] == "x"

    block = json.loads(header["soup"])
    assert block["method"] == "svd"
    assert block["anchor"] == "a.safetensors"
    assert block["target_rank"] == "max-input-rank"
    assert block["compute_dtype"] == "float32"
    assert [entry["coefficient"] for entry in block["inputs"]] == [0.25, 0.75]
    assert [entry["name"] for entry in block["inputs"]] == ["b.safetensors", "a.safetensors"]
    for entry in block["inputs"]:
        assert entry["file_sha256"].startswith("0x")
        assert entry["header_hash_sha256"] in ("0xdeadbeef", "0xfeed")

    # the anchor's model-spec hash described the anchor's tensors, not ours
    assert header["hash_sha256"] != "0xdeadbeef"
    assert header["hash_sha256"] == lora_soup.state_dict_sha256(merged)


def test_header_records_block_scales(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {P1: make_layer(16, 12, 4, seed=29)})
    _merged, header = soup_to_dict(tmp_path, [(a, 1.0)], block_scales=[("*attn1*", 0.25)])
    block = json.loads(header["soup"])
    assert block["block_scales"] == [{"pattern": "*attn1*", "coefficient": 0.25}]


# --------------------------------------------------------------------------
# CLI argument parsing
# --------------------------------------------------------------------------

def test_input_spec_keeps_a_windows_drive_letter():
    path, coefficient = lora_soup.parse_input_spec("d:/ai/tools/OneTrainer/workspace/run/save.safetensors:0.5")
    assert str(path).startswith("d:/ai")
    assert coefficient == 0.5


@pytest.mark.parametrize("spec", ["no-coefficient", "file.safetensors:abc", ":0.5"])
def test_bad_input_spec_is_refused(spec):
    with pytest.raises(SoupError):
        lora_soup.parse_input_spec(spec)


def test_bad_block_scale_spec_is_refused():
    with pytest.raises(SoupError):
        lora_soup.parse_block_scale("*attn1*")


def test_output_dtype_defaults_to_the_anchor(tmp_path):
    layers = {P1: make_layer(16, 12, 4, seed=30)}
    low = write_lora(tmp_path / "low.safetensors", layers, dtype=torch.float32)
    high = write_lora(tmp_path / "high.safetensors", layers, dtype=torch.float16)

    merged, _ = soup_to_dict(tmp_path, [(low, 0.25), (high, 0.75)])
    assert merged[P1 + ".lora_down.weight"].dtype == torch.float16

    merged, _ = soup_to_dict(tmp_path, [(low, 0.75), (high, 0.25)])
    assert merged[P1 + ".lora_down.weight"].dtype == torch.float32


# --------------------------------------------------------------------------
# LoKr: fixtures
#
# Magnitudes matter here and are not decorative. A LoKr's factors are *not* all
# the same size: LoKrModule.initialize_weights kaiming-initializes lokr_w1 and
# lokr_w2_a (bound 1/sqrt(fan_in), so O(1e-1)) and zero-initializes the last
# factor -- lokr_w2_b, or lokr_w2 in full-matrix mode -- which is then trained,
# and in a real checkpoint lands around 1e-3. torch.randn everywhere would make
# every delta ~1000x too large, which is precisely how a previous fp16 precision
# fault in this file survived its own test.
# --------------------------------------------------------------------------

def _kaiming(*shape, seed):
    """nn.init.kaiming_uniform_(a=sqrt(5)): Uniform(-b, b), b = 1/sqrt(fan_in)."""
    g = torch.Generator().manual_seed(seed)
    fan_in = shape[1] * math.prod(shape[2:]) if len(shape) > 1 else shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    return (torch.rand(*shape, generator=g) * 2 - 1) * bound


def _trained(*shape, seed, scale=1e-3):
    """The factor that starts at zero and is learned -- ~1e-3, not ~1."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def make_lokr(out_l, out_k, in_m, in_n, dim, *, kernel=None, w1="whole", w2="factored", seed=0):
    """One synthetic LoKr factor set, in the shapes LoKrModule stores.

    Returns ``(parts, shape)``: the tensors keyed by their short names, and the
    weight shape ``get_weight()`` would ``.view`` to.
    """
    parts = {}
    if w1 == "whole":
        parts["lokr_w1"] = _kaiming(out_l, in_m, seed=seed)
    else:
        parts["lokr_w1_a"] = _kaiming(out_l, dim, seed=seed + 1)
        parts["lokr_w1_b"] = _kaiming(dim, in_m, seed=seed + 2)

    if w2 == "whole":
        parts["lokr_w2"] = _trained(out_k, in_n, *(kernel or ()), seed=seed + 3)
    elif w2 == "tucker":
        assert kernel is not None, "Tucker only arises for a Conv2d with a real kernel"
        parts["lokr_t2"] = _kaiming(dim, dim, *kernel, seed=seed + 4)
        parts["lokr_w2_a"] = _kaiming(dim, out_k, seed=seed + 5)
        parts["lokr_w2_b"] = _trained(dim, in_n, seed=seed + 6)
    else:
        parts["lokr_w2_a"] = _kaiming(out_k, dim, seed=seed + 7)
        # For a Conv2d with a factored w2, lokr_w2_b folds the kernel into its
        # trailing dim -- the one case where get_weight()'s .view(shape) is not
        # a no-op, and the one case the file cannot tell from a Linear.
        parts["lokr_w2_b"] = _trained(dim, in_n * math.prod(kernel or ()), seed=seed + 8)

    shape = (out_l * out_k, in_m * in_n, *(kernel or ()))
    return parts, shape


def write_lokr(path, layers, header=None, dtype=torch.float16):
    """Serialize {prefix: (parts, alpha_or_None)} the way LoKrModule's state
    dict lands in a safetensors file."""
    state_dict = {}
    for prefix, (parts, alpha) in layers.items():
        for name, tensor in parts.items():
            state_dict[f"{prefix}.{name}"] = tensor.to(dtype).contiguous().clone()
        if alpha is not None:
            state_dict[prefix + ".alpha"] = torch.tensor(alpha, dtype=dtype)
    save_file(state_dict, str(path), header)
    return path


def lokr_delta_of(state_dict, prefix, dim, shape):
    """``make_kron(w1, w2).view(shape) * (alpha/dim)``, computed independently
    of the module under test.

    ``dim`` is passed in from how the fixture was *built*. Re-deriving it here
    would only re-run the loader's own inference and pin nothing -- the whole
    point is that the file does not store it.
    """
    def get(name):
        return state_dict.get(f"{prefix}.{name}")

    w1 = get("lokr_w1")
    w1 = w1.float() if w1 is not None else get("lokr_w1_a").float() @ get("lokr_w1_b").float()

    w2 = get("lokr_w2")
    if w2 is not None:
        w2 = w2.float()
    elif get("lokr_t2") is not None:
        w2 = rebuild_tucker(get("lokr_t2").float(), get("lokr_w2_a").float(), get("lokr_w2_b").float())
    else:
        w2 = get("lokr_w2_a").float() @ get("lokr_w2_b").float()

    alpha = float(get("alpha").item()) if get("alpha") is not None else float(dim)
    return make_kron(w1, w2).view(shape) * (alpha / dim)


# The two Linear factorizations used throughout. factorization(256) == (16, 16)
# and factorization(128) == factorization(96)-ish shapes are what LoKrModule
# would pick; the tests pass them explicitly so the fixture never depends on it.
LIN_256 = {"out_l": 16, "out_k": 16, "in_m": 16, "in_n": 16}


# --------------------------------------------------------------------------
# LoKr: the delta is exact, in all three w2 forms
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("w1_form", "w2_form", "kernel", "geom"),
    [
        ("whole", "factored", None, LIN_256),
        ("whole", "whole", None, LIN_256),
        ("pair", "factored", None, LIN_256),
        ("pair", "whole", None, LIN_256),
        # Conv2d 64 -> 96: factorization(96) = (8, 12), factorization(64) = (8, 8).
        ("whole", "whole", (3, 3), {"out_l": 8, "out_k": 12, "in_m": 8, "in_n": 8}),
        ("whole", "tucker", (3, 3), {"out_l": 8, "out_k": 12, "in_m": 8, "in_n": 8}),
    ],
)
def test_lokr_delta_is_the_closed_form_kronecker_product(tmp_path, w1_form, w2_form, kernel, geom):
    """dW = make_kron(w1, w2).view(shape) * (alpha/dim), exactly -- no
    approximation anywhere on the LoKr path.

    Every combination of w1 form (whole / decompose_both pair) and w2 form
    (whole / factored / Tucker) is covered, because each picks a different
    branch of _get_factors and a different place to read lokr_dim back from.
    """
    dim = 4
    parts, shape = make_lokr(**geom, dim=dim, kernel=kernel, w1=w1_form, w2=w2_form, seed=100)
    path = write_lokr(tmp_path / "lokr.safetensors", {P1: (parts, float(dim))})

    loaded = lora_soup.load_lora(path, 1.0)
    layer = loaded.layers[P1]
    assert isinstance(layer, lora_soup.LokrLayer)

    stored = load_file(str(path))
    expected = lokr_delta_of(stored, P1, dim, shape)

    assert layer.weight_shape() == shape
    assert rel_err(layer.weight(), expected) < 1e-6
    # ...and the 2-D form the merge actually consumes is the same numbers.
    assert rel_err(layer.delta(), expected.reshape(shape[0], -1)) < 1e-6


def test_lokr_delta_matches_a_hand_written_kronecker(tmp_path):
    """One case checked against the textbook definition rather than make_kron,
    so the suite is not purely self-referential about the index convention:
    kron(w1, w2)[i*out_k + k, j*in_n + n] == w1[i, j] * w2[k, n]."""
    dim = 4
    parts, shape = make_lokr(**LIN_256, dim=dim, seed=200)
    path = write_lokr(tmp_path / "lokr.safetensors", {P1: (parts, float(dim))})

    layer = lora_soup.load_lora(path, 1.0).layers[P1]
    w1 = parts["lokr_w1"].to(torch.float16).float()
    w2 = (parts["lokr_w2_a"].to(torch.float16).float() @ parts["lokr_w2_b"].to(torch.float16).float())

    out_k, in_n = w2.shape
    expected = torch.empty(shape)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            expected[i * out_k:(i + 1) * out_k, j * in_n:(j + 1) * in_n] = w1[i, j] * w2

    assert rel_err(layer.weight(), expected) < 1e-6


def test_lokr_scale_is_alpha_over_dim_not_alpha_over_anything_else(tmp_path):
    """dim is the *factor* rank, read back off the factor shapes -- the file
    never stores it. Doubling only alpha must double the delta exactly."""
    dim = 4
    parts, shape = make_lokr(**LIN_256, dim=dim, seed=300)
    at_one = write_lokr(tmp_path / "one.safetensors", {P1: (parts, float(dim))})
    at_two = write_lokr(tmp_path / "two.safetensors", {P1: (parts, float(2 * dim))})

    d1 = lora_soup.load_lora(at_one, 1.0).layers[P1].delta()
    d2 = lora_soup.load_lora(at_two, 1.0).layers[P1].delta()

    assert lora_soup.load_lora(at_one, 1.0).layers[P1].dim == dim
    assert rel_err(d2, 2.0 * d1) < 1e-6
    # And an absent alpha means alpha == dim, i.e. scale 1.0, as for LoRA.
    bare = write_lokr(tmp_path / "bare.safetensors", {P1: (parts, None)})
    assert rel_err(lora_soup.load_lora(bare, 1.0).layers[P1].delta(), d1) < 1e-6
    assert shape == (256, 256)


def test_lokr_with_both_factors_whole_reads_a_scale_of_one(tmp_path):
    """With no inner dim anywhere, lokr_dim is unrecoverable -- and irrelevant,
    because LoKrModule.initialize_weights does alpha.fill_(lokr_dim) in exactly
    that case, so alpha/dim is 1.0. The stored alpha must therefore be ignored,
    not divided by 1."""
    parts, shape = make_lokr(**LIN_256, dim=8, w1="whole", w2="whole", seed=400)
    path = write_lokr(tmp_path / "full.safetensors", {P1: (parts, 8.0)})

    layer = lora_soup.load_lora(path, 1.0).layers[P1]
    assert layer.scale == 1.0

    stored = load_file(str(path))
    expected = make_kron(stored[P1 + ".lokr_w1"].float(), stored[P1 + ".lokr_w2"].float()).view(shape)
    assert rel_err(layer.weight(), expected) < 1e-6


def test_lokr_conv_with_a_4d_w2_keeps_its_kernel_geometry(tmp_path):
    """A Conv2d LoKr whose w2 is stored whole (or Tucker) is 4-D, so the kernel
    is visible in the file and the emitted geometry matches what a conv LoRA
    over the same layer would report: (out, (in, k1, k2), (1, 1))."""
    parts, _shape = make_lokr(
        out_l=8, out_k=12, in_m=8, in_n=8, dim=4, kernel=(3, 3), w2="whole", seed=500
    )
    path = write_lokr(tmp_path / "conv.safetensors", {P1: (parts, 4.0)})

    layer = lora_soup.load_lora(path, 1.0).layers[P1]
    assert layer.geometry() == (96, (64, 3, 3), (1, 1))


def test_lokr_conv_with_a_factored_w2_reads_as_linear(tmp_path):
    """The one shape ambiguity, pinned so it is a known reading rather than a
    surprise: a Conv2d whose w2 is factored folds the kernel into lokr_w2_b, and
    nothing in the file distinguishes that from a Linear of in = in*k1*k2. The
    delta is exact either way -- only the emitted factor shape differs."""
    parts, shape = make_lokr(
        out_l=8, out_k=12, in_m=8, in_n=8, dim=2, kernel=(3, 3), w2="factored", seed=600
    )
    path = write_lokr(tmp_path / "conv.safetensors", {P1: (parts, 2.0)})

    layer = lora_soup.load_lora(path, 1.0).layers[P1]
    assert shape == (96, 64, 3, 3)  # what it really was
    assert layer.geometry() == (96, (576,), ())  # what the file can say
    # The numbers are still right: viewing the read-as-linear delta back to the
    # conv shape reproduces the closed form.
    stored = load_file(str(path))
    assert rel_err(layer.delta().reshape(shape), lokr_delta_of(stored, P1, 2, shape)) < 1e-6


# --------------------------------------------------------------------------
# LoKr: merging is linear in delta space
# --------------------------------------------------------------------------

def test_two_lokrs_merge_as_c1_dw1_plus_c2_dw2(tmp_path):
    """The whole reason LoKr belongs in this engine: it reduces to a dW, and
    dWs add. Checked against deltas computed outside the module."""
    dim = 4
    parts_a, shape = make_lokr(**LIN_256, dim=dim, seed=700)
    parts_b, _ = make_lokr(**LIN_256, dim=dim, seed=800)
    a = write_lokr(tmp_path / "a.safetensors", {P1: (parts_a, float(dim))})
    b = write_lokr(tmp_path / "b.safetensors", {P1: (parts_b, float(2 * dim))})

    c1, c2 = 0.3, 0.7
    merged, _report = lora_soup.merge_deltas(
        [lora_soup.load_lora(a, c1), lora_soup.load_lora(b, c2)], log=lambda _m: None
    )

    expected = (
        c1 * lokr_delta_of(load_file(str(a)), P1, dim, shape)
        + c2 * lokr_delta_of(load_file(str(b)), P1, dim, shape)
    )
    assert rel_err(merged[P1].delta, expected.reshape(shape[0], -1)) < FP32_REL_TOL


def test_a_lora_and_a_lokr_merge_together(tmp_path):
    """Mixed soups work, because the merge never learns which was which. Both
    sides describe the same layer, both reduce to a dW over the same geometry,
    and the sum is the sum."""
    dim = 4
    lokr_parts, shape = make_lokr(**LIN_256, dim=dim, seed=900)
    lokr_path = write_lokr(tmp_path / "lokr.safetensors", {P1: (lokr_parts, float(dim))})
    # A plain LoRA over the same 256x256 layer, at a comparable magnitude.
    down = _kaiming(8, 256, seed=901)
    up = _trained(256, 8, seed=902)
    lora_path = write_lora(tmp_path / "lora.safetensors", {P1: (down, up, 8.0)})

    c1, c2 = 0.4, 0.6
    merged, report = lora_soup.merge_deltas(
        [lora_soup.load_lora(lokr_path, c1), lora_soup.load_lora(lora_path, c2)],
        log=lambda _m: None,
    )
    assert report.partial_layers == 0

    expected = (
        c1 * lokr_delta_of(load_file(str(lokr_path)), P1, dim, shape).reshape(256, -1)
        + c2 * delta_of(load_file(str(lora_path)), P1)
    )
    assert rel_err(merged[P1].delta, expected) < FP32_REL_TOL

    # ...and it comes out the far end as a loadable plain LoRA.
    state_dict, _header = soup_to_dict(tmp_path, [(lokr_path, c1), (lora_path, c2)], rank=72)
    assert rel_err(delta_of(state_dict, P1), expected) < FP16_REL_TOL


def test_lokr_svd_output_round_trips_the_merged_delta(tmp_path):
    """At a rank sufficient to represent it, the emitted plain LoRA reproduces
    the merged LoKr delta to fp16 storage tolerance -- nothing looser.

    "Sufficient" is 128, not 64. Each input's delta has rank
    rank(w1)*rank(w2) = 16*4 = 64, and the sum of two rank-64 matrices has rank
    up to 128. The per-layer default (max over inputs, so 64 here) is therefore
    a *truncating* default for a multi-input soup -- exactly as it already is
    for plain LoRA, where two rank-4 inputs also sum to rank 8. LoKr does not
    change that trade-off, and this test asks for the rank that avoids it rather
    than loosening the tolerance until 64 passes.
    """
    dim = 4
    parts_a, shape = make_lokr(**LIN_256, dim=dim, seed=1000)
    parts_b, _ = make_lokr(**LIN_256, dim=dim, seed=1100)
    a = write_lokr(tmp_path / "a.safetensors", {P1: (parts_a, float(dim))})
    b = write_lokr(tmp_path / "b.safetensors", {P1: (parts_b, float(dim))})

    expected = (
        0.5 * lokr_delta_of(load_file(str(a)), P1, dim, shape)
        + 0.5 * lokr_delta_of(load_file(str(b)), P1, dim, shape)
    ).reshape(shape[0], -1)

    state_dict, _header = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)], rank=128)
    assert rank_of(state_dict, P1) == 128
    assert float(state_dict[P1 + ".alpha"].item()) == 128.0
    assert rel_err(delta_of(state_dict, P1), expected) < FP16_REL_TOL

    # And the default (64) is a real truncation, not a silent equality: if this
    # ever stops losing anything, the rank bound above has drifted.
    truncated, _header = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)])
    assert rank_of(truncated, P1) == 64
    assert rel_err(delta_of(truncated, P1), expected) > FP16_REL_TOL


def test_lokr_default_rank_is_the_kronecker_bound_not_lokr_dim(tmp_path):
    """rank(kron(w1, w2)) = rank(w1) * rank(w2), so a dim-4 LoKr over a 256x256
    Linear carries a delta of rank up to 64. Defaulting the SVD to lokr_dim
    would silently discard 94% of it -- the merge would 'succeed' and be
    wrong."""
    dim = 4
    parts, shape = make_lokr(**LIN_256, dim=dim, seed=1200)
    path = write_lokr(tmp_path / "lokr.safetensors", {P1: (parts, float(dim))})

    assert lora_soup.load_lora(path, 1.0).layers[P1].rank == 64

    state_dict, _header = soup_to_dict(tmp_path, [(path, 1.0)])
    assert rank_of(state_dict, P1) == 64
    expected = lokr_delta_of(load_file(str(path)), P1, dim, shape).reshape(shape[0], -1)
    assert rel_err(delta_of(state_dict, P1), expected) < FP16_REL_TOL


def test_lokr_block_scale_applies_in_delta_space(tmp_path):
    """455's ablation works on a LoKr for the same reason a soup does: the block
    scale multiplies a delta, and a LoKr has one."""
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=1300)
    path = write_lokr(tmp_path / "lokr.safetensors", {P1: (parts, float(dim)), P2: (parts, float(dim))})

    merged, _report = lora_soup.merge_deltas(
        [lora_soup.load_lora(path, 1.0)], block_scales=[("*attn1*", 0.0)], log=lambda _m: None
    )
    assert merged[P1].delta.abs().max().item() == 0.0
    assert merged[P2].delta.abs().max().item() > 0.0


# --------------------------------------------------------------------------
# LoKr: refusals
# --------------------------------------------------------------------------

def test_concat_refuses_a_lokr_input_with_its_own_reason(tmp_path):
    """--method concat is an exact LoRA-only rearrangement: it stacks blocks
    along a shared rank axis, which a Kronecker delta does not have. Silently
    falling back to SVD would hand back an approximation under a flag that
    promises exactness, so this is refused by name -- and distinguishably from
    the partial-key-set refusal, which is a different problem."""
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=1400)
    lokr_path = write_lokr(tmp_path / "lokr.safetensors", {P1: (parts, float(dim))})
    down = _kaiming(8, 256, seed=1401)
    up = _trained(256, 8, seed=1402)
    lora_path = write_lora(tmp_path / "lora.safetensors", {P1: (down, up, 8.0)})

    with pytest.raises(SoupError) as excinfo:
        lora_soup.soup_files([(lokr_path, 0.5), (lora_path, 0.5)], method="concat", log=lambda _m: None)
    message = str(excinfo.value)
    assert "Kronecker" in message
    assert "--method svd" in message
    assert "lokr.safetensors" in message
    # Not the "same key set" refusal, which is about absent layers.
    assert "same key set" not in message


def test_concat_still_works_for_an_all_lora_soup(tmp_path):
    """The LoKr refusal must not have become a refusal of concat in general."""
    layers = {P1: make_layer(16, 12, 4, seed=1500)}
    a = write_lora(tmp_path / "a.safetensors", layers)
    b = write_lora(tmp_path / "b.safetensors", layers)
    state_dict, _header = soup_to_dict(tmp_path, [(a, 0.5), (b, 0.5)], method="concat")
    assert rank_of(state_dict, P1) == 8


def test_lokr_missing_a_w2_factor_is_refused(tmp_path):
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=1600)
    del parts["lokr_w2_b"]
    path = write_lokr(tmp_path / "half.safetensors", {P1: (parts, float(dim))})

    with pytest.raises(SoupError, match="lokr_w2_b"):
        lora_soup.load_lora(path, 1.0)


def test_lokr_with_two_w2_forms_at_once_is_refused(tmp_path):
    """Whole / factored / Tucker are mutually exclusive by construction. A file
    with two of them is corrupt, not ambiguous, so it is refused rather than
    resolved by precedence."""
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=1700)
    parts["lokr_w2"] = _trained(16, 16, seed=1701)
    path = write_lokr(tmp_path / "both.safetensors", {P1: (parts, float(dim))})

    with pytest.raises(SoupError, match="only one w2 form"):
        lora_soup.load_lora(path, 1.0)


def test_lokr_disagreeing_about_its_own_dim_is_refused(tmp_path):
    """Every factor that carries lokr_dim must say the same number. It is the
    denominator of the scale; guessing which one is right would mis-scale the
    whole layer silently."""
    parts, _shape = make_lokr(**LIN_256, dim=4, w1="pair", seed=1800)
    parts["lokr_w2_a"] = _kaiming(16, 6, seed=1801)  # says dim = 6
    parts["lokr_w2_b"] = _trained(6, 16, seed=1802)
    path = write_lokr(tmp_path / "confused.safetensors", {P1: (parts, 4.0)})

    with pytest.raises(SoupError, match="disagrees with itself about lokr_dim"):
        lora_soup.load_lora(path, 1.0)


def test_a_layer_carrying_both_lora_and_lokr_factors_is_refused(tmp_path):
    """One layer has one delta. A file claiming two for the same prefix is not
    a merge problem, it is a corrupt file."""
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=1900)
    path = tmp_path / "both.safetensors"
    state_dict = {f"{P1}.{name}": tensor.contiguous().clone() for name, tensor in parts.items()}
    state_dict[P1 + ".lora_down.weight"] = _kaiming(4, 256, seed=1901)
    state_dict[P1 + ".lora_up.weight"] = _trained(256, 4, seed=1902)
    state_dict[P1 + ".alpha"] = torch.tensor(float(dim))
    save_file(state_dict, str(path))

    with pytest.raises(SoupError, match="both plain-LoRA and LoKr"):
        lora_soup.load_lora(path, 1.0)


def test_a_lokr_file_reports_its_storage_dtype(tmp_path):
    """The output inherits the anchor's dtype, and a LoKr anchor has no
    lora_down to read it off."""
    dim = 4
    parts, _shape = make_lokr(**LIN_256, dim=dim, seed=2000)
    path = write_lokr(tmp_path / "bf16.safetensors", {P1: (parts, float(dim))}, dtype=torch.bfloat16)
    assert lora_soup.load_lora(path, 1.0).dtype == torch.bfloat16
