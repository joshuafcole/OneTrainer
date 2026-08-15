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
        ("transformer.transformer_blocks.0.attn1.to_q.lokr_w1", "LoKr"),
        ("transformer.transformer_blocks.0.attn1.to_q.oft_R.oft_blocks", "OFT"),
        ("transformer.transformer_blocks.0.attn1.to_q.dora_scale", "DoRA"),
    ],
)
def test_non_lora_peft_files_are_refused_by_key_name(tmp_path, key, kind):
    """A file this script cannot decompose is refused, naming the key -- never
    merged approximately."""
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


def test_layer_missing_a_factor_is_refused(tmp_path):
    path = tmp_path / "half.safetensors"
    down, _up, _alpha = make_layer(16, 12, 4, seed=20)
    save_file({P1 + ".lora_down.weight": down}, str(path))

    with pytest.raises(SoupError, match="lora_up.weight"):
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
