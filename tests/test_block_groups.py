"""Tests for scripts/util/block_groups.py -- the 455 block-group taxonomy.

Pure safetensors + pytest, no torch model code and no real checkpoint: every
key set is synthetic, built here in the real anima key form
(``transformer.transformer_blocks.<i>.<part>...``, see ``AnimaLoRASaver``).
These pin the *decisions* the taxonomy makes -- exact-once coverage, the
band-rounding rule, the attn/norm split -- rather than incidental structure.
Run with::

    python -m pytest tests/test_block_groups.py -q
"""

import importlib.util
import os
import sys

import pytest

_here = os.path.dirname(__file__)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, relpath))
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


block_groups = _load("block_groups", "../scripts/util/block_groups.py")
lora_soup = _load("lora_soup", "../scripts/util/lora_soup.py")

BlockGroupError = block_groups.BlockGroupError

BLOCK_PREFIX = "transformer.transformer_blocks"

# Every real LoRA-decomposable leaf a CosmosTransformerBlock offers, and the
# part each one belongs to -- reproduced from BaseAnimaSetup.LAYER_PRESETS'
# comment plus diffusers' CosmosTransformerBlock/Attention/FeedForward naming.
LEAVES_PER_BLOCK = {
    "attn-self": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"],
    "attn-cross": ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
    "mlp": ["ff.net.0.proj", "ff.net.2"],
    "modulation": [
        "norm1.linear_1", "norm1.linear_2",
        "norm2.linear_1", "norm2.linear_2",
        "norm3.linear_1", "norm3.linear_2",
    ],
}
LEAVES_PER_BLOCK_COUNT = sum(len(v) for v in LEAVES_PER_BLOCK.values())  # 16


def block_leaves(index: int, block_prefix: str = BLOCK_PREFIX) -> dict[str, list[str]]:
    """{part_name: [full layer prefixes]} for one block index."""
    base = f"{block_prefix}.{index}"
    return {part: [f"{base}.{leaf}" for leaf in leaves] for part, leaves in LEAVES_PER_BLOCK.items()}


def synthetic_prefixes(block_count: int, block_prefix: str = BLOCK_PREFIX) -> list[str]:
    """Every block index x every part's leaves, in the real anima key form."""
    prefixes: list[str] = []
    for i in range(block_count):
        for leaves in block_leaves(i, block_prefix).values():
            prefixes.extend(leaves)
    return prefixes


@pytest.fixture(scope="module")
def config():
    return block_groups.load_groups()


# --------------------------------------------------------------------------
# exact-once coverage
# --------------------------------------------------------------------------

def test_exact_once_coverage_over_a_realistic_key_set(config):
    """The phase doc's first Done-when: every adapted layer covered exactly
    once, on a plausible block count."""
    prefixes = synthetic_prefixes(28)
    report = block_groups.coverage(prefixes, config)

    assert report.total == 28 * LEAVES_PER_BLOCK_COUNT
    assert report.covered == report.total
    assert report.unassigned == ()
    assert report.multiply_assigned == {}
    assert report.exact_once


def test_group_layers_reports_every_group_including_empty_ones(config):
    """The cross product is the full roster regardless of what's observed --
    a group with no members is a reportable zero, not an absent key."""
    prefixes = synthetic_prefixes(28)
    groups = block_groups.group_layers(prefixes, config)

    assert set(groups) == set(block_groups.all_group_names(config))
    assert len(groups) == 3 * 4  # 3 bands x 4 parts
    # every group here (28 blocks, all parts populated) has members
    assert all(members for members in groups.values())
    assert sum(len(members) for members in groups.values()) == 28 * LEAVES_PER_BLOCK_COUNT


# --------------------------------------------------------------------------
# band boundaries
# --------------------------------------------------------------------------

def _expected_bands(band_sizes: list[int]) -> dict[int, int]:
    """{index: band} from consecutive band sizes, e.g. [9, 9, 11] -> indices
    0-8 -> band 0, 9-17 -> band 1, 18-28 -> band 2."""
    expected: dict[int, int] = {}
    index = 0
    for band, size in enumerate(band_sizes):
        for _ in range(size):
            expected[index] = band
            index += 1
    return expected


def test_band_boundaries_when_block_count_divides_evenly():
    """27 blocks / 3 bands: no remainder, three equal bands of 9."""
    for index, band in _expected_bands([9, 9, 9]).items():
        assert block_groups.band_for_index(index, block_count=27, band_count=3) == band, f"index {index}"


def test_band_boundaries_when_block_count_does_not_divide_29():
    """29 / 3: sizes [9, 9, 11] -- the last band absorbs the remainder."""
    for index, band in _expected_bands([9, 9, 11]).items():
        assert block_groups.band_for_index(index, block_count=29, band_count=3) == band, f"index {index}"
    # no index falls off the end
    assert all(block_groups.band_for_index(i, 29, 3) in (0, 1, 2) for i in range(29))


def test_band_boundaries_when_block_count_does_not_divide_28():
    """28 / 3: sizes [9, 9, 10]."""
    for index, band in _expected_bands([9, 9, 10]).items():
        assert block_groups.band_for_index(index, block_count=28, band_count=3) == band, f"index {index}"


def test_band_boundaries_reflected_in_group_layers(config):
    """The same boundaries, seen through group membership counts rather than
    band_for_index directly -- catches a resolver that disagrees with its own
    band function."""
    prefixes = synthetic_prefixes(29)
    groups = block_groups.group_layers(prefixes, config)

    # attn-self has 4 leaves/block, so band sizes scale by 4.
    assert len(groups["early.attn-self"]) == 9 * 4
    assert len(groups["mid.attn-self"]) == 9 * 4
    assert len(groups["late.attn-self"]) == 11 * 4


# --------------------------------------------------------------------------
# fewer blocks than bands
# --------------------------------------------------------------------------

def test_single_block_checkpoint_still_assigns_totally(config):
    """1 block, 3 bands: base=0, so every index lands in the last band --
    still exact-once, nothing lost."""
    prefixes = synthetic_prefixes(1)
    report = block_groups.coverage(prefixes, config)
    groups = block_groups.group_layers(prefixes, config)

    assert report.exact_once
    assert report.covered == report.total == LEAVES_PER_BLOCK_COUNT
    assert groups["early.attn-self"] == []
    assert groups["mid.attn-self"] == []
    assert len(groups["late.attn-self"]) == 4


def test_two_block_checkpoint_still_assigns_totally(config):
    """2 blocks, 3 bands: same degenerate case, two blocks deep."""
    prefixes = synthetic_prefixes(2)
    report = block_groups.coverage(prefixes, config)
    groups = block_groups.group_layers(prefixes, config)

    assert report.exact_once
    assert report.covered == report.total == 2 * LEAVES_PER_BLOCK_COUNT
    assert groups["early.mlp"] == []
    assert groups["mid.mlp"] == []
    assert len(groups["late.mlp"]) == 2 * 2


# --------------------------------------------------------------------------
# patterns_for, cross-checked against lora_soup's real matcher
# --------------------------------------------------------------------------

def test_patterns_for_select_exactly_that_groups_members_and_no_others(config):
    """Every group's patterns, run through lora_soup's real fnmatch matcher,
    must select that group's members and nothing else -- including the
    attn1.norm_q/norm_k distractors, which sit right next to attn-self/
    attn-cross's real members and share their block index, so a glob that
    over-matches (e.g. 'attn1.*' instead of 'attn1.to_*') would only be
    caught by having a non-member with the same block+attn-number present.
    """
    block_count = 29
    prefixes = synthetic_prefixes(block_count)
    norm_distractors = [
        f"{BLOCK_PREFIX}.{i}.attn{n}.norm_{qk}" for i in range(block_count) for n in (1, 2) for qk in ("q", "k")
    ]
    all_prefixes = prefixes + norm_distractors
    groups = block_groups.group_layers(prefixes, config)

    for group_name, members in groups.items():
        patterns = block_groups.patterns_for(group_name, prefixes, config)
        block_scales = [(pattern, 0.0) for pattern in patterns]
        member_set = set(members)
        for prefix in all_prefixes:
            scale = lora_soup.block_scale_for(prefix, block_scales)
            if prefix in member_set:
                assert scale == 0.0, f"{group_name}: {prefix} should be selected by {patterns}"
            else:
                assert scale == 1.0, f"{group_name}: {prefix} should NOT be selected by {patterns}"


def test_patterns_for_enumerates_per_block_index_not_a_single_band_wildcard(config):
    """One pattern per block in the band, not one pattern standing in for the
    whole band -- a band name never appears literally in a real key, so a
    single '*early*'-shaped glob would select nothing."""
    prefixes = synthetic_prefixes(29)
    patterns = block_groups.patterns_for("early.mlp", prefixes, config)

    assert len(patterns) == 9  # early band is blocks 0-8 at 29/3
    assert patterns == sorted(patterns, key=lambda p: int(p.split(".")[2]))
    for pattern in patterns:
        assert "early" not in pattern


# --------------------------------------------------------------------------
# strays and the attn/norm exclusion
# --------------------------------------------------------------------------

def test_stray_prefix_outside_transformer_blocks_is_unassigned(config):
    prefixes = synthetic_prefixes(4) + ["text_encoder.embed_tokens"]
    assert block_groups.assign("text_encoder.embed_tokens", config, block_count=4) is None

    report = block_groups.coverage(prefixes, config)
    assert "text_encoder.embed_tokens" in report.unassigned
    assert not report.exact_once
    assert report.covered == 4 * LEAVES_PER_BLOCK_COUNT


def test_attn_norm_keys_are_excluded_from_the_attention_groups(config):
    """The subtle one: LAYER_PRESETS' attention regex is
    '^(?=.*attn)(?!.*norm).*' -- attn proper excludes anything with 'norm' in
    it. attn1.norm_q / attn1.norm_k never appear in a real adapted key set
    (QK-norm isn't LoRA-decomposable), but the taxonomy must still refuse
    them rather than silently filing them under attn-self.
    """
    block_count = 4
    prefixes = synthetic_prefixes(block_count) + [
        f"{BLOCK_PREFIX}.2.attn1.norm_q",
        f"{BLOCK_PREFIX}.2.attn1.norm_k",
        f"{BLOCK_PREFIX}.2.attn2.norm_q",
    ]

    # the real attn1 sibling at the same block/index IS attn-self:
    assert block_groups.assign(f"{BLOCK_PREFIX}.2.attn1.to_q", config, block_count) is not None

    # the norm sublayers are not:
    assert block_groups.assign(f"{BLOCK_PREFIX}.2.attn1.norm_q", config, block_count) is None
    assert block_groups.assign(f"{BLOCK_PREFIX}.2.attn1.norm_k", config, block_count) is None
    assert block_groups.assign(f"{BLOCK_PREFIX}.2.attn2.norm_q", config, block_count) is None

    report = block_groups.coverage(prefixes, config)
    assert f"{BLOCK_PREFIX}.2.attn1.norm_q" in report.unassigned
    assert f"{BLOCK_PREFIX}.2.attn1.norm_k" in report.unassigned
    assert f"{BLOCK_PREFIX}.2.attn2.norm_q" in report.unassigned

    # and they must not have silently inflated attn-self/attn-cross either
    groups = block_groups.group_layers(prefixes, config)
    assert f"{BLOCK_PREFIX}.2.attn1.norm_q" not in groups["mid.attn-self"]
    assert f"{BLOCK_PREFIX}.2.attn2.norm_q" not in groups["mid.attn-cross"]


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------

def test_assign_returns_none_for_an_index_beyond_block_count(config):
    """block_count is normally self-consistent (group_layers/coverage derive
    it from the very prefixes they resolve), so this path only bites a direct
    caller passing a stale/mismatched block_count. It must still return None
    per the documented contract, not raise -- assign() is supposed to be the
    total function that never throws on a merely-out-of-scope prefix."""
    assert block_groups.assign(f"{BLOCK_PREFIX}.10.attn1.to_q", config, block_count=5) is None


def test_load_groups_defaults_to_the_file_beside_the_module():
    config = block_groups.load_groups()
    assert config.model
    assert config.band_count == 3
    assert config.band_names == ("early", "mid", "late")
    assert {part.name for part in config.parts} == {"attn-self", "attn-cross", "mlp", "modulation"}


def test_patterns_for_rejects_an_unknown_group_name(config):
    with pytest.raises(BlockGroupError):
        block_groups.patterns_for("nonexistent.part", synthetic_prefixes(3), config)
    with pytest.raises(BlockGroupError):
        block_groups.patterns_for("early.nonexistent", synthetic_prefixes(3), config)
