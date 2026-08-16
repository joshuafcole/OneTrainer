"""Tests for scripts/util/block_groups.py — groups fitted to the layers present.

Pure safetensors + pytest, no torch model code and no real checkpoint: every key
set is synthetic, built here in the real anima key form
(``transformer.transformer_blocks.<i>.<part>...``, see ``AnimaLoRASaver``).

What these pin is the inversion the module is built on — groups are derived from
the prefixes, so **totality is structural** — plus the two things that remain
genuinely fallible: the naming layer's gaps, and whether a group's patterns
select exactly its members when run through the real matcher.

Run with::

    python -m pytest tests/test_block_groups.py -q
"""

import importlib.util
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

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

# Every real LoRA-decomposable leaf a CosmosTransformerBlock offers — from
# BaseAnimaSetup.LAYER_PRESETS plus diffusers' CosmosTransformerBlock naming.
ATTN_LEAVES = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
]
MLP_LEAVES = ["ff.net.0.proj", "ff.net.2"]
MOD_LEAVES = [f"norm{n}.linear_{k}" for n in (1, 2, 3) for k in (1, 2)]


@pytest.fixture
def config():
    return block_groups.load_groups()


def prefixes_for(leaves, block_count=28):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(block_count) for leaf in leaves]


# --- totality is structural --------------------------------------------------

@pytest.mark.parametrize(
    "label,leaves",
    [
        # The key set that motivated the rewrite: production trains attn-only,
        # and an exhaustive taxonomy left six of twelve groups empty.
        ("attn-only", ATTN_LEAVES),
        ("full", ATTN_LEAVES + MLP_LEAVES + MOD_LEAVES),
        ("mlp-only", MLP_LEAVES),
        # An architecture the naming layer has never seen.
        ("unknown-arch", ["mixer.w_in", "mixer.w_out", "gate.proj"]),
    ],
)
@pytest.mark.parametrize("granularity", ["coarse", "fine"])
def test_fit_covers_every_layer_exactly_once(config, label, leaves, granularity):
    """Coverage cannot fail, because the groups are built from the prefixes.

    Asserted over four very different key sets and both granularities rather
    than argued: the claim is about the algorithm, so an architecture the config
    knows nothing about has to hold it too."""
    prefixes = prefixes_for(leaves)
    fitted = block_groups.fit(prefixes, config, granularity)

    seen = [p for members in fitted.groups.values() for p in members]
    assert sorted(seen) == sorted(prefixes), label
    assert len(seen) == len(set(seen)), f"{label}: a layer landed in two groups"
    assert fitted.total_layers == len(prefixes)


@pytest.mark.parametrize("granularity", ["coarse", "fine"])
def test_an_attention_only_adapter_yields_no_empty_groups(config, granularity):
    """The measurement this module exists to answer.

    Under the old fixed taxonomy this key set produced 6 populated groups and 6
    empty ones — and an ablation grid would have rendered all twelve."""
    fitted = block_groups.fit(prefixes_for(ATTN_LEAVES), config, granularity)
    assert fitted.groups, "fit produced no groups at all"
    assert all(members for members in fitted.groups.values())


def test_fine_resolves_within_attention_where_coarse_cannot(config):
    """The point of the knob: for an attn-only adapter coarse can only say
    self-vs-cross, while fine separates q/k/v/out — 24 coordinates, not 6."""
    prefixes = prefixes_for(ATTN_LEAVES)
    coarse = block_groups.fit(prefixes, config, "coarse")
    fine = block_groups.fit(prefixes, config, "fine")

    assert len(coarse.groups) == 6  # 3 bands x {attn-self, attn-cross}
    assert len(fine.groups) == 24  # 3 bands x 8 leaves
    assert set(coarse.groups) == {
        f"{b}.{p}" for b in ("early", "mid", "late") for p in ("attn-self", "attn-cross")
    }
    assert "early.attn1.to_q" in fine.groups
    # Both partition the same layers — finer is a refinement, not a different set.
    assert coarse.total_layers == fine.total_layers


# --- the naming layer, which CAN have gaps -----------------------------------

def test_an_unknown_architecture_is_grouped_and_named_by_its_raw_leaves(config):
    """No rule matches, so the labels are raw — and the fit still works. That is
    what makes a new model usable before anyone edits the config."""
    prefixes = prefixes_for(["mixer.w_in", "gate.proj"])
    fitted = block_groups.fit(prefixes, config, "coarse")

    assert "early.mixer.w_in" in fitted.groups
    assert set(fitted.unrecognized_parts) == {"mixer.w_in", "gate.proj"}
    assert fitted.total_layers == len(prefixes)


def test_a_ruleless_granularity_reports_no_unrecognized_parts(config):
    """``fine`` has no rules, so a raw leaf name is the intended output, not a
    miss. A signal that fires on every identity fit would carry nothing."""
    fitted = block_groups.fit(prefixes_for(ATTN_LEAVES), config, "fine")
    assert fitted.unrecognized_parts == ()


def test_unknown_granularity_names_the_ones_that_exist(config):
    with pytest.raises(BlockGroupError) as e:
        block_groups.fit(prefixes_for(ATTN_LEAVES), config, "medium")
    assert "coarse" in str(e.value) and "fine" in str(e.value)


# --- patterns must select exactly their members ------------------------------

@pytest.mark.parametrize("granularity", ["coarse", "fine"])
def test_patterns_select_exactly_that_groups_members(config, granularity):
    """Run through lora_soup's real fnmatch matcher, with distractors.

    The old taxonomy carried a regex *and* a glob per part and the two could
    drift; this emits the member prefixes themselves, so the property should
    hold by construction — which is exactly why it is worth asserting against
    the real matcher rather than trusting it. The ``attn1.norm_q`` distractors
    share a block index with real attention members, so an over-broad pattern is
    only catchable with them present."""
    prefixes = prefixes_for(ATTN_LEAVES)
    distractors = [
        f"{BLOCK_PREFIX}.{i}.attn{n}.norm_{qk}"
        for i in range(28) for n in (1, 2) for qk in ("q", "k")
    ]
    fitted = block_groups.fit(prefixes, config, granularity)

    for group, members in fitted.groups.items():
        scales = [(pattern, 0.0) for pattern in fitted.patterns_for(group)]
        member_set = set(members)
        for prefix in prefixes + distractors:
            scale = lora_soup.block_scale_for(prefix, scales)
            expected = 0.0 if prefix in member_set else 1.0
            assert scale == expected, f"{group}: wrong selection for {prefix}"


def test_patterns_for_rejects_a_group_not_in_this_fit(config):
    """A real group name from another checkpoint is still not a coordinate of
    *this* one — the whole premise of fitting."""
    fitted = block_groups.fit(prefixes_for(ATTN_LEAVES), config, "coarse")
    with pytest.raises(BlockGroupError):
        fitted.patterns_for("early.mlp")


# --- layers outside the block prefix -----------------------------------------

def test_layers_outside_the_block_prefix_are_kept_not_dropped(config):
    """A text-encoder LoRA is a real thing to train. Dropping such layers to
    preserve a tidy invariant would silently shrink the coordinate system and
    make every other group's share wrong."""
    prefixes = prefixes_for(ATTN_LEAVES) + [
        "text_encoder.layers.0.self_attn.q_proj",
        "text_encoder.layers.1.mlp.fc1",
    ]
    fitted = block_groups.fit(prefixes, config, "coarse")

    unblocked = {
        g: m for g, m in fitted.groups.items()
        if g.startswith(block_groups.UNBLOCKED_BAND + ".")
    }
    assert sum(len(m) for m in unblocked.values()) == 2
    assert fitted.total_layers == len(prefixes)


# --- band rounding (unchanged behaviour, still load-bearing) -----------------

@pytest.mark.parametrize(
    "block_count,expected",
    [(30, [10, 10, 10]), (29, [9, 9, 11]), (28, [9, 9, 10]), (2, [0, 0, 2])],
)
def test_band_rounding_puts_the_remainder_in_the_last_band(block_count, expected):
    sizes = [0, 0, 0]
    for i in range(block_count):
        sizes[block_groups.band_for_index(i, block_count, 3)] += 1
    assert sizes == expected


def test_bands_are_derived_from_the_observed_count_not_the_model(config):
    """A 12-block checkpoint bands as 4/4/4 even though anima has 28."""
    fitted = block_groups.fit(prefixes_for(["attn1.to_q"], block_count=12), config, "coarse")
    assert fitted.block_count == 12
    assert len(fitted.groups["early.attn-self"]) == 4


# --- key-set reading (unchanged) ---------------------------------------------

def test_lokr_keys_collapse_to_the_same_layer_prefixes(tmp_path):
    """LoKr's seven suffixes and LoRA's three name the same layer — the taxonomy
    is over layers, and the factorization is not one of a layer's coordinates."""
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    path = tmp_path / "lokr.safetensors"
    save_file(
        {
            f"{prefix}.lokr_w1_a": torch.zeros(4, 2),
            f"{prefix}.lokr_w1_b": torch.zeros(2, 4),
            f"{prefix}.lokr_w2": torch.zeros(4, 4),
            f"{prefix}.alpha": torch.tensor(2.0),
        },
        str(path),
    )
    assert block_groups.read_layer_prefixes(path) == [prefix]


def test_lokr_w1_suffix_does_not_swallow_lokr_w1_a():
    """``.lokr_w1`` is a prefix of ``.lokr_w1_a``; endswith order must not let
    the short one claim the long one's key and invent a phantom layer."""
    keys = [
        f"{BLOCK_PREFIX}.0.attn1.to_q.lokr_w1_a",
        f"{BLOCK_PREFIX}.0.attn1.to_q.lokr_w1_b",
    ]
    assert block_groups._layer_prefixes_from_keys(keys) == [f"{BLOCK_PREFIX}.0.attn1.to_q"]
