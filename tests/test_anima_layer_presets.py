"""What `BaseAnimaSetup.LAYER_PRESETS` actually selects on Anima's transformer.

A layer preset is a claim about an architecture, not a preference, so it can be
measured. This measures all four against the real module tree of
`CosmosTransformer3DModel` built from Anima-Base-v1.0's published
`transformer/config.json` -- on the `meta` device, so no weights are downloaded
and no memory is allocated.

Two results are worth stating up front, because both are easy to get wrong by
reading the pattern strings:

* "blocks" is not a coarser surface than a leaf-level filter. `transformer_block`
  is a substring filter on the container, so it adapts *every* Linear inside every
  block -- attention, feedforward and the three adaLN modulations alike. A
  hypothetical "attn-mlp plus the adaLN modulations" preset would select exactly
  the same 448 modules, which is why there isn't one.
* The whole difference between "attn-mlp" and "blocks" is those six adaLN Linears
  per block: 280 -> 448 modules, 1.51x the LoRA parameters at equal rank. The
  comment on LAYER_PRESETS quotes both numbers; they are produced here so they
  cannot drift away from the code that motivated them.
"""

from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.util.ModuleFilter import ModuleFilter

import torch
from torch.nn import Conv2d, Linear

from diffusers import CosmosTransformer3DModel

import pytest

# Verbatim from https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers
# -> transformer/config.json. Pinned rather than fetched: the point of the test is
# that the filters match *this* architecture, and a test that needs the network is
# a test that gets skipped.
ANIMA_TRANSFORMER_CONFIG = {
    "adaln_lora_dim": 256,
    "attention_head_dim": 128,
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
    "in_channels": 16,
    "max_size": [128, 240, 240],
    "mlp_ratio": 4.0,
    "num_attention_heads": 16,
    "num_layers": 28,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 1024,
}

# CosmosAdaLayerNormZero holds linear_1 (the adaln-LoRA down projection) and
# linear_2 (shift/scale/gate). Three of them per block.
ADALN_LEAVES = frozenset(f"norm{i}.linear_{j}" for i in (1, 2, 3) for j in (1, 2))

# Linears outside the transformer blocks: patchify, unpatchify, the final adaLN
# and the timestep embedder.
TRUNK = frozenset({
    "norm_out.linear_1",
    "norm_out.linear_2",
    "patch_embed.proj",
    "proj_out",
    "time_embed.t_embedder.linear_1",
    "time_embed.t_embedder.linear_2",
})


@pytest.fixture(scope="module")
def target_modules() -> dict[str, Linear | Conv2d]:
    """Every module `LoRAModuleWrapper` would consider adapting, by name.

    Mirrors `LoRAModuleWrapper.__collect_target_modules`: Linear and Conv2d only.
    """
    with torch.device("meta"):
        transformer = CosmosTransformer3DModel.from_config(ANIMA_TRANSFORMER_CONFIG)

    return {
        name: module
        for name, module in transformer.named_modules()
        if isinstance(module, Linear | Conv2d)
    }


def _select(patterns, target_modules, use_regex: bool = False) -> set[str]:
    """The module names these patterns select, over the real filter code.

    Deliberately routed through `",".join(...).split(",")`, because that is the
    round trip a preset makes in production: the layer-filter widget joins the
    pattern list into `config.layer_filter`, and every LoRA setup splits it again.
    It is what turns "full"'s empty list into the single empty pattern that
    `ModuleFilter` treats as match-everything.
    """
    filters = [ModuleFilter(p, use_regex=use_regex) for p in ",".join(patterns).split(",")]
    return {name for name in target_modules if any(f.matches(name) for f in filters)}


def _preset(name: str, target_modules) -> set[str]:
    definition = BaseAnimaSetup.LAYER_PRESETS[name]
    if isinstance(definition, dict):
        return _select(definition["patterns"], target_modules, bool(definition.get("regex", False)))
    return _select(definition, target_modules)


def _lora_parameters(names: set[str], target_modules, rank: int = 16) -> int:
    """Parameter count of a rank-`rank` LoRA over `names` (down [r, in] + up [out, r])."""
    return sum(rank * (target_modules[n].in_features + target_modules[n].out_features) for n in names)


def test_blocks_adapts_every_linear_inside_a_block_including_the_adaln_modulations(target_modules):
    """`transformer_block` filters on the container, so nothing block-internal escapes it."""
    blocks = _preset("blocks", target_modules)

    assert set(target_modules) - blocks == TRUNK
    assert all(name.startswith("transformer_blocks.") for name in blocks)

    # 16 Linears per block: 4 in attn1, 4 in attn2, 2 in ff, 6 in the adaLN modulations.
    assert len(blocks) == ANIMA_TRANSFORMER_CONFIG["num_layers"] * 16


def test_a_leaf_level_detail_filter_would_be_a_synonym_for_blocks(target_modules):
    """attn-mlp + the adaLN modulations is not a surface between attn-mlp and blocks.

    Written down because it is the natural next preset to reach for -- "give style
    LoRAs the conditioning offsets" -- and because both spellings of it, substring
    and regex, land on precisely the set `blocks` already selects.
    """
    blocks = _preset("blocks", target_modules)

    as_substrings = _select(
        ["attn1", "attn2", "ff", "norm1.linear", "norm2.linear", "norm3.linear"], target_modules
    )
    as_regex = _select(
        [r"^(?=.*attn)(?!.*norm).*", r"^(?=.*ff\.net).*", r"^(?=.*norm[123]\.linear).*"],
        target_modules,
        use_regex=True,
    )

    assert as_substrings == blocks
    assert as_regex == blocks


def test_the_adaln_modulations_are_the_whole_gap_between_attn_mlp_and_blocks(target_modules):
    attn_mlp = _preset("attn-mlp", target_modules)
    blocks = _preset("blocks", target_modules)

    added = blocks - attn_mlp
    assert {name.split(".", 2)[-1] for name in added} == set(ADALN_LEAVES)
    assert len(added) == ANIMA_TRANSFORMER_CONFIG["num_layers"] * len(ADALN_LEAVES)


def test_blocks_costs_1_51x_attn_mlp_at_equal_rank(target_modules):
    """The numbers quoted on LAYER_PRESETS, produced rather than asserted from memory."""
    attn_mlp = _preset("attn-mlp", target_modules)
    blocks = _preset("blocks", target_modules)

    assert (len(attn_mlp), len(blocks)) == (280, 448)
    assert _lora_parameters(attn_mlp, target_modules) == 22_937_600
    assert _lora_parameters(blocks, target_modules) == 34_635_776

    # rank-independent, so a reader can carry the ratio to any rank they like
    for rank in (4, 16, 32, 128):
        ratio = _lora_parameters(blocks, target_modules, rank) / _lora_parameters(attn_mlp, target_modules, rank)
        assert round(ratio, 2) == 1.51


def test_every_anima_layer_preset_selects_something(target_modules):
    """A preset matching nothing fails at setup with "no modules were matched"."""
    for name in BaseAnimaSetup.LAYER_PRESETS:
        selected = _preset(name, target_modules)
        assert selected, f"layer preset {name!r} matches no module in Anima's transformer"
        if name == "full":
            assert selected == set(target_modules)


def test_the_presets_are_distinct_and_widen_in_dropdown_order(target_modules):
    """Two presets selecting the same set are two dropdown rows the user cannot tell apart."""
    selected = {name: _preset(name, target_modules) for name in BaseAnimaSetup.LAYER_PRESETS}

    for a, b in zip(selected, list(selected)[1:], strict=False):
        assert selected[a] != selected[b], f"layer presets {a!r} and {b!r} select the same modules"

    ordered = [len(selected[name]) for name in ("attn-only", "attn-mlp", "blocks", "full")]
    assert ordered == [224, 280, 448, 454]
    assert set(selected) == {"attn-only", "attn-mlp", "blocks", "full"}
