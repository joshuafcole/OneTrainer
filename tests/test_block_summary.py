"""Tests for scripts/util/block_summary.py — the summary a caller can act on.

``block_summary`` was the one member of the ``block_*`` family with no tests,
and it acquired a reason for them when it began emitting each group's
``--block-scale`` patterns. Everything else in the payload is a number a reader
looks at; the patterns are a thing a caller *sends back*, and a pattern that
looks right while selecting the wrong layers is the failure mode the whole
fitted-groups design exists to prevent.

So the property pinned here is the round trip, through the real matcher:
**scaling a group by what the summary said selects exactly the layers the
summary counted, and nothing else.** ``test_block_groups.py`` asserts the same
property of ``FittedGroups.patterns_for``; this asserts that what leaves the
process is that, and not a lossy rendering of it.

The distractors matter. ``attn1.norm_q`` shares a block index and a prefix stem
with real attention members, so an over-broad pattern is only catchable with
them in the key set.

Run with::

    python -m pytest tests/test_block_summary.py -q
"""

import importlib.util
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

_here = os.path.dirname(__file__)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


block_groups = _load("block_groups", "../scripts/util/block_groups.py")
lora_soup = _load("lora_soup", "../scripts/util/lora_soup.py")
block_summary = _load("block_summary", "../scripts/util/block_summary.py")

BLOCK_PREFIX = "transformer.transformer_blocks"
ATTN_LEAVES = ["attn1.to_q", "attn1.to_v", "attn2.to_q", "attn2.to_v"]
NORM_LEAVES = ["attn1.norm_q", "attn2.norm_k"]
BLOCKS = 6
RANK, DIM = 4, 16


def prefixes(leaves, blocks=BLOCKS):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


def write_adapter(path, prefix_list, seed=0):
    """A LoRA file over ``prefix_list``, deltas arbitrary but non-degenerate.

    The magnitudes are not the subject here — the selection is — so the only
    property the fixture owes is that no layer's delta is zero, since a zero
    group would make a share of 0.0 ambiguous between "empty" and "cancelled".
    """
    g = torch.Generator().manual_seed(seed)
    state = {}
    for i, prefix in enumerate(prefix_list):
        state[prefix + ".lora_down.weight"] = torch.randn(RANK, DIM, generator=g)
        state[prefix + ".lora_up.weight"] = torch.randn(DIM, RANK, generator=g) * (i + 1)
        state[prefix + ".alpha"] = torch.tensor(float(RANK))
    save_file(state, str(path))
    return path


@pytest.fixture
def adapter(tmp_path):
    """Attention layers *and* the norm leaves that neighbour them."""
    return write_adapter(
        tmp_path / "a.safetensors", prefixes(ATTN_LEAVES) + prefixes(NORM_LEAVES)
    )


@pytest.mark.parametrize("granularity", ["coarse", "fine"])
def test_a_groups_patterns_select_exactly_the_layers_it_counted(adapter, granularity):
    """The round trip, through ``lora_soup``'s own matcher.

    Scaling by a group's emitted patterns must move that group's members and
    leave every other layer at 1.0 — including the norm leaves, which an
    over-broad ``*attn1*`` would sweep up.
    """
    out = block_summary.summarize(adapter, granularity=granularity)
    every_prefix = prefixes(ATTN_LEAVES) + prefixes(NORM_LEAVES)

    for group in out["groups"]:
        scales = [(pattern, 0.0) for pattern in group["patterns"]]
        selected = {
            prefix
            for prefix in every_prefix
            if lora_soup.block_scale_for(prefix, scales) == 0.0
        }
        assert len(selected) == group["layer_count"], (
            f"{group['group']}: selected {len(selected)} layers, counted "
            f"{group['layer_count']}"
        )


@pytest.mark.parametrize("granularity", ["coarse", "fine"])
def test_every_layer_is_selected_by_exactly_one_group(adapter, granularity):
    """Coverage is structural (groups are fitted to the layers present), so the
    patterns inherit it. Asserted at the wire rather than at the fit, because
    this is the level a caller builds a scale table from: two groups both
    claiming a layer would multiply their coefficients on it."""
    out = block_summary.summarize(adapter, granularity=granularity)

    claims = {}
    for group in out["groups"]:
        scales = [(pattern, 0.0) for pattern in group["patterns"]]
        for prefix in prefixes(ATTN_LEAVES) + prefixes(NORM_LEAVES):
            if lora_soup.block_scale_for(prefix, scales) == 0.0:
                claims.setdefault(prefix, []).append(group["group"])

    assert claims, "no layer was claimed by any group"
    multiply_claimed = {p: g for p, g in claims.items() if len(g) > 1}
    assert not multiply_claimed, multiply_claimed
    assert set(claims) == set(prefixes(ATTN_LEAVES) + prefixes(NORM_LEAVES))


def test_a_pattern_does_not_escape_its_own_granularity(adapter):
    """A ``fine`` group is a leaf path; a ``coarse`` one pools several. The fine
    patterns must therefore be a strict refinement — every fine group's members
    inside exactly one coarse group — or the two levels are not the same
    coordinate system viewed at two resolutions, and switching between them in a
    UI would silently re-target a coefficient."""
    fine = block_summary.summarize(adapter, granularity="fine")
    coarse = block_summary.summarize(adapter, granularity="coarse")
    every_prefix = prefixes(ATTN_LEAVES) + prefixes(NORM_LEAVES)

    def members(group):
        scales = [(pattern, 0.0) for pattern in group["patterns"]]
        return {p for p in every_prefix if lora_soup.block_scale_for(p, scales) == 0.0}

    coarse_members = [members(g) for g in coarse["groups"]]
    for group in fine["groups"]:
        m = members(group)
        containing = [c for c in coarse_members if m <= c]
        assert len(containing) == 1, f"{group['group']} straddles {len(containing)} coarse groups"


def test_the_payload_survives_the_process_boundary(adapter):
    """It is emitted as JSON and parsed by pydantic on the other side. A tensor
    or a numpy scalar that reads fine in-process is an ``Invalid JSON`` on the
    agent, reported as an out-of-date checkout — see ``result_channel``."""
    out = block_summary.summarize(adapter, granularity="fine")
    round_tripped = json.loads(json.dumps(out))

    assert round_tripped == out
    assert all(isinstance(p, str) for g in out["groups"] for p in g["patterns"])


def test_patterns_are_present_for_every_group(adapter):
    """The field is what makes a group actuatable; a group without it is a row a
    UI can render and not act on."""
    out = block_summary.summarize(adapter, granularity="coarse")

    assert out["groups"]
    for group in out["groups"]:
        assert group["patterns"], f"{group['group']} carries no patterns"
        assert len(group["patterns"]) == group["layer_count"]
