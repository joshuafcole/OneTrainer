"""Tests for scripts/util/block_compare.py — orientation, not just magnitude.

The module exists because energy profiles cannot distinguish "these two runs
changed the same thing the same way" from "these two runs changed the same
thing in unrelated directions", and a merge lives or dies on that difference.
So the cases pinned here are the ones a caller would act on differently:
identical, scaled, opposed, orthogonal — plus the accumulation identity that
makes a per-layer streaming computation equal the whole-vector cosine it claims
to be, since every group number depends on it.

Adapters are built here as plain LoRA with an explicit ``lora_up`` and
``lora_down``, so the delta of each layer is a quantity the test controls
outright rather than one it has to trust the loader to reproduce.

Run with::

    python -m pytest tests/test_block_compare.py -q
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


block_groups = _load("block_groups", "../scripts/util/block_groups.py")
lora_soup = _load("lora_soup", "../scripts/util/lora_soup.py")
block_compare = _load("block_compare", "../scripts/util/block_compare.py")

CompareError = block_compare.CompareError
BLOCK_PREFIX = "transformer.transformer_blocks"
LEAVES = ["attn1.to_q", "attn1.to_v", "attn2.to_q", "attn2.to_v"]
BLOCKS = 6
RANK, DIM = 4, 16


@pytest.fixture
def config():
    return block_groups.load_groups()


def prefixes(blocks=BLOCKS, leaves=LEAVES):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


def write_adapter(path, deltas_by_prefix, alpha=float(RANK)):
    """A LoRA file whose per-layer delta is *exactly* the tensor supplied.

    ``delta = (alpha/rank) * up @ down``, so putting the whole delta in ``up``
    against an identity-like ``down`` and setting ``alpha == rank`` makes the
    scale factor 1 and the delta readable straight off the input. Anything
    cleverer here would make a failure ambiguous between the fixture and the
    code under test."""
    state = {}
    for prefix, delta in deltas_by_prefix.items():
        out_features, in_features = delta.shape
        down = torch.zeros(RANK, in_features)
        down[: min(RANK, in_features), : min(RANK, in_features)] = torch.eye(
            min(RANK, in_features)
        )
        # up @ down keeps only the first RANK columns of `up`; build `up` so the
        # product is the requested delta restricted to those columns.
        up = torch.zeros(out_features, RANK)
        up[:, : min(RANK, in_features)] = delta[:, : min(RANK, in_features)]
        state[prefix + ".lora_down.weight"] = down.contiguous()
        state[prefix + ".lora_up.weight"] = up.contiguous()
        state[prefix + ".alpha"] = torch.tensor(alpha)
    save_file(state, str(path))
    return path


def deltas(seed, scale=1.0, blocks=BLOCKS, leaves=LEAVES):
    g = torch.Generator().manual_seed(seed)
    return {
        p: torch.randn(DIM, DIM, generator=g) * scale for p in prefixes(blocks, leaves)
    }


def realized(path):
    """What the loader actually reports per layer — the fixture's own output,
    read back, so the identity test compares against measured deltas rather
    than against what the fixture intended them to be."""
    loaded = lora_soup.load_lora(path, 1.0)
    return {k: v.delta().to(torch.float64) for k, v in loaded.layers.items()}


# --- the four cases a merge would act on differently -------------------------

def test_an_adapter_compared_with_itself_is_perfectly_aligned(tmp_path):
    """The identity control. Anything that fails here makes every other number
    in the module unreadable."""
    a = write_adapter(tmp_path / "a.safetensors", deltas(1))
    out = block_compare.compare(a, a, granularity="coarse")

    assert out["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert out["scale_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert out["norm_diff"] == pytest.approx(0.0, abs=1e-9)
    assert out["relative_diff"] == pytest.approx(0.0, abs=1e-9)
    for g in out["groups"]:
        assert g["cosine"] == pytest.approx(1.0, abs=1e-12), g["group"]


def test_a_scaled_copy_is_aligned_but_not_identical(tmp_path):
    """The case that motivates reporting cosine and scale separately: same
    adapter at 3x strength. A method that saw only ``relative_diff`` would call
    this a large disagreement, when it is a knob."""
    base = deltas(2)
    a = write_adapter(tmp_path / "a.safetensors", base)
    b = write_adapter(tmp_path / "b.safetensors", {k: v * 3.0 for k, v in base.items()})
    out = block_compare.compare(a, b, granularity="coarse")

    assert out["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert out["scale_ratio"] == pytest.approx(3.0, rel=1e-6)
    # |A - 3A| / max(|A|, 3|A|) = 2|A| / 3|A|
    assert out["relative_diff"] == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_a_negated_adapter_is_exactly_opposed(tmp_path):
    """Sign has to survive to the report: averaging these two annihilates both,
    and only a negative cosine says so in advance."""
    base = deltas(3)
    a = write_adapter(tmp_path / "a.safetensors", base)
    b = write_adapter(tmp_path / "b.safetensors", {k: -v for k, v in base.items()})
    out = block_compare.compare(a, b, granularity="coarse")

    assert out["cosine"] == pytest.approx(-1.0, abs=1e-9)
    assert out["scale_ratio"] == pytest.approx(1.0, rel=1e-6)
    for g in out["groups"]:
        assert g["cosine"] == pytest.approx(-1.0, abs=1e-9), g["group"]


def test_independent_adapters_are_near_orthogonal(tmp_path):
    """Two unrelated runs. The bound is loose because these are finite random
    draws, not orthogonal by construction — but it is far from the +-1 that the
    aligned and opposed cases produce, which is the discrimination that matters."""
    a = write_adapter(tmp_path / "a.safetensors", deltas(4))
    b = write_adapter(tmp_path / "b.safetensors", deltas(5))
    out = block_compare.compare(a, b, granularity="coarse")

    assert abs(out["cosine"]) < 0.15
    assert out["relative_diff"] > 1.0  # farther apart than either is from zero


# --- the accumulation identity every group number rests on -------------------

def test_group_cosine_equals_the_cosine_of_the_concatenation(tmp_path):
    """The module never materialises a group's ΔW; it sums per-layer dots and
    squared norms. That is only the concatenated cosine if the algebra holds,
    so it is checked against a directly-built concatenation.

    Deliberately run with **different scales per layer** — a uniform fixture
    would satisfy a wrong implementation that averaged per-layer cosines
    instead of accumulating, since the two coincide when norms are equal."""
    base_a, base_b = deltas(6), deltas(7)
    for i, prefix in enumerate(sorted(base_a)):
        base_a[prefix] = base_a[prefix] * (1.0 + i)
        base_b[prefix] = base_b[prefix] * (1.0 + (len(base_a) - i))
    a = write_adapter(tmp_path / "a.safetensors", base_a)
    b = write_adapter(tmp_path / "b.safetensors", base_b)

    got_a, got_b = realized(a), realized(b)
    fitted = block_groups.fit(sorted(got_a), block_groups.load_groups(), "coarse")
    out = block_compare.compare(a, b, granularity="coarse")
    reported = {g["group"]: g for g in out["groups"]}

    assert len(reported) == len(fitted.groups)
    for group, members in fitted.groups.items():
        va = torch.cat([got_a[p].flatten() for p in members])
        vb = torch.cat([got_b[p].flatten() for p in members])
        expected = float(torch.dot(va, vb) / (va.norm() * vb.norm()))
        assert reported[group]["cosine"] == pytest.approx(expected, rel=1e-9), group
        assert reported[group]["norm_diff"] == pytest.approx(
            float((va - vb).norm()), rel=1e-9
        ), group

    # And the whole-adapter figure is the concatenation over every group.
    all_a = torch.cat([got_a[p].flatten() for p in sorted(got_a)])
    all_b = torch.cat([got_b[p].flatten() for p in sorted(got_b)])
    assert out["cosine"] == pytest.approx(
        float(torch.dot(all_a, all_b) / (all_a.norm() * all_b.norm())), rel=1e-9
    )


def test_a_group_with_no_energy_reports_zero_not_nan(tmp_path):
    """A zeroed group has no orientation. NaN would propagate silently through
    any aggregation a caller builds on top; 0.0 is the honest answer."""
    base_a, base_b = deltas(8), deltas(9)
    for prefix in base_a:
        if ".attn1." in prefix:
            base_a[prefix] = torch.zeros_like(base_a[prefix])
    a = write_adapter(tmp_path / "a.safetensors", base_a)
    b = write_adapter(tmp_path / "b.safetensors", base_b)
    out = block_compare.compare(a, b, granularity="coarse")

    selfattn = [g for g in out["groups"] if g["part"] == "attn-self"]
    assert selfattn
    for g in selfattn:
        assert g["cosine"] == 0.0
        assert g["scale_ratio"] == 0.0
        assert g["norm_a"] == pytest.approx(0.0, abs=1e-12)


# --- key sets that do not line up --------------------------------------------

def test_layers_unique_to_one_side_are_named_not_intersected_away(tmp_path):
    """The comparison proceeds on the shared layers — but silently doing so
    would report agreement over a key set the caller never chose. Naming the
    difference is what makes a partial comparison safe to read."""
    a = write_adapter(tmp_path / "a.safetensors", deltas(10, blocks=6))
    b = write_adapter(tmp_path / "b.safetensors", deltas(10, blocks=4))
    out = block_compare.compare(a, b, granularity="coarse")

    assert out["shared_layer_count"] == 4 * len(LEAVES)
    assert len(out["only_in_a"]) == 2 * len(LEAVES)
    assert out["only_in_b"] == []
    assert {k.split(".")[2] for k in out["only_in_a"]} == {"4", "5"}
    # Bands are fitted to the SHARED blocks, so the coordinate system describes
    # what was actually compared rather than what either file happens to hold.
    assert out["block_count"] == 4


def test_disjoint_adapters_refuse_rather_than_report_nothing(tmp_path):
    """An empty intersection is not a comparison with no findings; it means the
    two files do not target the same model."""
    a = write_adapter(tmp_path / "a.safetensors", deltas(11))
    b = write_adapter(
        tmp_path / "b.safetensors",
        {f"text_encoder.layers.{i}.mlp.fc1": torch.randn(DIM, DIM) for i in range(3)},
    )
    with pytest.raises(CompareError) as e:
        block_compare.compare(a, b, granularity="coarse")
    assert "no layers in common" in str(e.value)


def test_a_shape_mismatch_on_a_shared_name_is_refused(tmp_path):
    """Same layer name, different base geometry — subtracting these would be
    meaningless, and broadcasting would make it meaningless *and* silent."""
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    a = write_adapter(tmp_path / "a.safetensors", {prefix: torch.randn(DIM, DIM)})
    b = write_adapter(tmp_path / "b.safetensors", {prefix: torch.randn(DIM * 2, DIM)})
    with pytest.raises(CompareError) as e:
        block_compare.compare(a, b, granularity="coarse")
    assert "shape mismatch" in str(e.value)


def test_granularity_refines_the_comparison_the_same_way_summary_does(tmp_path):
    """Coarse and fine must partition the same shared layers, or a caller
    comparing two granularities is comparing two different populations."""
    a = write_adapter(tmp_path / "a.safetensors", deltas(12))
    b = write_adapter(tmp_path / "b.safetensors", deltas(13))
    coarse = block_compare.compare(a, b, granularity="coarse")
    fine = block_compare.compare(a, b, granularity="fine")

    assert coarse["shared_layer_count"] == fine["shared_layer_count"]
    assert len(fine["groups"]) > len(coarse["groups"])
    assert sum(g["layer_count"] for g in fine["groups"]) == fine["shared_layer_count"]
    # The whole-adapter figures are group-independent, so they must agree exactly.
    assert fine["cosine"] == pytest.approx(coarse["cosine"], rel=1e-12)
    assert fine["norm_diff"] == pytest.approx(coarse["norm_diff"], rel=1e-12)
