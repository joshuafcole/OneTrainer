"""Tests for scripts/util/block_contribution.py — the activation term.

The thing this script exists to detect is *small*: a few percent of divergence
between a merge weighted by activation-aware contribution and one weighted by
weight norms alone. So a bug that scales, biases, or averages the wrong way
does not announce itself. It produces a plausible table.

Two families of test guard that. First, the algebra: the contribution is the
quantity it claims to be, computed against the loader's own deltas rather than
the fixture's intent. Second, the **isotropic identity** —

    E || dW x ||^2 = ||dW||_F^2 * E||x||^2 / d_in

which is the null this whole measurement is defined against. If activations are
white, contribution must reduce to the Frobenius norm exactly; a test that pins
that is what makes a *non*-null result on real activations mean something,
because it rules out the measurement inventing structure of its own.

The capture half is model-shaped and can only be smoke-tested on a box with
Anima, so what is tested here is the part that can be: prefix resolution
against a synthetic module tree, the CFG call arithmetic, and the
recomputation guard.

Run with::

    python -m pytest tests/test_block_contribution.py -q
"""

import importlib.util
import os
import sys

import torch

import pytest
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
block_contribution = _load("block_contribution", "../scripts/util/block_contribution.py")

ContributionError = block_contribution.ContributionError
BLOCK_PREFIX = "transformer.transformer_blocks"
LEAVES = ["attn1.to_q", "attn1.to_v"]
BLOCKS = 2
RANK, DIM = 4, 8


def prefixes(blocks=BLOCKS, leaves=LEAVES):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


def write_adapter(path, deltas_by_prefix, alpha=float(RANK)):
    """A LoRA file whose per-layer delta is exactly the tensor supplied."""
    state = {}
    for prefix, delta in deltas_by_prefix.items():
        out_features, in_features = delta.shape
        k = min(RANK, in_features)
        down = torch.zeros(RANK, in_features)
        down[:k, :k] = torch.eye(k)
        up = torch.zeros(out_features, RANK)
        up[:, :k] = delta[:, :k]
        state[prefix + ".lora_down.weight"] = down.contiguous()
        state[prefix + ".lora_up.weight"] = up.contiguous()
        state[prefix + ".alpha"] = torch.tensor(alpha)
    save_file(state, str(path))
    return path


def deltas(seed, scale=1.0, blocks=BLOCKS, leaves=LEAVES):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(DIM, DIM, generator=g) * scale
            for p in prefixes(blocks, leaves)}


def make(tmp_path, seeds, blocks=BLOCKS):
    return [
        write_adapter(tmp_path / f"a{i}.safetensors", deltas(s, blocks=blocks))
        for i, s in enumerate(seeds)
    ]


def realized(paths):
    return [
        {k: v.delta().to(torch.float64) for k, v in lora_soup.load_lora(p, 1.0).layers.items()}
        for p in paths
    ]


def activations(labels, rows=32, seed=0, blocks=BLOCKS, leaves=LEAVES, dim=DIM):
    g = torch.Generator().manual_seed(seed)
    return {
        label: {p: torch.randn(rows, dim, generator=g) for p in prefixes(blocks, leaves)}
        for label in labels
    }


def run(paths, acts, **kw):
    return block_contribution.score(
        paths=paths, activations=acts, prompts=list(acts), **kw
    )


# --- the algebra ------------------------------------------------------------

def test_contribution_is_the_quantity_it_claims_to_be(tmp_path):
    """``||dW X^T||_F^2 / rows``, against the loader's own deltas.

    Compared to a direct computation rather than to the fixture's intended
    delta, so a loader-side scale bug cannot cancel itself out of both sides."""
    paths = make(tmp_path, [1, 2, 3])
    acts = activations(["a", "b"])
    out = run(paths, acts)
    real = realized(paths)

    for layer in out["layers"]:
        prefix = layer["layer"]
        for p, label in enumerate(out["prompt_labels"]):
            x = acts[label][prefix].to(torch.float64)
            for i, d in enumerate(real):
                want = float(((d[prefix] @ x.T) ** 2).sum()) / x.shape[0]
                assert layer["contribution_sq"][p][i] == pytest.approx(want, rel=1e-12)


def test_frobenius_is_the_gram_diagonal(tmp_path):
    """The isotropic prediction is emitted beside the measurement, and it is
    the same number ``block_gram`` puts on its diagonal — so the two tables can
    be divided without a units argument."""
    paths = make(tmp_path, [4, 5])
    out = run(paths, activations(["a"]))
    real = realized(paths)
    for layer in out["layers"]:
        for i, d in enumerate(real):
            want = float((d[layer["layer"]] ** 2).sum())
            assert layer["frobenius_sq"][i] == pytest.approx(want, rel=1e-12)


def test_activation_energy_is_per_row(tmp_path):
    paths = make(tmp_path, [6])
    acts = activations(["a"], rows=64)
    out = run(paths, acts)
    for layer in out["layers"]:
        x = acts["a"][layer["layer"]].to(torch.float64)
        assert layer["rows"] == [64]
        assert layer["activation_energy"][0] == pytest.approx(
            float((x * x).sum()) / 64, rel=1e-12
        )


def test_isotropic_activations_reduce_contribution_to_the_frobenius_norm(tmp_path):
    """**The null this measurement is defined against.**

    With ``X^T X = c I`` exactly -- orthonormal rows, scaled -- the identity
    ``||dW X^T||_F^2 = c ||dW||_F^2`` holds with no expectation and no sampling
    error. Any deviation here is the script inventing structure, which is the
    one failure mode that would be invisible on real data: a spurious
    "activation effect" that is really an indexing or normalisation bug.
    """
    paths = make(tmp_path, [7, 8, 9])
    g = torch.Generator().manual_seed(11)
    # DIM orthonormal rows in DIM dimensions, scaled: X^T X = scale^2 I.
    q, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=g).to(torch.float64))
    scale = 3.0
    x = (q * scale).to(torch.float32)
    acts = {"iso": {p: x.clone() for p in prefixes()}}
    out = run(paths, acts)

    for layer in out["layers"]:
        for i in range(3):
            predicted = layer["frobenius_sq"][i] * scale**2 / DIM
            # 1e-6, not 1e-12: the capture stores activations in float32 (that
            # is what the memory budget buys), so orthonormality survives the
            # round-trip only to float32 precision. Four orders of magnitude
            # below the ~1% anisotropy this script is looking for.
            assert layer["contribution_sq"][0][i] == pytest.approx(predicted, rel=1e-6)


def test_anisotropic_activations_move_the_ranking(tmp_path):
    """The negative control on the null above.

    An identity that holds for every ``X`` would be a tautology, not a
    measurement. Here two adapters are built so that the Frobenius norm ranks
    them one way and a directional ``X`` ranks them the other -- exactly the
    case the whole script exists to detect. If this ever passes trivially,
    contribution has silently become a function of the weights alone.
    """
    prefix = prefixes()[0]
    # `big` has more total mass; `aimed` puts less mass entirely along e0.
    big = torch.zeros(DIM, DIM)
    big[:, 1:] = 1.0
    aimed = torch.zeros(DIM, DIM)
    aimed[:, 0] = 0.9
    paths = [
        write_adapter(tmp_path / "big.safetensors", {prefix: big}),
        write_adapter(tmp_path / "aimed.safetensors", {prefix: aimed}),
    ]
    x = torch.zeros(16, DIM)
    x[:, 0] = 1.0  # activations live entirely on e0
    out = run(paths, {"e0": {prefix: x}})

    layer = out["layers"][0]
    assert layer["frobenius_sq"][0] > layer["frobenius_sq"][1]
    assert layer["contribution_sq"][0][0] < layer["contribution_sq"][0][1]


def test_scoring_is_invariant_to_prompt_order(tmp_path):
    """Prompts are labels, not a fit — reordering them permutes the rows and
    changes nothing else."""
    paths = make(tmp_path, [12, 13])
    acts = activations(["a", "b", "c"])
    forward = run(paths, acts)
    backward = run(paths, {k: acts[k] for k in ["c", "b", "a"]})
    for lf, lb in zip(forward["layers"], backward["layers"], strict=True):
        assert lf["contribution_sq"] == list(reversed(lb["contribution_sq"]))
        assert lf["frobenius_sq"] == lb["frobenius_sq"]


# --- what it refuses --------------------------------------------------------

def test_a_layer_missing_from_one_prompt_is_an_error_not_a_zero(tmp_path):
    """A zero contribution and an absent capture are different claims, and only
    one of them is about the adapter."""
    paths = make(tmp_path, [14])
    acts = activations(["a", "b"])
    del acts["b"][prefixes()[0]]
    with pytest.raises(ContributionError, match="captured for some prompts but not"):
        run(paths, acts)


def test_a_width_mismatch_is_refused_not_broadcast(tmp_path):
    paths = make(tmp_path, [15])
    acts = activations(["a"], dim=DIM)
    acts["a"][prefixes()[0]] = torch.randn(8, DIM + 1)
    with pytest.raises(ContributionError, match="not about the same base model"):
        run(paths, acts)


def test_no_overlap_between_adapters_and_capture_is_refused(tmp_path):
    paths = make(tmp_path, [16])
    acts = {"a": {"transformer.somewhere.else.to_q": torch.randn(8, DIM)}}
    with pytest.raises(ContributionError, match="not about the same model"):
        run(paths, acts)


def test_a_layer_captured_but_not_shared_is_named_not_scored(tmp_path):
    """Both directions of the mismatch are reported, as ``block_gram`` reports
    dropped layers: a table over a quietly reduced key set describes a
    different object than the caller asked about."""
    paths = make(tmp_path, [17])
    acts = activations(["a"])
    acts["a"]["transformer.transformer_blocks.9.attn1.to_q"] = torch.randn(8, DIM)
    out = run(paths, acts)
    assert out["unscored_layers"] == ["transformer.transformer_blocks.9.attn1.to_q"]
    assert out["uncaptured_layers"] == []


def test_a_layer_shared_but_not_captured_is_named_not_scored(tmp_path):
    paths = make(tmp_path, [18])
    acts = activations(["a"])
    absent = prefixes()[-1]
    for per_layer in acts.values():
        del per_layer[absent]
    out = run(paths, acts)
    assert out["uncaptured_layers"] == [absent]
    assert absent not in [layer["layer"] for layer in out["layers"]]


def test_prompts_and_activation_sets_must_correspond(tmp_path):
    paths = make(tmp_path, [19])
    with pytest.raises(ContributionError, match="positional"):
        block_contribution.score(
            paths=paths, activations=activations(["a", "b"]), prompts=["only one"]
        )


def test_too_many_adapters_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(block_contribution, "MAX_ADAPTERS", 2)
    paths = make(tmp_path, [20, 21, 22])
    with pytest.raises(ContributionError, match="at most 2 adapters"):
        run(paths, activations(["a"]))


def test_the_adapter_cap_is_above_the_gram_cap():
    """This one's *memory* is linear in the adapter count -- one delta at a time,
    and the deltas dominate -- so the ceiling that makes the Gram sensible would
    refuse a mixture this can afford. Pinned because the two numbers look like
    they should agree, and because the pair table added later makes the
    arithmetic quadratic without moving the constraint."""
    block_gram = _load("block_gram", "../scripts/util/block_gram.py")
    assert block_contribution.MAX_ADAPTERS > block_gram.MAX_ADAPTERS


# --- the pair tables: what a merge coefficient is actually solved against -----

def test_the_gram_diagonal_is_the_contribution(tmp_path):
    """Not two computations that agree -- one, reported twice. A diagonal that
    could drift from ``contribution_sq`` would let a planner and a report
    disagree about the same adapter."""
    paths = make(tmp_path, [1, 2, 3])
    out = run(paths, activations(["a", "b"]), emit_layer_gram=True)
    for layer in out["layers"]:
        for p in range(len(out["prompt_labels"])):
            for i in range(3):
                assert layer["contribution_gram"][p][i][i] == layer["contribution_sq"][p][i]
        for i in range(3):
            assert layer["frobenius_gram"][i][i] == layer["frobenius_sq"][i]


def test_the_pair_tables_are_symmetric(tmp_path):
    """Exactly, not approximately: a solver reading this expects a real
    symmetric matrix, and float noise in the lower triangle is how a real
    problem grows a complex eigenvalue."""
    paths = make(tmp_path, [7, 8, 9])
    out = run(paths, activations(["a"]), emit_layer_gram=True)
    for layer in out["layers"]:
        for table in [layer["frobenius_gram"], *layer["contribution_gram"]]:
            for i, row in enumerate(table):
                for j, value in enumerate(row):
                    assert value == table[j][i]


def test_the_gram_predicts_a_merge_the_diagonal_does_not(tmp_path):
    """**The reason the off-diagonals are computed.**

    ``|| sum_i c_i dW_i X ||^2 = c^T G c``. Checked against a directly merged
    delta, and checked to *differ* from the diagonal-only prediction
    ``sum_i c_i^2 contribution_i`` -- because if those agreed on real adapters
    the whole pair table would be dead weight."""
    paths = make(tmp_path, [11, 12, 13])
    acts = activations(["a"])
    out = run(paths, acts, emit_layer_gram=True)
    real = realized(paths)
    coefficients = [0.5, 0.3, -0.2]

    quadratic_form_matched = False
    for layer in out["layers"]:
        prefix = layer["layer"]
        x = acts["a"][prefix].to(torch.float64)
        merged = sum(c * d[prefix] for c, d in zip(coefficients, real, strict=True))
        want = float(((merged @ x.T) ** 2).sum()) / x.shape[0]

        gram = layer["contribution_gram"][0]
        got = sum(
            coefficients[i] * coefficients[j] * gram[i][j]
            for i in range(3)
            for j in range(3)
        )
        assert got == pytest.approx(want, rel=1e-10)
        quadratic_form_matched = True

        diagonal_only = sum(
            coefficients[i] ** 2 * layer["contribution_sq"][0][i] for i in range(3)
        )
        assert diagonal_only != pytest.approx(want, rel=1e-3)

    assert quadratic_form_matched, "no layer was checked"


def test_the_frobenius_gram_is_the_weight_space_inner_product(tmp_path):
    """The same quantity ``block_gram`` reports, so the activation-weighted
    table has a null to be read against without a units argument."""
    paths = make(tmp_path, [14, 15])
    out = run(paths, activations(["a"]), emit_layer_gram=True)
    real = realized(paths)
    for layer in out["layers"]:
        prefix = layer["layer"]
        for i in range(2):
            for j in range(2):
                want = float((real[i][prefix] * real[j][prefix]).sum())
                assert layer["frobenius_gram"][i][j] == pytest.approx(want, rel=1e-12)


def test_a_group_gram_is_the_sum_of_its_layers(tmp_path):
    """A coefficient is solved per group, so the group table has to be the thing
    the per-layer tables add up to -- not a mean, not a representative layer."""
    paths = make(tmp_path, [16, 17, 18])
    out = run(paths, activations(["a", "b"]), emit_layer_gram=True)
    by_group = {}
    for layer in out["layers"]:
        by_group.setdefault(layer["group"], []).append(layer)

    assert {g["group"] for g in out["groups"]} == set(by_group)
    for entry in out["groups"]:
        members = by_group[entry["group"]]
        assert entry["layer_count"] == len(members)
        for i in range(3):
            for j in range(3):
                want = sum(m["frobenius_gram"][i][j] for m in members)
                assert entry["frobenius_gram"][i][j] == pytest.approx(want, rel=1e-12)
                for p in range(2):
                    want_c = sum(m["contribution_gram"][p][i][j] for m in members)
                    assert entry["contribution_gram"][p][i][j] == pytest.approx(
                        want_c, rel=1e-12
                    )


def test_layer_grams_are_off_by_default(tmp_path):
    """``layers * prompts * N^2`` floats is diagnostic detail; the group tables
    are what a planner reads, so the default response does not carry it."""
    paths = make(tmp_path, [19, 20])
    out = run(paths, activations(["a"]))
    assert out["groups"], "group tables are not optional"
    for layer in out["layers"]:
        assert "contribution_gram" not in layer
        assert "frobenius_gram" not in layer


# --- capture: the parts that do not need a model -----------------------------

class _Tree(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "attn1": torch.nn.ModuleDict({
                    "to_q": torch.nn.Linear(DIM, DIM),
                    "to_v": torch.nn.Linear(DIM, DIM),
                }),
                "norm": torch.nn.ModuleDict({"scale": torch.nn.LayerNorm(DIM)}),
            })
            for _ in range(BLOCKS)
        ])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block["attn1"]["to_q"](x) + block["attn1"]["to_v"](x)
        return x


def test_prefixes_resolve_to_the_modules_the_wrapper_named():
    """``LoRAModuleWrapper(model.transformer, "transformer", ...)`` names layers
    ``transformer.<named_modules path>``, so resolution is that inverted."""
    tree = _Tree()
    resolved, unmatched = resolve = block_contribution.resolve_modules(tree, prefixes())
    assert unmatched == []
    assert sorted(resolved) == prefixes()
    assert resolved[f"{BLOCK_PREFIX}.0.attn1.to_q"] is tree.transformer_blocks[0]["attn1"]["to_q"]
    assert resolve[0] is resolved


def test_an_unresolvable_prefix_is_returned_not_raised():
    """An adapter that also targets the text encoder is a normal thing to be
    handed; the right answer is to score the transformer and say what was left."""
    tree = _Tree()
    resolved, unmatched = block_contribution.resolve_modules(
        tree, [*prefixes(), "text_encoder.layers.0.q_proj", "transformer.nope.to_q"]
    )
    assert sorted(resolved) == prefixes()
    assert unmatched == ["text_encoder.layers.0.q_proj", "transformer.nope.to_q"]


def test_a_non_linear_module_is_not_a_target():
    """The delta is 2-D ``(out, in)`` and the capture is rows of ``in``; a
    LayerNorm has neither, so hooking one would produce a shape error deep in
    scoring instead of an absence here."""
    resolved, unmatched = block_contribution.resolve_modules(
        _Tree(), [f"{BLOCK_PREFIX}.0.norm.scale"]
    )
    assert resolved == {}
    assert unmatched == [f"{BLOCK_PREFIX}.0.norm.scale"]


def _capture_over(tree, calls, max_tokens=8, rows=16, passes=1):
    capture = block_contribution.ActivationCapture(calls, max_tokens, seed=3)
    resolved, _ = block_contribution.resolve_modules(tree, prefixes())
    capture.attach(resolved, tree)
    with torch.no_grad():
        for _ in range(passes):
            tree(torch.randn(rows, DIM))
    capture.detach()
    return capture


def test_capture_keeps_only_the_calls_it_was_asked_for():
    """With CFG the sampler makes two transformer calls per step, conditional
    first. Recording the wrong one averages the negative prompt into the
    measurement, silently."""
    tree = _Tree()
    capture = _capture_over(tree, calls=[0, 2], passes=4)
    assert capture.call_index == 3
    for prefix in prefixes():
        # Two recorded calls, per_call rows each.
        assert capture.rows[prefix][0].shape[0] == capture.per_call
        assert len(capture.rows[prefix]) == 2


def test_capture_is_capped_at_max_tokens():
    tree = _Tree()
    capture = _capture_over(tree, calls=[0, 1, 2], max_tokens=7, rows=16, passes=3)
    for x in capture.collected().values():
        assert x.shape == (7, DIM)


def test_a_recomputed_forward_is_not_counted_twice():
    """Gradient checkpointing re-runs a block's forward. Without the
    ``(layer, call)`` guard those tokens are counted twice — and only for the
    checkpointed layers, which is a per-layer bias in exactly the quantity
    being compared per layer."""
    tree = _Tree()
    capture = block_contribution.ActivationCapture([0], max_tokens=8, seed=3)
    resolved, _ = block_contribution.resolve_modules(tree, prefixes())
    capture.attach(resolved, tree)
    x = torch.randn(16, DIM)
    with torch.no_grad():
        for block in tree.transformer_blocks:
            capture.call_index = 0  # the root hook fired once; replay a block
            block["attn1"]["to_q"](x)
            block["attn1"]["to_q"](x)
    capture.detach()
    for prefix in prefixes(leaves=["attn1.to_q"]):
        assert len(capture.rows[prefix]) == 1


def test_the_same_seed_picks_the_same_tokens():
    """Common random numbers across adapters is what makes the *ratio* between
    two adapters accurate where the absolute number is not. The tokens must
    therefore be a function of (layer, call, seed) and nothing else."""
    torch.manual_seed(99)
    tree = _Tree()
    resolved, _ = block_contribution.resolve_modules(tree, prefixes())
    x = torch.randn(64, DIM)

    def once():
        capture = block_contribution.ActivationCapture([0], max_tokens=8, seed=5)
        capture.attach(resolved, tree)
        with torch.no_grad():
            tree(x)
        capture.detach()
        return capture.collected()

    first, second = once(), once()
    for prefix in prefixes():
        assert torch.equal(first[prefix], second[prefix])


def test_a_different_layer_gets_different_tokens():
    """The per-layer digest must actually vary, or every layer sees the same
    token subset and any layer-to-layer structure is the subsample's."""
    torch.manual_seed(100)
    tree = _Tree()
    resolved, _ = block_contribution.resolve_modules(tree, prefixes())
    capture = block_contribution.ActivationCapture([0], max_tokens=8, seed=5)
    capture.attach(resolved, tree)
    with torch.no_grad():
        tree(torch.randn(64, DIM))
    capture.detach()
    picked = capture.collected()
    a = picked[f"{BLOCK_PREFIX}.0.attn1.to_q"]
    b = picked[f"{BLOCK_PREFIX}.0.attn1.to_v"]
    # Same input tensor, same call — so equality here would mean the digest is
    # not reaching the generator.
    assert not torch.equal(a, b)


@pytest.mark.parametrize("total,count,expected", [
    (25, 4, [0, 8, 16, 24]),
    (25, 1, [0]),
    (1, 4, [0]),
    (4, 4, [0, 1, 2, 3]),
    (25, 100, list(range(25))),
])
def test_capture_steps_spread_over_the_trajectory(total, count, expected):
    """Endpoints included: contribution at step 0 is layout and at the last
    step is texture, and an adapter can be strong in one regime and absent in
    the other."""
    assert block_contribution.spread_steps(total, count) == expected


# --- the capture cache ------------------------------------------------------

def _meta(**over):
    base = {
        "base_model": "/models/anima", "height": 1024, "width": 1024,
        "diffusion_steps": 25, "capture_steps": [0, 8, 16, 24], "cfg_scale": 1.0,
        "negative_prompt": "", "seed": 0, "max_tokens": 256,
        "device": "cuda", "calls_per_step": 1,
    }
    base.update(over)
    return base


def test_a_capture_round_trips_through_a_file(tmp_path):
    acts = activations(["line0", "line1"], rows=5)
    prompts = ["a photo", "a drawing"]
    path = tmp_path / "cap.safetensors"
    block_contribution.save_activations(path, acts, _meta(), prompts)

    back, meta, back_prompts, key = block_contribution.load_activations(path)
    assert back_prompts == prompts
    assert meta["base_model"] == "/models/anima"
    assert sorted(back) == ["line0", "line1"]
    for label in acts:
        for prefix, x in acts[label].items():
            assert torch.equal(back[label][prefix], x)
    assert key == block_contribution.cache_key(_meta(), prompts, prefixes())


def test_a_cached_capture_scores_identically_to_a_fresh_one(tmp_path):
    """The cache exists so the GPU half runs once. It is only worth anything if
    the numbers are the same either way."""
    paths = make(tmp_path, [30, 31])
    acts = activations(["line0"], rows=12)
    direct = run(paths, acts)
    path = tmp_path / "cap.safetensors"
    block_contribution.save_activations(path, acts, _meta(), ["a photo"])
    back, _, prompts, _ = block_contribution.load_activations(path)
    cached = block_contribution.score(paths=paths, activations=back, prompts=prompts)
    for a, b in zip(direct["layers"], cached["layers"], strict=True):
        assert a["contribution_sq"] == b["contribution_sq"]


@pytest.mark.parametrize("field,value", [
    ("base_model", "/models/other"),
    ("height", 512),
    ("width", 512),
    ("diffusion_steps", 30),
    ("capture_steps", [0, 12, 24]),
    ("cfg_scale", 1.5),
    ("negative_prompt", "blurry"),
    ("seed", 1),
    ("max_tokens", 128),
])
def test_every_capture_parameter_changes_the_key(field, value):
    """**The whole contract of the cache.**

    A parameter missing from the key is one a caller can change while being
    served the old activations — a stale answer delivered with no warning and
    no way to notice. Each one is pinned individually rather than by a single
    happy-path test, because the failure mode of forgetting one is silence."""
    prompts = ["a photo"]
    before = block_contribution.cache_key(_meta(), prompts, prefixes())
    after = block_contribution.cache_key(_meta(**{field: value}), prompts, prefixes())
    assert before != after


def test_the_prompts_and_the_layer_set_are_in_the_key():
    """The layer set especially: a capture taken for an attention-only adapter
    never hooked the MLPs, so reusing it for a wider adapter would score a
    subset and report it as a smaller layer count — which reads as a fact about
    the adapter."""
    base = block_contribution.cache_key(_meta(), ["a photo"], prefixes())
    assert base != block_contribution.cache_key(_meta(), ["a drawing"], prefixes())
    assert base != block_contribution.cache_key(_meta(), ["a photo", "b"], prefixes())
    assert base != block_contribution.cache_key(
        _meta(), ["a photo"], [*prefixes(), "transformer.mlp.0.proj"]
    )


def test_the_key_does_not_depend_on_layer_order():
    """The layer set is a set; a loader that returns it in a different order is
    not a different capture."""
    prompts = ["a photo"]
    forward = block_contribution.cache_key(_meta(), prompts, prefixes())
    backward = block_contribution.cache_key(_meta(), prompts, list(reversed(prefixes())))
    assert forward == backward


def test_a_foreign_safetensors_file_is_not_read_as_a_capture(tmp_path):
    path = tmp_path / "notacapture.safetensors"
    save_file({"a": torch.zeros(2, 2)}, str(path))
    with pytest.raises(ContributionError, match="not a block_contribution capture"):
        block_contribution.load_activations(path)


# --------------------------------------------------------------------- sys.path bootstrap

def test_a_script_that_imports_modules_puts_the_repo_root_on_sys_path():
    """Running a file by path puts only *that file's* directory on sys.path.

    Not the repo root and not the cwd -- so ``from modules.… import …`` inside
    ``scripts/util/`` raises ``ModuleNotFoundError`` unless the script first puts
    the checkout root there itself. ``block_contribution`` did not, and its
    activation-capture path (the only one that loads a model) could not run
    standalone as a result. rehearsal-agent invokes these by path
    (``[ot_python, "-X", "utf8", str(script), *args]``), which is exactly the
    shape that breaks; ``cwd`` being the checkout does not help, because a script
    invocation never puts the cwd on the path.

    Two mechanisms are in use and both are accepted: OT's own ``script_imports()``
    (which also loads ZLUDA on Windows -- right for anything that will touch CUDA)
    and a bare ``parents[2]`` insert (right for a module that is *imported* by
    tests and must not drag ZLUDA in). Checked over the whole directory, because
    the next script to reach for ``modules`` will hit the same wall.
    """
    import ast
    import pathlib as _pathlib

    util = _pathlib.Path(__file__).resolve().parent.parent / "scripts" / "util"
    offenders = []
    for path in sorted(util.glob("*.py")):
        # import_util.py *is* the bootstrap: its own modules.zluda import runs
        # after the insert it exists to perform.
        if path.name == "import_util.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_modules = any(
            (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("modules"))
            or (isinstance(n, ast.Import) and any(a.name.startswith("modules") for a in n.names))
            for n in ast.walk(tree)
        )
        if not imports_modules:
            continue
        bootstraps = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "script_imports"
            for n in ast.walk(tree)
        ) or ("parents[2]" in source or "parent.parent.parent" in source)
        if not bootstraps:
            offenders.append(path.name)
    assert not offenders, (
        "these scripts import modules.* but never put the checkout root on sys.path, so "
        f"they raise ModuleNotFoundError when run by path: {offenders}"
    )


def test_capture_sets_train_config_before_building_the_sampler():
    """A sampling-only consumer must set ``model.train_config``, and set it first.

    Upstream's materialize/evict API (#1617) is how the sampler places each
    model part: every phase materializes the one it needs and evicts the rest.
    Both devices come off the *model's* ``train_config``, which
    ``BaseModel.__init__`` leaves at ``None``. Miss it and the sampler's first
    phase switch raises ``'NoneType' object has no attribute 'temp_device'`` --
    but only after the base model has finished loading, so the cost of finding
    out is the whole expensive half of a capture. ``scripts/sample.py`` sets it
    for the same reason.

    Checked by AST rather than by running the thing: as this module's docstring
    says, the capture half needs Anima and a box. What is provable here is the
    order of two statements, which is exactly what was wrong.
    """
    import ast as _ast
    import pathlib as _pathlib

    source = _pathlib.Path(_here, "../scripts/util/block_contribution.py").read_text(
        encoding="utf-8"
    )
    fn = next(
        n
        for n in _ast.walk(_ast.parse(source))
        if isinstance(n, _ast.FunctionDef) and n.name == "capture_activations"
    )

    assigns = [
        n.lineno
        for n in _ast.walk(fn)
        if isinstance(n, _ast.Assign)
        for t in n.targets
        if isinstance(t, _ast.Attribute) and t.attr == "train_config"
    ]
    samplers = [
        n.lineno
        for n in _ast.walk(fn)
        if isinstance(n, _ast.Call)
        and isinstance(n.func, _ast.Name)
        and n.func.id == "AnimaSampler"
    ]

    assert assigns, (
        "capture_activations never assigns model.train_config -- the sampler reads "
        "its devices from it and BaseModel defaults it to None"
    )
    assert samplers, "expected capture_activations to construct an AnimaSampler"
    assert min(assigns) < min(samplers), (
        "model.train_config is assigned after the sampler is built; the sampler "
        "resolves devices through the model, so it must be set first"
    )
