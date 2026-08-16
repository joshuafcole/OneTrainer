"""Tests for scripts/util/block_gram.py — the primitive everything else is
derived from.

Because the analysis on top is pure linear algebra over these numbers, a wrong
Gram does not fail loudly downstream; it produces a confident eigendecomposition
of the wrong operator. So what is pinned here is the algebra itself: the entries
are the inner products they claim to be, the triangle packs and unpacks to a
symmetric matrix, and the summed Gram equals the Gram of the concatenation.

The last one is the load-bearing identity — run-space PCA is an
eigendecomposition of `total_gram`, and it is only the right operator if
summing per-layer inner products equals taking the inner product of the whole
flattened adapters.

Run with::

    python -m pytest tests/test_block_gram.py -q
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
block_gram = _load("block_gram", "../scripts/util/block_gram.py")

GramError = block_gram.GramError
BLOCK_PREFIX = "transformer.transformer_blocks"
LEAVES = ["attn1.to_q", "attn1.to_v", "attn2.to_q", "attn2.to_v"]
BLOCKS = 4
RANK, DIM = 4, 16


def prefixes(blocks=BLOCKS, leaves=LEAVES):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


def write_adapter(path, deltas_by_prefix, alpha=float(RANK)):
    """A LoRA file whose per-layer delta is exactly the tensor supplied.

    ``delta = (alpha/rank) * up @ down``; an identity-like ``down`` with
    ``alpha == rank`` makes the scale 1 so the fixture's intent and the loader's
    output are the same object."""
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


def unpack(triangle, n):
    """The upper triangle back to a full symmetric matrix — the inverse the
    consumer has to implement, exercised here so its convention is pinned."""
    m = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i, n):
            m[i][j] = m[j][i] = triangle[k]
            k += 1
    return m


def realized(paths):
    return [
        {k: v.delta().to(torch.float64) for k, v in lora_soup.load_lora(p, 1.0).layers.items()}
        for p in paths
    ]


# --- the algebra ------------------------------------------------------------

def test_entries_are_the_inner_products_they_claim_to_be(tmp_path):
    """Checked against directly-computed dot products of the loader's own
    output, per layer and per pair — the definition, not a property of it."""
    paths = make(tmp_path, [1, 2, 3])
    out = block_gram.gram(paths)
    got = realized(paths)
    n = out["adapter_count"]

    assert n == 3
    for entry in out["layers"]:
        m = unpack(entry["gram"], n)
        for i in range(n):
            for j in range(n):
                expected = float(
                    torch.dot(got[i][entry["layer"]].flatten(),
                              got[j][entry["layer"]].flatten())
                )
                assert m[i][j] == pytest.approx(expected, rel=1e-9), (entry["layer"], i, j)


def test_the_diagonal_is_the_squared_norm(tmp_path):
    """No separate norms field exists, so anything that wants a norm reads the
    diagonal. If that were not true the omission would be a silent data loss."""
    paths = make(tmp_path, [4, 5])
    out = block_gram.gram(paths)
    got = realized(paths)
    n = out["adapter_count"]

    for entry in out["layers"]:
        m = unpack(entry["gram"], n)
        for i in range(n):
            norm = float(got[i][entry["layer"]].norm())
            assert m[i][i] == pytest.approx(norm**2, rel=1e-9)


def test_summed_layers_equal_the_gram_of_the_whole_adapters(tmp_path):
    """The identity run-space PCA rests on: eigendecomposing `total_gram` is
    eigendecomposing the covariance of the adapters-as-vectors only if summing
    per-layer inner products equals the inner product of the concatenations.

    Uneven per-layer scales on purpose — with equal norms an implementation
    that averaged instead of summed would agree up to a constant and pass."""
    base = [deltas(s) for s in (6, 7, 8)]
    for d in base:
        for i, prefix in enumerate(sorted(d)):
            d[prefix] = d[prefix] * (1.0 + 3.0 * i)
    paths = [write_adapter(tmp_path / f"a{i}.safetensors", d) for i, d in enumerate(base)]

    out = block_gram.gram(paths)
    got = realized(paths)
    n = out["adapter_count"]
    total = unpack(out["total_gram"], n)

    order = sorted(got[0])
    flat = [torch.cat([g[p].flatten() for p in order]) for g in got]
    for i in range(n):
        for j in range(n):
            assert total[i][j] == pytest.approx(float(torch.dot(flat[i], flat[j])), rel=1e-9)

    # ...and it really is the sum of the per-layer entries, not a second pass.
    summed = [[0.0] * n for _ in range(n)]
    for entry in out["layers"]:
        m = unpack(entry["gram"], n)
        for i in range(n):
            for j in range(n):
                summed[i][j] += m[i][j]
    for i in range(n):
        for j in range(n):
            assert summed[i][j] == pytest.approx(total[i][j], rel=1e-9)


def test_the_gram_is_positive_semidefinite_and_symmetric(tmp_path):
    """Both hold by construction, which is exactly why they are worth checking:
    the consumer eigendecomposes this, and a negative eigenvalue there would be
    read as a real (if small) component rather than as a broken matrix."""
    paths = make(tmp_path, [9, 10, 11, 12])
    out = block_gram.gram(paths)
    n = out["adapter_count"]
    m = torch.tensor(unpack(out["total_gram"], n), dtype=torch.float64)

    assert torch.allclose(m, m.T)
    eigenvalues = torch.linalg.eigvalsh(m)
    assert float(eigenvalues.min()) > -1e-6 * float(eigenvalues.max())


def test_duplicate_adapters_make_the_gram_rank_deficient(tmp_path):
    """The population's real risk in miniature. Three adapters where two are
    identical span two dimensions, and the spectrum has to say so — otherwise a
    chain of resumed runs would look like N independent atoms."""
    d = deltas(13)
    paths = [
        write_adapter(tmp_path / "a0.safetensors", d),
        write_adapter(tmp_path / "a1.safetensors", d),  # identical to a0
        write_adapter(tmp_path / "a2.safetensors", deltas(14)),
    ]
    out = block_gram.gram(paths)
    m = torch.tensor(unpack(out["total_gram"], 3), dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(m)

    assert float(eigenvalues[0]) < 1e-6 * float(eigenvalues[-1])
    assert float(eigenvalues[1]) > 1e-3 * float(eigenvalues[-1])


# --- key sets and refusals ---------------------------------------------------

def test_only_layers_shared_by_ALL_adapters_are_used(tmp_path):
    """Intersection is across every adapter, not pairwise. A layer in 16 of 17
    files still cannot be a coordinate of a 17-way Gram, and quietly using it
    for the pairs that have it would make the matrix inconsistent."""
    paths = [
        write_adapter(tmp_path / "a0.safetensors", deltas(15, blocks=4)),
        write_adapter(tmp_path / "a1.safetensors", deltas(15, blocks=3)),
        write_adapter(tmp_path / "a2.safetensors", deltas(15, blocks=2)),
    ]
    out = block_gram.gram(paths)

    assert out["shared_layer_count"] == 2 * len(LEAVES)
    assert len(out["layers"]) == 2 * len(LEAVES)
    # And what was left out is named, per file that had it.
    assert len(out["dropped_layers"]) == 2
    assert all(str(p) in out["dropped_layers"] for p in paths[:2])
    assert out["block_count"] == 2


def test_a_size_mismatch_on_a_shared_name_is_refused(tmp_path):
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    paths = [
        write_adapter(tmp_path / "a0.safetensors", {prefix: torch.randn(DIM, DIM)}),
        write_adapter(tmp_path / "a1.safetensors", {prefix: torch.randn(DIM * 2, DIM)}),
    ]
    with pytest.raises(GramError) as e:
        block_gram.gram(paths)
    assert "different sizes" in str(e.value)


def test_disjoint_adapters_refuse(tmp_path):
    paths = [
        write_adapter(tmp_path / "a0.safetensors", deltas(16)),
        write_adapter(tmp_path / "a1.safetensors",
                      {f"text_encoder.layers.{i}.mlp.fc1": torch.randn(DIM, DIM)
                       for i in range(2)}),
    ]
    with pytest.raises(GramError) as e:
        block_gram.gram(paths)
    assert "no layers common" in str(e.value)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_adapters_is_not_a_gram(tmp_path, count):
    paths = make(tmp_path, list(range(count))) if count else []
    with pytest.raises(GramError) as e:
        block_gram.gram(paths)
    assert "at least 2" in str(e.value)


def test_every_layer_carries_its_group_label(tmp_path):
    """So a consumer can aggregate to block groups without re-deriving the
    taxonomy — and so the two views cannot silently disagree about which layer
    is where."""
    paths = make(tmp_path, [17, 18])
    out = block_gram.gram(paths, granularity="coarse")
    fitted = block_groups.fit(sorted(prefixes()), block_groups.load_groups(), "coarse")

    for entry in out["layers"]:
        assert entry["layer"] in fitted.groups[entry["group"]]
    assert {e["group"] for e in out["layers"]} == set(fitted.groups)
