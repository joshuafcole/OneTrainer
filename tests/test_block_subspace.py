"""Tests for scripts/util/block_subspace.py — the measure that survives the
objection block_gram cannot answer.

The whole reason this script exists is that a whole-vector cosine of ~0 does not
mean two adapters learned unrelated things: the bulk of a LoRA delta is training
noise, and two runs can share their leading directions exactly while the inner
product of the flattened deltas reports nothing. So the centrepiece here is
`test_a_shared_top_direction_is_invisible_to_the_inner_product`, which builds
that exact case — φ at k=1 is 1.0 while the Gram cosine is 0.

The second load-bearing test is the null. φ is read against a floor of k/m, not
against zero, and every conclusion drawn from this tool is a comparison to that
floor. `test_independent_random_subspaces_sit_at_the_analytic_floor` is what
makes the floor a measured fact rather than a claim in a docstring.

Third is `test_the_randomized_basis_agrees_with_a_dense_svd`, which guards the
one performance shortcut taken: svd_lowrank instead of a full decomposition.

Run with::

    python -m pytest tests/test_block_subspace.py -q
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
block_subspace = _load("block_subspace", "../scripts/util/block_subspace.py")

SubspaceError = block_subspace.SubspaceError
BLOCK_PREFIX = "transformer.transformer_blocks"
LEAVES = ["attn1.to_q", "attn1.to_v"]
BLOCKS = 2


def prefixes(blocks=BLOCKS, leaves=LEAVES):
    return [f"{BLOCK_PREFIX}.{i}.{leaf}" for i in range(blocks) for leaf in leaves]


def write_lora(path, factors_by_prefix):
    """A LoRA whose per-layer delta is exactly ``up @ down``.

    ``alpha == rank`` makes the loader's ``alpha/rank`` scale exactly 1, so the
    fixture's factors and the realised delta are the same object — which is what
    lets a test name a layer's singular vectors instead of inferring them."""
    state = {}
    for prefix, (up, down) in factors_by_prefix.items():
        rank = down.shape[0]
        assert up.shape[1] == rank, (up.shape, down.shape)
        state[prefix + ".lora_down.weight"] = down.contiguous()
        state[prefix + ".lora_up.weight"] = up.contiguous()
        state[prefix + ".alpha"] = torch.tensor(float(rank))
    save_file(state, str(path))
    return path


def orthonormal(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=g))
    return q[:, :cols].contiguous()


def factored(u, singular_values, v):
    """``(up, down)`` for a delta with exactly these singular vectors and values.

    ``delta = up @ down = (U diag(s)) Vᵀ``, so the caller names the left basis,
    the spectrum and the right basis directly. Values must be descending or the
    "top-k" the script slices is not the k the test meant."""
    s = torch.tensor(singular_values, dtype=torch.float32)
    assert all(a >= b for a, b in zip(singular_values, singular_values[1:]))
    return (u * s, v.T)


def unpack(triangle, n):
    m = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i, n):
            m[i][j] = m[j][i] = triangle[k]
            k += 1
    return m


def phi(out, layer_index, side, k_index, n):
    return unpack(out["layers"][layer_index][f"phi_{side}"][k_index], n)


def simple(tmp_path, count, rank=4, dim=16, seed0=0):
    """``count`` adapters with independent random rank-``rank`` deltas."""
    paths = []
    for i in range(count):
        factors = {}
        for j, p in enumerate(prefixes()):
            seed = seed0 + 100 * i + j
            factors[p] = factored(
                orthonormal(dim, rank, seed),
                list(range(rank, 0, -1)),
                orthonormal(dim, rank, seed + 50),
            )
        paths.append(write_lora(tmp_path / f"a{i}.safetensors", factors))
    return paths


# --- what φ measures that a cosine does not ---------------------------------

def test_a_shared_top_direction_is_invisible_to_the_inner_product(tmp_path):
    """The case this whole script exists for, built exactly.

    Two adapters whose left bases share their *first* column and nothing else,
    and whose right bases share nothing. The shared right-space of zero makes
    every term of ⟨dW_a, dW_b⟩ vanish, so the Gram reports a cosine of exactly 0
    — while φ_left at k=1 is 1.0, because the leading direction is identical.

    A population read only through block_gram would call these two adapters
    unrelated. They are not.
    """
    dim, rank = 32, 4
    left = orthonormal(dim, 8, 1)
    right_a, right_b = orthonormal(dim, rank, 2), orthonormal(dim, rank, 3)
    # Right bases from independent draws are not exactly orthogonal, so make the
    # cosine-is-zero claim exact by forcing it: project b's right basis off a's.
    residual = right_b - right_a @ (right_a.T @ right_b)
    right_b, _ = torch.linalg.qr(residual)
    right_b = right_b[:, :rank].contiguous()

    spectrum = [4.0, 3.0, 2.0, 1.0]
    a = {p: factored(left[:, :rank], spectrum, right_a) for p in prefixes()}
    b = {p: factored(left[:, [0, 4, 5, 6]], spectrum, right_b) for p in prefixes()}
    paths = [
        write_lora(tmp_path / "a.safetensors", a),
        write_lora(tmp_path / "b.safetensors", b),
    ]

    out = block_subspace.subspace(paths, k_values=[1, 2, 4])
    assert out["k_values"] == [1, 2, 4]
    for entry_index in range(len(out["layers"])):
        # k=1: the same direction, so complete overlap.
        assert phi(out, entry_index, "left", 0, 2)[0][1] == pytest.approx(1.0, abs=1e-6)
        # k=2: one shared direction out of two -> 1/2. k=4 -> 1/4.
        assert phi(out, entry_index, "left", 1, 2)[0][1] == pytest.approx(0.5, abs=1e-5)
        assert phi(out, entry_index, "left", 2, 2)[0][1] == pytest.approx(0.25, abs=1e-5)
        # The right spaces were made exactly orthogonal.
        for k_index in range(3):
            assert phi(out, entry_index, "right", k_index, 2)[0][1] == pytest.approx(
                0.0, abs=1e-5
            )

    # ...and the measure that misses it, on the very same files.
    gram = unpack(block_gram.gram(paths)["total_gram"], 2)
    cosine = gram[0][1] / (gram[0][0] * gram[1][1]) ** 0.5
    assert cosine == pytest.approx(0.0, abs=1e-6)


def test_independent_random_subspaces_sit_at_the_analytic_floor(tmp_path):
    """φ is read against k/m, and here is the k/m.

    Two independent uniformly-random k-dimensional subspaces of ℝ^m satisfy
    E‖UᵀU'‖_F² = k²/m, so E[φ] = k/m — emphatically not 0. At m=512, k=4 that
    floor is 0.0078, which is the same order as the whole-adapter cosines that
    motivated this measurement. Reporting φ without it would repeat the mistake
    the tool was built to correct.
    """
    dim, rank = 512, 4
    paths = simple(tmp_path, 3, rank=rank, dim=dim, seed0=1000)
    out = block_subspace.subspace(paths, k_values=[rank])

    values = []
    for entry in out["layers"]:
        assert entry["rows"] == dim and entry["cols"] == dim
        m = unpack(entry["phi_left"][0], 3)
        r = unpack(entry["phi_right"][0], 3)
        values += [m[i][j] for i in range(3) for j in range(i + 1, 3)]
        values += [r[i][j] for i in range(3) for j in range(i + 1, 3)]

    floor = rank / dim
    mean = sum(values) / len(values)
    assert mean == pytest.approx(floor, abs=0.004), (mean, floor)
    # The point of the test, stated as an assertion: the floor is not zero, and
    # it is far enough above zero to be mistaken for a signal.
    assert mean > 0.5 * floor


def test_phi_ignores_scale_and_sign_where_a_cosine_cannot(tmp_path):
    """A subspace has no magnitude and no direction along itself.

    3x the same adapter and -1x the same adapter are both φ=1 everywhere,
    against a Gram cosine of +1 and -1 respectively. That is the intended
    difference: for merging, "same subspace at a different strength" and "same
    subspace inverted" are both alignment, and the coefficient search — not the
    geometry — is what decides the sign and scale.
    """
    dim, rank = 16, 4
    base = {
        p: factored(orthonormal(dim, rank, 7 + j), [4.0, 3.0, 2.0, 1.0],
                    orthonormal(dim, rank, 70 + j))
        for j, p in enumerate(prefixes())
    }
    scaled = {p: (up * 3.0, down) for p, (up, down) in base.items()}
    negated = {p: (-up, down) for p, (up, down) in base.items()}
    paths = [
        write_lora(tmp_path / "a.safetensors", base),
        write_lora(tmp_path / "b.safetensors", scaled),
        write_lora(tmp_path / "c.safetensors", negated),
    ]

    out = block_subspace.subspace(paths, k_values=[1, 4])
    for entry_index in range(len(out["layers"])):
        for k_index in range(2):
            for side in ("left", "right"):
                m = phi(out, entry_index, side, k_index, 3)
                for i in range(3):
                    for j in range(3):
                        assert m[i][j] == pytest.approx(1.0, abs=1e-5), (i, j, side)

    gram = unpack(block_gram.gram(paths)["total_gram"], 3)
    def cos(i, j):
        return gram[i][j] / (gram[i][i] * gram[j][j]) ** 0.5
    assert cos(0, 1) == pytest.approx(1.0, abs=1e-6)
    assert cos(0, 2) == pytest.approx(-1.0, abs=1e-6)


def test_orthogonal_subspaces_report_zero(tmp_path):
    dim, rank = 32, 4
    basis = orthonormal(dim, 2 * rank, 11)
    right = orthonormal(dim, 2 * rank, 12)
    spectrum = [4.0, 3.0, 2.0, 1.0]
    a = {p: factored(basis[:, :rank], spectrum, right[:, :rank]) for p in prefixes()}
    b = {p: factored(basis[:, rank:], spectrum, right[:, rank:]) for p in prefixes()}
    paths = [
        write_lora(tmp_path / "a.safetensors", a),
        write_lora(tmp_path / "b.safetensors", b),
    ]

    out = block_subspace.subspace(paths, k_values=[1, 4])
    for entry_index in range(len(out["layers"])):
        for k_index in range(2):
            for side in ("left", "right"):
                assert phi(out, entry_index, side, k_index, 2)[0][1] == pytest.approx(
                    0.0, abs=1e-5
                )


# --- the arithmetic and the shortcut ----------------------------------------

def test_the_diagonal_is_one_because_a_basis_is_orthonormal(tmp_path):
    """The triangle's diagonal is φ(i, i, k), which is 1 by construction — and
    is therefore the only in-band evidence that svd_lowrank returned an
    orthonormal basis rather than something merely close to one."""
    paths = simple(tmp_path, 4)
    out = block_subspace.subspace(paths, k_values=[1, 2, 4])
    for entry in out["layers"]:
        for k_index in range(3):
            for side in ("left", "right"):
                m = unpack(entry[f"phi_{side}"][k_index], 4)
                for i in range(4):
                    assert m[i][i] == pytest.approx(1.0, abs=1e-6)


def test_the_randomized_basis_agrees_with_a_dense_svd(tmp_path):
    """Guards the one shortcut taken.

    svd_lowrank is a randomized range finder, not an exact decomposition; it is
    used because a dense SVD over a few hundred layers by a few dozen adapters is
    hours rather than seconds. If it drifted, φ would be quietly wrong with no
    downstream symptom — an eigendecomposition of the wrong operator, again.
    """
    dim, rank = 48, 6
    paths = simple(tmp_path, 3, rank=rank, dim=dim, seed0=2000)
    out = block_subspace.subspace(paths, k_values=[1, 3, 6])

    realized = [lora_soup.load_lora(p, 1.0).layers for p in paths]
    for entry in out["layers"]:
        dense = []
        for layers in realized:
            u, _s, vh = torch.linalg.svd(
                layers[entry["layer"]].delta().to(torch.float64), full_matrices=False
            )
            dense.append((u, vh.T))
        for k_index, k in enumerate(entry["k_effective"]):
            for side, slot in (("left", 0), ("right", 1)):
                got = unpack(entry[f"phi_{side}"][k_index], 3)
                for i in range(3):
                    for j in range(3):
                        overlap = dense[i][slot][:, :k].T @ dense[j][slot][:, :k]
                        want = float((overlap**2).sum()) / k
                        assert got[i][j] == pytest.approx(want, abs=1e-5), (
                            entry["layer"], side, k, i, j
                        )


def test_the_measurement_does_not_move_between_runs(tmp_path):
    """svd_lowrank draws a random test matrix. Unseeded, two runs of the same
    survey would disagree in the third decimal — which is where the floor lives,
    so it would not be a rounding difference, it would be the answer."""
    paths = simple(tmp_path, 3, seed0=3000)
    first = block_subspace.subspace(paths, k_values=[1, 4])
    second = block_subspace.subspace(paths, k_values=[1, 4])
    assert first["layers"] == second["layers"]


def test_shape_rank_and_energy_are_reported_per_layer(tmp_path):
    """The floor is k/rows and k/cols, so a consumer that cannot see rows and
    cols cannot interpret a single number this tool emits. Energy is here so a
    weighted aggregate does not need a second call to block_gram."""
    dim, rank = 24, 3
    paths = simple(tmp_path, 2, rank=rank, dim=dim, seed0=4000)
    out = block_subspace.subspace(paths, k_values=[1, rank])
    realized = [lora_soup.load_lora(p, 1.0).layers for p in paths]

    assert len(out["layers"]) == len(prefixes())
    for entry in out["layers"]:
        assert entry["rows"] == dim
        assert entry["cols"] == dim
        assert entry["ranks"] == [rank, rank]
        for i, layers in enumerate(realized):
            delta = layers[entry["layer"]].delta().to(torch.float64)
            assert entry["energy"][i] == pytest.approx(float((delta * delta).sum()),
                                                       rel=1e-9)


def test_k_is_clamped_to_the_smallest_rank_and_says_so(tmp_path):
    """A rank-2 run in a population of rank-8 runs is a fact about the
    population, not an error. Asking for k=8 there would slice directions one
    adapter does not have; the clamp is reported per k so a caller reading
    phi_left[2] knows which k it actually got."""
    dim = 32
    wide = {
        p: factored(orthonormal(dim, 8, 20 + j), [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    orthonormal(dim, 8, 80 + j))
        for j, p in enumerate(prefixes())
    }
    narrow = {
        p: factored(orthonormal(dim, 2, 30 + j), [2.0, 1.0], orthonormal(dim, 2, 90 + j))
        for j, p in enumerate(prefixes())
    }
    paths = [
        write_lora(tmp_path / "wide.safetensors", wide),
        write_lora(tmp_path / "narrow.safetensors", narrow),
    ]

    out = block_subspace.subspace(paths, k_values=[1, 4, 16])
    for entry in out["layers"]:
        assert entry["ranks"] == [8, 2]
        assert entry["k_effective"] == [1, 2, 2]
        # Clamped entries are genuinely the same measurement, not a copy: k=4
        # and k=16 both became k=2, so they must agree exactly.
        assert entry["phi_left"][1] == entry["phi_left"][2]


def test_every_layer_carries_its_group_label(tmp_path):
    paths = simple(tmp_path, 2, seed0=5000)
    out = block_subspace.subspace(paths, granularity="coarse")
    fitted = block_groups.fit(sorted(prefixes()), block_groups.load_groups(), "coarse")

    for entry in out["layers"]:
        assert entry["layer"] in fitted.groups[entry["group"]]
    assert {e["group"] for e in out["layers"]} == set(fitted.groups)


# --- refusals ----------------------------------------------------------------

def test_only_layers_shared_by_all_adapters_are_used_and_the_rest_are_named(tmp_path):
    dim, rank = 16, 4
    def build(blocks, seed):
        return {
            p: factored(orthonormal(dim, rank, seed + j), [4.0, 3.0, 2.0, 1.0],
                        orthonormal(dim, rank, seed + 60 + j))
            for j, p in enumerate(prefixes(blocks=blocks))
        }
    paths = [
        write_lora(tmp_path / "a0.safetensors", build(2, 40)),
        write_lora(tmp_path / "a1.safetensors", build(1, 40)),
    ]
    out = block_subspace.subspace(paths)

    assert out["shared_layer_count"] == len(LEAVES)
    assert len(out["dropped_layers"]) == 1
    assert str(paths[0]) in out["dropped_layers"]


def test_a_shape_mismatch_on_a_shared_name_is_refused(tmp_path):
    prefix = f"{BLOCK_PREFIX}.0.attn1.to_q"
    paths = [
        write_lora(tmp_path / "a0.safetensors",
                   {prefix: factored(orthonormal(16, 4, 1), [4.0, 3.0, 2.0, 1.0],
                                     orthonormal(16, 4, 2))}),
        write_lora(tmp_path / "a1.safetensors",
                   {prefix: factored(orthonormal(32, 4, 3), [4.0, 3.0, 2.0, 1.0],
                                     orthonormal(16, 4, 4))}),
    ]
    with pytest.raises(SubspaceError) as e:
        block_subspace.subspace(paths)
    assert "different shapes" in str(e.value)


def test_disjoint_adapters_refuse(tmp_path):
    dim, rank = 16, 4
    a = {p: factored(orthonormal(dim, rank, 1 + j), [4.0, 3.0, 2.0, 1.0],
                     orthonormal(dim, rank, 11 + j))
         for j, p in enumerate(prefixes())}
    b = {f"text_encoder.layers.{i}.mlp.fc1":
         factored(orthonormal(dim, rank, 21 + i), [4.0, 3.0, 2.0, 1.0],
                  orthonormal(dim, rank, 31 + i))
         for i in range(2)}
    paths = [
        write_lora(tmp_path / "a0.safetensors", a),
        write_lora(tmp_path / "a1.safetensors", b),
    ]
    with pytest.raises(SubspaceError) as e:
        block_subspace.subspace(paths)
    assert "no layers common" in str(e.value)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_adapters_is_not_a_comparison(tmp_path, count):
    paths = simple(tmp_path, count, seed0=6000) if count else []
    with pytest.raises(SubspaceError) as e:
        block_subspace.subspace(paths)
    assert "at least 2" in str(e.value)


def test_a_k_below_one_is_refused(tmp_path):
    paths = simple(tmp_path, 2, seed0=7000)
    with pytest.raises(SubspaceError) as e:
        block_subspace.subspace(paths, k_values=[0, 4])
    assert "at least 1" in str(e.value)
