"""Tests for scripts/util/lora_namespace.py — the namespace every reader goes through.

The failure this exists to prevent does not look like a bug. Two saves of the
same run, in two of OneTrainer's LoRA output formats, share no key at all, so
``block_gram`` intersects their layers, gets nothing, and reports "no layers
common to all adapters — these do not target the same model". That sentence is
about the *model*; the disagreement is about spelling. A merge would be worse
still: nothing to merge is not an error either.

So what is pinned here is that the namespace a file arrived in is not
observable downstream — the same run saved two ways loads to the same layers
with the same deltas — and that a format this cannot honestly translate is
refused by name rather than silently mis-read.

Run with::

    python -m pytest tests/test_lora_namespace.py -q
"""

import importlib.util
import json
import os
import sys

import torch

import pytest
from safetensors.torch import save_file

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lora_namespace = _load("lora_namespace", "../scripts/util/lora_namespace.py")
lora_soup = _load("lora_soup", "../scripts/util/lora_soup.py")
block_groups = _load("block_groups", "../scripts/util/block_groups.py")

NamespaceError = lora_namespace.NamespaceError
SoupError = lora_soup.SoupError

ANIMA_HEADER = {"modelspec.architecture": "Anima/lora", "modelspec.sai_model_spec": "1.0.0"}
RANK, DIM = 4, 16
LEAVES = ["attn1.to_q", "ff.net.0.proj"]
BLOCKS = 2


def canonical_prefixes():
    return [f"transformer.transformer_blocks.{i}.{leaf}" for i in range(BLOCKS) for leaf in LEAVES]


def lora_state(prefixes, seed=0):
    """A plain-LoRA state dict in the canonical namespace, native suffixes."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for i, prefix in enumerate(prefixes):
        state[f"{prefix}.lora_down.weight"] = torch.randn(RANK, DIM, generator=generator)
        state[f"{prefix}.lora_up.weight"] = torch.randn(DIM, RANK, generator=generator)
        state[f"{prefix}.alpha"] = torch.tensor(float(RANK) + i)
    return state


def lokr_state(prefixes, seed=0, factor=4, dim=2):
    """A LoKr state dict in the canonical namespace.

    LoKr is not incidental here. Its factor keys carry no ``lora_up`` sibling,
    so the A/B suffix rename never fires on them: a LoKr save in an A/B format
    is *readable* and merely misnamed, which is exactly how the namespace
    difference reaches ``block_gram`` as an empty intersection rather than as a
    file of unrecognised keys."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for prefix in prefixes:
        state[f"{prefix}.lokr_w1"] = torch.randn(factor, factor, generator=generator)
        state[f"{prefix}.lokr_w2_a"] = torch.randn(factor, dim, generator=generator)
        state[f"{prefix}.lokr_w2_b"] = torch.randn(dim, factor, generator=generator)
        state[f"{prefix}.alpha"] = torch.tensor(float(dim))
    return state


def as_comfy(state):
    """``state`` through the COMFY save path, from the saver's own conversion.

    Built from ``AnimaModel``'s forward tables rather than hand-written native
    names, so this fixture cannot agree with a canonicalization that has drifted
    from what the saver writes."""
    from modules.model.AnimaModel import AnimaModel
    from modules.util.convert_lora_util import convert_lora_suffix_ab, lora_original_conversion
    from modules.util.convert_util import convert
    from modules.util.enum.ModelType import ModelType

    model = AnimaModel(model_type=ModelType.ANIMA)
    component = model.model_type.denoising_model_part()
    state = convert(state, lora_original_conversion(model, model.lora_diffusers_to_comfy()), strict=True)
    state = convert(state, [(component, "diffusion_model")], strict=False)
    return convert_lora_suffix_ab(state, peft_convention=False)


def write(tmp_path, name, state, header=ANIMA_HEADER):
    path = tmp_path / name
    save_file(state, str(path), header)
    return path


class TestTheFamilyIsReadOffTheKeys:
    """Classification is structural and model-free — the common case (a file
    already canonical) must not pay for a model import to find that out."""

    def test_canonical_component_prefixes_are_canonical(self):
        assert lora_namespace.family(f"{p}.lora_down.weight" for p in canonical_prefixes()) == "canonical"

    def test_a_bundled_embedding_does_not_make_a_file_foreign(self):
        keys = [f"{canonical_prefixes()[0]}.alpha", "bundle_emb.tok.qwen"]
        assert lora_namespace.family(keys) == "canonical"

    def test_the_comfy_denoising_prefix_is_comfy(self):
        assert lora_namespace.family(["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"]) == "comfy"

    def test_a_kohya_flattened_file_is_its_own_family(self):
        assert lora_namespace.family(["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"]) == "kohya_flat"

    def test_a_top_segment_no_model_declares_is_foreign(self):
        # ORIGINAL_LORA strips the component prefix, leaving Anima's bare "net." wrapper. Nothing here
        # can tell that from an unknown convention, and saying so is the point.
        assert lora_namespace.family(["net.blocks.0.self_attn.q_proj.lora_A.weight"]) == "foreign"


class TestTheSameRunSavedTwoWaysIsTheSameAdapter:
    """The reported failure, and the property that fixes it."""

    def test_a_comfy_lokr_save_shares_every_layer_with_its_canonical_twin(self, tmp_path):
        canonical = lokr_state(canonical_prefixes())
        old = write(tmp_path, "canonical.safetensors", canonical)
        new = write(tmp_path, "comfy.safetensors", as_comfy(canonical))

        assert sorted(lora_soup.load_lora(old, 1.0).layers) == sorted(lora_soup.load_lora(new, 1.0).layers)
        assert sorted(lora_soup.load_lora(new, 1.0).layers) == canonical_prefixes()

    def test_a_comfy_lora_save_carries_the_same_deltas(self, tmp_path):
        canonical = lora_state(canonical_prefixes())
        old = lora_soup.load_lora(write(tmp_path, "canonical.safetensors", canonical), 1.0)
        new = lora_soup.load_lora(write(tmp_path, "comfy.safetensors", as_comfy(canonical)), 1.0)

        for prefix in canonical_prefixes():
            # exact, not close: the COMFY save is a rename, so the tensors are the same bytes and the
            # per-layer alpha (deliberately distinct per layer here) has to survive the trip too.
            assert torch.equal(old.layers[prefix].delta(), new.layers[prefix].delta())

    def test_a_comfy_save_fits_the_taxonomy_it_would_otherwise_fall_out_of(self, tmp_path):
        # block_groups.json is written in the canonical namespace: without canonicalization every layer
        # of a COMFY file lands in the "unblocked" band under a raw leaf-path group.
        path = write(tmp_path, "comfy.safetensors", as_comfy(lora_state(canonical_prefixes())))
        assert block_groups.read_layer_prefixes(path) == canonical_prefixes()

        fitted = block_groups.fit(block_groups.read_layer_prefixes(path), block_groups.load_groups(None), "coarse")
        assert fitted.block_count == BLOCKS
        assert not any(name.startswith(block_groups.UNBLOCKED_BAND) for name in fitted.groups)


class TestAFormatThisCannotTranslateIsRefusedByName:
    """A wrong un-flatten yields layer names that look right and group wrong.
    Refusing is the answer that can be acted on."""

    def test_a_kohya_file_names_the_formats_that_would_work(self, tmp_path):
        path = write(tmp_path, "kohya.safetensors", {
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.zeros(RANK, DIM),
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.zeros(DIM, RANK),
        })
        with pytest.raises(SoupError) as excinfo:
            lora_soup.load_lora(path, 1.0)
        assert "kohya" in str(excinfo.value)
        assert "COMFY_LORA" in str(excinfo.value)

    def test_a_comfy_file_with_no_provenance_says_what_is_missing(self, tmp_path):
        path = write(tmp_path, "anonymous.safetensors", as_comfy(lora_state(canonical_prefixes())), header={})
        with pytest.raises(SoupError) as excinfo:
            lora_soup.load_lora(path, 1.0)
        assert "modelspec.architecture" in str(excinfo.value)

    def test_the_train_config_identifies_the_model_when_the_spec_does_not(self, tmp_path):
        header = {"ot_config": json.dumps({"model_type": "ANIMA"})}
        path = write(tmp_path, "configured.safetensors", as_comfy(lora_state(canonical_prefixes())), header=header)
        assert sorted(lora_soup.load_lora(path, 1.0).layers) == canonical_prefixes()


class TestTheSuffixConventionIsNotTheAdapter:
    """A/B is a spelling of the same factors, and DIFFUSERS folds the alpha away."""

    def test_an_ab_suffix_file_in_canonical_names_needs_no_model_at_all(self, tmp_path):
        canonical = lora_state(canonical_prefixes())
        folded = {}
        for prefix in canonical_prefixes():
            alpha = float(canonical[f"{prefix}.alpha"])
            folded[f"{prefix}.lora_A.weight"] = canonical[f"{prefix}.lora_down.weight"]
            # DIFFUSERS folds alpha/rank into lora_B and drops the .alpha key; the load-time scale is then
            # alpha == rank, i.e. 1.0, and the delta must come back unchanged.
            folded[f"{prefix}.lora_B.weight"] = canonical[f"{prefix}.lora_up.weight"] * (alpha / RANK)

        loaded = lora_soup.load_lora(write(tmp_path, "diffusers.safetensors", folded), 1.0)
        reference = lora_soup.load_lora(write(tmp_path, "canonical.safetensors", canonical), 1.0)
        for prefix in canonical_prefixes():
            # atol, not the default: the fold is lossless in exact arithmetic but reorders the float32
            # multiply ((alpha/rank)*up)@down vs (alpha/rank)*(up@down), and the last bit moves.
            assert torch.allclose(
                loaded.layers[prefix].delta(), reference.layers[prefix].delta(), atol=1e-5)


class TestTheKeyOnlyPathAgreesWithTheTensorPath:
    """``block_groups`` reads a key set without materializing a tensor; it must
    still see the layers the merge engine would."""

    def test_both_paths_report_the_same_prefixes(self, tmp_path):
        for name, state in (("lora", lora_state(canonical_prefixes())), ("lokr", lokr_state(canonical_prefixes()))):
            path = write(tmp_path, f"{name}.safetensors", as_comfy(state))
            assert block_groups.read_layer_prefixes(path) == sorted(lora_soup.load_lora(path, 1.0).layers)


class TestNativizeWritesWhatTheSaverWouldHaveWritten:
    """The forward direction, checked against the saver rather than against
    itself.

    ``as_comfy`` is built from ``AnimaModel``'s own tables through
    ``LoRASaverMixin._save_comfy``'s steps, so agreeing with it is the only
    claim worth making: a nativize that agreed only with ``canonicalize`` would
    be self-consistent and could still be writing names ComfyUI has never seen.
    """

    def test_a_diffusers_lora_becomes_exactly_the_comfy_save(self):
        src = lora_state(canonical_prefixes(), seed=3)
        mine = lora_namespace.nativize(dict(src), ANIMA_HEADER, "<lora>")
        theirs = as_comfy(dict(src))
        assert set(mine) == set(theirs)
        for key in mine:
            assert torch.allclose(mine[key], theirs[key], atol=1e-5)

    def test_a_lokr_too(self):
        # LoKr is the case a suffix-only conversion silently gets wrong: its
        # factors have no ``lora_up`` sibling, so the A/B rename never fires on
        # them and only the *namespace* half of the work is visible.
        src = lokr_state(canonical_prefixes(), seed=4)
        mine = lora_namespace.nativize(dict(src), ANIMA_HEADER, "<lokr>")
        theirs = as_comfy(dict(src))
        assert set(mine) == set(theirs)
        for key in mine:
            assert torch.allclose(mine[key], theirs[key], atol=1e-5)

    def test_every_denoising_key_carries_the_prefix_comfy_matches(self):
        # The whole point of the conversion: ComfyUI matches by name, and a file
        # whose keys it cannot match loads zero of them without erroring.
        out = lora_namespace.nativize(
            lora_state(canonical_prefixes()), ANIMA_HEADER, "<lora>"
        )
        assert out
        assert all(key.startswith("diffusion_model.") for key in out)
        assert not any("transformer_blocks." in key for key in out)

    def test_nativize_then_canonicalize_is_the_identity_on_names_and_values(self):
        src = lora_state(canonical_prefixes(), seed=5)
        native = lora_namespace.nativize(dict(src), ANIMA_HEADER, "<lora>")
        there_and_back = lora_namespace.canonicalize(dict(native), ANIMA_HEADER, "<lora>")
        straight = lora_namespace.canonicalize(dict(src), ANIMA_HEADER, "<lora>")
        assert set(there_and_back) == set(straight)
        for key in straight:
            assert torch.allclose(there_and_back[key], straight[key], atol=1e-5)

    def test_a_file_already_native_survives_a_second_pass(self):
        # The failure the fork converter had: it prefixed an already-native key a
        # second time (``diffusion_model.diffusion_model.blocks.…``), matched
        # nothing, and exited 0. Going through canonical first is what makes this
        # idempotent rather than merely lucky.
        native = as_comfy(lora_state(canonical_prefixes(), seed=6))
        again = lora_namespace.nativize(dict(native), ANIMA_HEADER, "<lora>")
        assert set(again) == set(native)
        assert not any(key.startswith("diffusion_model.diffusion_model.") for key in again)

    def test_a_header_that_names_no_model_is_refused_by_name(self):
        with pytest.raises(lora_namespace.NamespaceError, match="which model it was trained for"):
            lora_namespace.nativize(
                lora_state(canonical_prefixes()), {}, "<no header>"
            )

    def test_a_kohya_file_is_refused_rather_than_guessed_at(self):
        state = {"lora_unet_blocks_0_attn_q.lora_down.weight": torch.zeros(2, 2)}
        with pytest.raises(lora_namespace.NamespaceError, match="kohya"):
            lora_namespace.nativize(state, ANIMA_HEADER, "<kohya>")

