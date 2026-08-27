"""The latent / text cache salts, exercised through the dataloader seam.

The property under test is the one that makes reusing a non-cleared cache safe:
**two runs share a cache directory only if every input to the tensors in it is
the same.** So almost every test here is differential -- it builds the real
DiskCache twice through the real seam, changes one thing, and compares the two
directories production chose. Nothing restates a salt.

The last section is not about OneTrainer at all. It pins the two mgds behaviours
this whole mechanism exists to work around, so that if either is ever fixed
upstream the failure lands here, next to the reason.
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.dataLoader.mixin.DataLoaderMgdsMixin import dataset_concepts
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.StableDiffusionBaseDataLoader import StableDiffusionBaseDataLoader
from modules.util.bucket_limits import ANIMA_MAX_BUCKET_RESOLUTION
from modules.util.bucket_tiers import BUCKET_KEEP_NAME, BUCKET_REPEAT_NAME, bucketing_params
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.ModelType import ModelType

from mgds.pipelineModules.DiskCache import DiskCache

import torch

import pytest

QUANTIZATION = 64

_A_TIER = [{"max_size": "8", "strategy": "donate", "mode": "move"}]


class _Loader(DataLoaderText2ImageMixin):
    """The mixin with its abstract members stubbed. Every concrete text2image
    loader routes its caches through the mixin's seam, so exercising the mixin
    directly covers all of them without a model, a VAE or a GPU."""

    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup):
        return []

    def _debug_modules(self, config, model):
        return []


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.STABLE_DIFFUSION_15
    config.base_model_name = "/models/sd15.safetensors"
    config.cache_dir = "/workspace/cache"
    config.batch_size = 4
    config.multi_gpu = False
    config.latent_caching = True
    config.aspect_ratio_bucketing = True
    config.concepts = []  # resolved in memory, so no concept file is read
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _concept(path: str, **overrides) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.path = path
    for key, value in overrides.items():
        setattr(concept, key, value)
    return concept


def _caches(config: TrainConfig, max_resolution: int | None = None) -> tuple[DiskCache, DiskCache]:
    """The image and text DiskCache the real seam builds for ``config``.

    Primes the two values _create_dataset hands the seam, using the same
    production helpers _create_dataset uses, so the salt is chosen end to end by
    the code under test.
    """
    loader = _Loader()
    loader._cache_bucketing = bucketing_params(
        config, quantization=QUANTIZATION, batch_size=config.batch_size, max_resolution=max_resolution)
    loader._cache_concepts = dataset_concepts(config, is_validation=False)

    modules = loader._cache_modules_from_names(
        model=None,
        model_setup=None,
        image_split_names=["latent_image"],
        image_aggregate_names=["crop_resolution", "image_path"],
        text_split_names=["tokens", "text_encoder_hidden_state"],
        sort_names=["tokens", "crop_resolution", "image_path", "latent_image", "prompt", "concept"],
        config=config,
        text_caching=True,
        before_cache_image_fun=lambda: None,
    )
    caches = [m for m in modules if isinstance(m, DiskCache)]
    image = [c for c in caches if os.path.basename(os.path.dirname(c.cache_dir)) == "image"]
    text = [c for c in caches if os.path.basename(os.path.dirname(c.cache_dir)) == "text"]
    assert len(image) == 1 and len(text) == 1
    return image[0], text[0]


def _dirs(config: TrainConfig, max_resolution: int | None = None) -> tuple[str, str]:
    image, text = _caches(config, max_resolution)
    return image.cache_dir, text.cache_dir


def _image_dir(config: TrainConfig, max_resolution: int | None = None) -> str:
    return _dirs(config, max_resolution)[0]


def _text_dir(config: TrainConfig) -> str:
    return _dirs(config)[1]


# --- shape -----------------------------------------------------------------

def test_each_cache_nests_one_salt_directory_under_its_own():
    config = _config()
    image_dir, text_dir = _dirs(config)

    for cache_dir, kind in ((image_dir, "image"), (text_dir, "text")):
        parent, salt = os.path.split(cache_dir)
        assert parent == os.path.join(config.cache_dir, kind)
        assert len(salt) == 16 and all(c in "0123456789abcdef" for c in salt), salt

    # The two caches hold different tensors and must never share a directory.
    assert os.path.basename(image_dir) != os.path.basename(text_dir)


def test_the_same_config_always_lands_in_the_same_directories():
    assert _dirs(_config()) == _dirs(_config())


def test_the_seam_refuses_to_guess_when_it_was_not_wired():
    loader = _Loader()  # never went through _create_dataset
    with pytest.raises(RuntimeError, match="_create_dataset"):
        loader._cache_modules_from_names(
            model=None, model_setup=None,
            image_split_names=["latent_image"], image_aggregate_names=[],
            text_split_names=["tokens"], sort_names=["concept"],
            config=_config(), text_caching=True, before_cache_image_fun=lambda: None,
        )


# --- model identity --------------------------------------------------------

def test_the_image_cache_tracks_the_vae_and_the_resolution():
    baseline = _image_dir(_config())
    assert _image_dir(_config(resolution="768")) != baseline

    other_vae = _config()
    other_vae.vae.model_name = "/models/fixed-vae.safetensors"
    assert _image_dir(other_vae) != baseline

    # an empty vae.model_name means "use the base model's VAE", so the base
    # model is part of the latent's identity
    assert _image_dir(_config(base_model_name="/models/other.safetensors")) != baseline


def test_the_image_cache_ignores_text_only_changes():
    baseline = _image_dir(_config())
    other_encoder = _config()
    other_encoder.text_encoder.model_name = "/models/other-clip"
    assert _image_dir(other_encoder) == baseline


def test_the_text_cache_tracks_the_encoders_and_the_embeddings():
    baseline = _text_dir(_config())

    other_encoder = _config()
    other_encoder.text_encoder.model_name = "/models/other-clip"
    assert _text_dir(other_encoder) != baseline

    assert _text_dir(_config(text_encoder_layer_skip=2)) != baseline
    assert _text_dir(_config(text_encoder_2_sequence_length=256)) != baseline

    added_embedding = _config()
    added_embedding.embedding.model_name = "/embeddings/token.safetensors"
    assert _text_dir(added_embedding) != baseline

    more_tokens = _config()
    more_tokens.embedding.model_name = "/embeddings/token.safetensors"
    more_tokens.embedding.token_count = 4
    assert _text_dir(more_tokens) not in (baseline, _text_dir(added_embedding))


def test_the_text_cache_ignores_excluded_encoders_and_image_changes():
    baseline = _text_dir(_config())

    excluded = _config()
    excluded.text_encoder_3.include = False
    excluded.text_encoder_3.model_name = "/models/garbage"
    assert _text_dir(excluded) == _text_dir(_config(text_encoder_3=excluded.text_encoder_3))

    assert _text_dir(_config(resolution="768")) == baseline
    other_vae = _config()
    other_vae.vae.model_name = "/models/fixed-vae.safetensors"
    assert _text_dir(other_vae) == baseline


# --- the cached item's shape ----------------------------------------------

def test_adding_a_first_bucket_tier_moves_the_image_cache():
    """The reason this slice blocks on bucketing.

    The keep/repeat tags are added to the DiskCache's aggregate_names the moment a
    tier exists. get_item reads those names out of the cached item without
    checking, and the cache's only staleness test is row count -- so reusing a
    pre-tier cache raises KeyError from inside the dataloader. The salt is what
    stops that, and it only works if the *names* the cache is built with are part
    of it: this asserts the two facts together.
    """
    without_tiers, _ = _caches(_config())
    with_tier, _ = _caches(_config(aspect_ratio_bucket_min_tiers=_A_TIER))

    assert BUCKET_KEEP_NAME not in without_tiers.aggregate_names
    assert {BUCKET_KEEP_NAME, BUCKET_REPEAT_NAME} <= set(with_tier.aggregate_names)
    assert with_tier.cache_dir != without_tiers.cache_dir, \
        "a cache whose aggregate names grew must not be reused"


def test_turning_masked_training_on_moves_the_image_cache():
    """Split names take the same unguarded path as aggregate names, and a concrete
    loader derives them from config. Driven through StableDiffusionBaseDataLoader
    so the names come from the loader's own rules, not from this test."""
    def caches(masked: bool) -> DiskCache:
        config = _config(masked_training=masked)
        loader = object.__new__(StableDiffusionBaseDataLoader)
        loader._cache_bucketing = bucketing_params(
            config, quantization=QUANTIZATION, batch_size=config.batch_size)
        loader._cache_concepts = []
        modules = StableDiffusionBaseDataLoader._cache_modules(loader, config, None, None)
        image = [m for m in modules
                 if isinstance(m, DiskCache) and os.path.basename(os.path.dirname(m.cache_dir)) == "image"]
        assert len(image) == 1
        return image[0]

    unmasked, masked = caches(False), caches(True)
    assert "latent_mask" not in unmasked.split_names
    assert "latent_mask" in masked.split_names
    assert masked.cache_dir != unmasked.cache_dir, \
        "a cache whose split names grew must not be reused"


# --- bucket geometry -------------------------------------------------------

def test_the_image_cache_tracks_every_bucket_dimension():
    baseline = _image_dir(_config())

    assert _image_dir(_config(aspect_ratio_bucketing=False)) != baseline
    assert _image_dir(_config(aspect_ratio_bucket_tolerance=0.1)) != baseline
    assert _image_dir(_config(aspect_ratio_bucket_resolution_mode="rotate")) != baseline
    assert _image_dir(_config(aspect_ratio_bucket_min_tiers=_A_TIER)) != baseline
    # the per-dataloader long-edge cap is a constructor argument, not a config
    # field, and it changes the crop of every extreme rung
    assert _image_dir(_config(), max_resolution=ANIMA_MAX_BUCKET_RESOLUTION) != baseline


def test_the_text_cache_ignores_bucketing():
    """Bucketing never reaches a text encoder. Its one route into the text
    pipeline is borrow-copy, which appends rows -- and a changed row count is what
    the DiskCache's own length check already rebuilds on."""
    baseline = _text_dir(_config())
    assert _text_dir(_config(aspect_ratio_bucketing=False)) == baseline
    assert _text_dir(_config(aspect_ratio_bucket_min_tiers=_A_TIER)) == baseline
    assert _text_dir(_config(aspect_ratio_bucket_tolerance=0.1)) == baseline


def test_bucket_settings_are_inert_while_bucketing_is_off():
    """A user who never turns bucketing on must not re-encode because they looked
    at the tier editor: none of these values reach the pipeline."""
    off = _image_dir(_config(aspect_ratio_bucketing=False))
    assert _image_dir(_config(aspect_ratio_bucketing=False, aspect_ratio_bucket_tolerance=0.1)) == off
    assert _image_dir(_config(aspect_ratio_bucketing=False, aspect_ratio_bucket_min_tiers=_A_TIER)) == off


def test_batch_size_only_counts_once_a_tier_exists():
    """batch_size is an input to the rebalancing planner and to nothing else, and
    the planner is a no-op without tiers -- so changing it must not re-encode a
    dataset that has no tiers, and must re-plan one that has."""
    assert _image_dir(_config(batch_size=8)) == _image_dir(_config(batch_size=4))

    tiered = {"aspect_ratio_bucket_min_tiers": _A_TIER}
    assert _image_dir(_config(batch_size=8, **tiered)) != _image_dir(_config(batch_size=4, **tiered))


def test_an_equivalent_tier_list_is_the_same_cache():
    """The planner sorts tiers by max_size and ignores rows with no max_size, so
    neither reordering the list nor leaving a blank row in the editor changes a
    single crop."""
    two_tiers = [
        {"max_size": "8", "strategy": "donate", "mode": "move"},
        {"max_size": "16", "strategy": "drop", "mode": "move"},
    ]
    reordered = list(reversed(two_tiers))
    with_blank_row = [*two_tiers, {"max_size": "", "strategy": ""}]

    baseline = _image_dir(_config(aspect_ratio_bucket_min_tiers=two_tiers))
    assert _image_dir(_config(aspect_ratio_bucket_min_tiers=reordered)) == baseline
    assert _image_dir(_config(aspect_ratio_bucket_min_tiers=with_blank_row)) == baseline


# --- dataset content -------------------------------------------------------

def _write(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(content, bytes) else "w") as fh:
        fh.write(content)


def test_an_added_image_moves_both_caches():
    """The failure the fingerprint exists for: a concept gains a file at a path
    that did not change, and mgds' group key cannot tell."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = os.path.join(tmp_dir, "concept")
        _write(os.path.join(concept_dir, "a.png"), b"image a")
        _write(os.path.join(concept_dir, "a.txt"), "caption a")
        config = _config(concepts=[_concept(concept_dir)])
        before = _dirs(config)

        _write(os.path.join(concept_dir, "b.png"), b"image b")
        after = _dirs(config)

        assert before[0] != after[0], "the latent cache must not be reused"
        assert before[1] != after[1], "the text cache must not be reused either"


def test_a_caption_edit_moves_only_the_text_cache():
    """The split that makes the fingerprint affordable: rewording a caption must
    not push a whole dataset back through the VAE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = os.path.join(tmp_dir, "concept")
        _write(os.path.join(concept_dir, "a.png"), b"image a")
        _write(os.path.join(concept_dir, "a.txt"), "a cat")
        config = _config(concepts=[_concept(concept_dir)])
        before = _dirs(config)

        _write(os.path.join(concept_dir, "a.txt"), "a dog")  # same length on purpose
        after = _dirs(config)

        assert before[1] != after[1], "the text cache must be re-encoded"
        assert before[0] == after[0], "the latent cache must be reused untouched"


def test_the_concepts_come_from_the_concept_file_when_the_field_is_unset():
    """config.concepts is None in the normal UI and CLI paths -- the concepts live
    in a JSON file. A salt that read the field would fingerprint nothing at all,
    which is the whole feature silently switched off."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = os.path.join(tmp_dir, "concept")
        _write(os.path.join(concept_dir, "a.png"), b"image a")
        concept_file = os.path.join(tmp_dir, "concepts.json")
        _write(concept_file, json.dumps([{"path": concept_dir, "enabled": True}]))

        from_file = _config(concepts=None, concept_file_name=concept_file)
        assert _image_dir(from_file) == _image_dir(_config(concepts=[_concept(concept_dir)]))
        assert _image_dir(from_file) != _image_dir(_config(concepts=[]))


def test_a_validation_concept_does_not_move_the_training_cache():
    """_create_mgds feeds a dataset only the concepts of the matching type, so the
    salt must fingerprint the same subset -- otherwise editing a validation set
    re-encodes the training latents."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train")
        validation_dir = os.path.join(tmp_dir, "validation")
        _write(os.path.join(train_dir, "a.png"), b"image a")
        _write(os.path.join(validation_dir, "v.png"), b"image v")

        training_only = _config(concepts=[_concept(train_dir)])
        with_validation = _config(concepts=[
            _concept(train_dir),
            _concept(validation_dir, type=ConceptType.VALIDATION),
        ])
        assert _image_dir(with_validation) == _image_dir(training_only)


# --- the mgds behaviours this exists to work around ------------------------

def _prebuilt_cache(tmp_dir: str, aggregate_names: list[str], split_item: dict | None = None) -> str:
    """A cache directory holding two rows, written the way DiskCache writes them."""
    cache_dir = os.path.join(tmp_dir, "g", "variation-0")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save([{name: f"row{i}-{name}" for name in aggregate_names} for i in range(2)],
               os.path.join(cache_dir, "aggregate.pt"))
    if split_item is not None:
        for i in range(2):
            torch.save(split_item, os.path.join(cache_dir, f"{i}.pt"))
    return cache_dir


def _started_cache(tmp_dir: str, split_names: list[str], aggregate_names: list[str]) -> DiskCache:
    cache = DiskCache(cache_dir=tmp_dir, split_names=split_names, aggregate_names=aggregate_names)
    cache.pipeline = types.SimpleNamespace(device=torch.device("cpu"))
    # the state __init_variations would have built for one group of two rows
    cache.group_variations = {"g": 1}
    cache.group_indices = {"g": [0, 1]}
    cache.group_output_samples = {"g": 2}
    cache.variations_initialized = True
    cache.current_variation = 0
    cache.start(0)
    return cache


def test_mgds_reuses_a_cache_whose_names_changed_and_then_raises():
    """Why the salt carries the DiskCache's own name sets.

    mgds' staleness check is the row count, so a name set that grew under an
    unchanged population is invisible to it, and get_item then indexes a key that
    was never written. If mgds ever guards this, these assertions fail and the
    salt's `names` field can be reconsidered -- that is what they are for.
    """
    old_names = ["crop_resolution", "image_path"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        _prebuilt_cache(tmp_dir, old_names, split_item={"latent_image": "x"})

        reused = _started_cache(tmp_dir, [], [*old_names, BUCKET_KEEP_NAME, BUCKET_REPEAT_NAME])
        assert reused.aggregate_cache["g"][0] is not None, "mgds did not consider the cache stale"
        with pytest.raises(KeyError, match=BUCKET_KEEP_NAME):
            reused.get_item(0, BUCKET_KEEP_NAME)

        # the same hazard on the split side, which is how masked_training reaches it
        split = _started_cache(tmp_dir, ["latent_image", "latent_mask"], old_names)
        with pytest.raises(KeyError, match="latent_mask"):
            split.get_item(0, "latent_mask")

        # control: an unchanged name set is served, which is the whole point of reuse
        unchanged = _started_cache(tmp_dir, [], old_names)
        assert unchanged.get_item(0, "crop_resolution")["crop_resolution"] == "row0-crop_resolution"
