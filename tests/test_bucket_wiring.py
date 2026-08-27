"""OneTrainer's half of population-aware aspect-bucket rebalancing.

The planner itself lives in mgds and is tested there. What is only testable here
is the *wiring*: which pipeline modules OneTrainer builds, and with which
arguments. Two claims matter and neither is visible from mgds' own suite.

1. **Turning the feature off leaves the pipeline exactly as it was.** Every
   rebalancing argument is a default and every extra module is absent, so a user
   who never opens the tier editor trains on the same batches as before.

2. **Turning it on reaches the planner.** Tiers, tolerance, resolution mode and
   the pixel cap arrive at the modules that need them, and the two modules that
   must agree on bucket geometry are handed the same numbers by construction.

The disabled-path assertions are written against mgds' own constructor defaults
rather than against literals, so they keep meaning what they say if mgds renames
or re-defaults an argument instead of quietly passing against a moved goalpost.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.bucket_tiers import (
    BUCKET_BUDGET_NAME,
    BUCKET_GROUP_NAME,
    BUCKET_KEEP_NAME,
    BUCKET_OVERRIDE_RUNG_NAME,
    BUCKET_REPEAT_NAME,
    bucketing_params,
    parse_bucket_tiers,
    parse_resolution_mode,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType

from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.AspectBucketRebalance import AspectBucketRebalance
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.VariationSorting import VariationSorting
from mgds.util.bucketRebalancing import BORROW_COPY, CANONICAL_RUNG_ASPECTS, BucketTier, plan_rebalance

QUANTIZATION = 64

_A_TIER = [{"max_size": "8", "strategy": "donate", "mode": "move"}]


class _Loader(DataLoaderText2ImageMixin):
    """The mixin with its abstract members stubbed.

    Every concrete text2image loader routes through the mixin's builders, so
    exercising the mixin directly covers all fifteen of them without needing a
    model, a VAE or a GPU.
    """

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
    config.batch_size = 4
    config.multi_gpu = False
    config.latent_caching = True
    config.aspect_ratio_bucketing = True
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _params(config, max_resolution=None):
    return bucketing_params(
        config, quantization=QUANTIZATION, batch_size=config.batch_size, max_resolution=max_resolution)


def _only(modules, cls):
    found = [m for m in modules if isinstance(m, cls)]
    assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
    return found[0]


def _mgds_default(cls, argument):
    """The default mgds itself declares for a constructor argument."""
    parameter = inspect.signature(cls.__init__).parameters[argument]
    assert parameter.default is not inspect.Parameter.empty, f"{cls.__name__}.{argument} has no default"
    return parameter.default


def _sorter(loader, config):
    modules = loader._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image"],
        config=config,
        before_cache_image_fun=lambda: None,
        autocast_context=[None],
        train_dtype=DataType.FLOAT_32,
    )
    cls = AspectBatchSorting if config.latent_caching else InlineAspectBatchSorting
    return _only(modules, cls)


def _image_cache(cache_modules):
    """The image DiskCache; _cache_modules_from_names also builds a text one."""
    caches = [m for m in cache_modules if isinstance(m, DiskCache) and os.path.basename(m.cache_dir) == "image"]
    assert len(caches) == 1
    return caches[0]


def _cache_modules(loader, config):
    return loader._cache_modules_from_names(
        model=None,
        model_setup=None,
        image_split_names=["latent_image"],
        image_aggregate_names=["crop_resolution", "image_path"],
        text_split_names=["tokens"],
        sort_names=["tokens", "crop_resolution", "image_path", "latent_image", "prompt", "concept"],
        config=config,
        text_caching=True,
        before_cache_image_fun=lambda: None,
    )


# --- the disabled path -----------------------------------------------------

def test_bucketing_off_builds_the_pipeline_it_always_built():
    config = _config(aspect_ratio_bucketing=False)
    loader = _Loader()

    modules = loader._aspect_bucketing_in(config, _params(config))

    assert [type(m) for m in modules] == [CalcAspect, SingleAspectCalculation]
    assert loader._bucket_rebalance_modules(config, _params(config)) == []


def test_bucketing_off_leaves_the_sorter_and_the_cache_untagged():
    config = _config(aspect_ratio_bucketing=False)
    loader = _Loader()

    cache = _cache_modules(loader, config)
    disk_cache = _image_cache(cache)
    variation_sorting = _only(cache, VariationSorting)
    sorter = _sorter(loader, config)

    assert BUCKET_KEEP_NAME not in disk_cache.aggregate_names
    assert BUCKET_REPEAT_NAME not in disk_cache.aggregate_names
    assert BUCKET_KEEP_NAME not in variation_sorting.names
    assert sorter.keep_in_name is None
    assert sorter.repeat_in_name is None


def test_bucketing_on_with_nothing_configured_is_the_previous_behaviour():
    """The default-on aspect bucketing path must not change under a user who never
    touches the new controls, so every rebalancing argument is mgds' own default."""
    config = _config()
    bucketing = AspectBucketing_of(_Loader(), config)

    for argument in ("bucket_aspect_tolerance", "min_bucket_tiers", "keep_out_name",
                     "repeat_out_name", "override_rung_in_name", "resolution_mode", "max_resolution"):
        default = _mgds_default(AspectBucketing, argument)
        actual = getattr(bucketing, argument)
        # mgds normalises None into an empty list for the tier list
        expected = [] if (argument == "min_bucket_tiers" and default is None) else default
        assert actual == expected, f"{argument} is {actual!r}, not mgds' default {expected!r}"


def test_bucketing_on_with_no_tiers_does_not_change_the_latent_cache():
    """The tags are two extra values in every cached item. Emitting them for a feature
    nobody switched on would invalidate every existing cache to record a constant."""
    loader = _Loader()
    config = _config()

    cache = _cache_modules(loader, config)
    sorter = _sorter(loader, config)

    assert _image_cache(cache).aggregate_names == ["crop_resolution", "image_path"]
    assert sorter.keep_in_name is None
    assert sorter.repeat_in_name is None


def test_no_tiers_means_the_planner_changes_nothing_whatever_the_batch_size():
    """batch_size is the one argument the wiring always passes, so pin that it is
    inert until a tier asks for a fill. Otherwise the previous test's claim of "mgds
    defaults everywhere" would have a hole exactly where it matters."""
    aspects = [1.0, 1.0, 1.0, 1.25, 0.5, 2.0, 3.0]

    for batch_size in (1, 2, 4, 8, 64):
        plan = plan_rebalance(aspects, list(CANONICAL_RUNG_ASPECTS), [], batch_size)
        assert all(plan.keep)
        assert plan.override_rung == [None] * len(aspects)
        assert plan.repeat == [1] * len(aspects)
        assert plan.copies == []


def AspectBucketing_of(loader, config, max_resolution=None):
    modules = loader._aspect_bucketing_in(config, _params(config, max_resolution))
    return _only(modules, AspectBucketing)


# --- tiers reach the planner ----------------------------------------------

def test_tiers_reach_the_bucketer_and_the_tags_are_switched_on():
    config = _config(aspect_ratio_bucket_min_tiers=[
        {"max_size": "2", "strategy": "drop", "mode": "move"},
        {"max_size": "8", "strategy": "borrow", "mode": "move"},
    ])
    bucketing = AspectBucketing_of(_Loader(), config)

    assert bucketing.min_bucket_tiers == [
        BucketTier(max_size=2, strategy="drop", mode="move"),
        BucketTier(max_size=8, strategy="borrow", mode="move"),
    ]
    assert bucketing.keep_out_name == BUCKET_KEEP_NAME
    assert bucketing.repeat_out_name == BUCKET_REPEAT_NAME
    assert bucketing.override_rung_in_name is None
    assert bucketing.batch_size == config.batch_size


def test_the_tolerance_reaches_the_bucketer():
    config = _config(aspect_ratio_bucket_tolerance=0.1)
    assert AspectBucketing_of(_Loader(), config).bucket_aspect_tolerance == pytest.approx(0.1)


def test_the_resolution_mode_reaches_the_bucketer():
    config = _config(aspect_ratio_bucket_resolution_mode="rotate")
    assert AspectBucketing_of(_Loader(), config).resolution_mode == "rotate"


def test_borrow_copy_moves_planning_upstream_of_the_image_load():
    """borrow-copy is the one strategy that adds rows, and rows can only be added
    before the images load. So it gets a path-stage planner and the bucketer drops
    to being a consumer of the result -- planning in both places would double it."""
    config = _config(aspect_ratio_bucket_min_tiers=[
        {"max_size": "8", "strategy": "borrow", "mode": BORROW_COPY},
    ])
    loader = _Loader()

    rebalance = _only(loader._bucket_rebalance_modules(config, _params(config)), AspectBucketRebalance)
    bucketing = AspectBucketing_of(loader, config)

    assert rebalance.min_bucket_tiers == [BucketTier(max_size=8, strategy="borrow", mode=BORROW_COPY)]
    assert rebalance.override_rung_out_name == BUCKET_OVERRIDE_RUNG_NAME
    assert bucketing.override_rung_in_name == BUCKET_OVERRIDE_RUNG_NAME
    assert bucketing.min_bucket_tiers == []
    assert bucketing.keep_out_name is None
    assert bucketing.repeat_out_name is None


def test_a_non_copy_tier_needs_no_path_stage_planner():
    config = _config(aspect_ratio_bucket_min_tiers=[{"max_size": "8", "strategy": "borrow", "mode": "move"}])
    assert _Loader()._bucket_rebalance_modules(config, _params(config)) == []


def test_the_planner_and_the_bucketer_are_given_the_same_geometry():
    """They pool aspect rungs using these three numbers. Different values, different
    pooling, and the plan the bucketer executes is not the plan that was made."""
    config = _config(
        aspect_ratio_bucket_tolerance=0.1,
        aspect_ratio_bucket_min_tiers=[{"max_size": "8", "strategy": "borrow", "mode": BORROW_COPY}],
    )
    loader = _Loader()

    rebalance = _only(loader._bucket_rebalance_modules(config, _params(config, 1920)), AspectBucketRebalance)
    bucketing = AspectBucketing_of(loader, config, 1920)

    assert rebalance.quantization == bucketing.quantization == QUANTIZATION
    assert rebalance.bucket_aspect_tolerance == bucketing.bucket_aspect_tolerance
    assert rebalance.max_resolution == bucketing.max_resolution == 1920
    assert rebalance.batch_size == bucketing.batch_size


def test_the_borrow_copy_rows_get_their_own_derived_paths():
    """The planner appends duplicate rows and remaps only image_path and concept, so
    anything derived from image_path ahead of it still holds the pre-copy row count
    and IndexErrors on the first minted row."""
    config = _config(masked_training=True, custom_conditioning_image=True)
    loader = _Loader()

    assert not [m for m in loader._enumerate_input_modules(config) if isinstance(m, ModifyPath)]
    derived = [m.out_name for m in loader._derive_path_modules(config) if isinstance(m, ModifyPath)]
    assert derived == ["sample_prompt_path", "mask_path", "cond_path"]


# --- the tags ride the pipeline like crop_resolution ----------------------

def test_the_tags_are_cached_and_carried_like_crop_resolution():
    config = _config(aspect_ratio_bucket_min_tiers=_A_TIER)
    loader = _Loader()

    cache = _cache_modules(loader, config)
    disk_cache = _image_cache(cache)
    variation_sorting = _only(cache, VariationSorting)

    assert BUCKET_KEEP_NAME in disk_cache.aggregate_names
    assert BUCKET_REPEAT_NAME in disk_cache.aggregate_names
    # with latent caching on the cache restores them, so VariationSorting must not
    # also try to carry them -- exactly as it does not carry crop_resolution
    assert BUCKET_KEEP_NAME not in variation_sorting.names
    assert "crop_resolution" not in variation_sorting.names


def test_without_latent_caching_the_sorter_carries_the_tags():
    config = _config(latent_caching=False, aspect_ratio_bucket_min_tiers=_A_TIER)
    loader = _Loader()

    variation_sorting = _only(_cache_modules(loader, config), VariationSorting)

    assert BUCKET_KEEP_NAME in variation_sorting.names
    assert BUCKET_REPEAT_NAME in variation_sorting.names


@pytest.mark.parametrize("latent_caching", [True, False])
def test_the_batch_sorter_reads_the_tags_when_a_tier_exists(latent_caching):
    config = _config(latent_caching=latent_caching, aspect_ratio_bucket_min_tiers=_A_TIER)
    loader = _Loader()
    _cache_modules(loader, config)

    sorter = _sorter(loader, config)

    assert sorter.keep_in_name == BUCKET_KEEP_NAME
    assert sorter.repeat_in_name == BUCKET_REPEAT_NAME


# --- the SAMPLES budget ----------------------------------------------------

class _StubVariationSorting(VariationSorting):
    """VariationSorting over a fixed population, with the pipeline stubbed out.

    Only ``__init_variations`` is under test and it reads nothing but the two
    accessors below.
    """

    def __init__(self, balancing, strategy, population, **kwargs):
        super().__init__(
            names=["concept"],
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=["concept.path"],
            group_enabled_in_name="concept.enabled",
            **kwargs,
        )
        self._balancing = balancing
        self._strategy = strategy
        self._population = population

    def _get_previous_length(self, name):
        return self._population

    def _get_previous_item(self, variation, name, index):
        return {
            "concept.balancing": self._balancing,
            "concept.balancing_strategy": self._strategy,
            "concept.path": "concept",
            "concept.enabled": True,
        }[name]


def test_a_samples_budget_is_only_honoured_if_the_budget_tag_is_wired():
    """Why the group/budget tags are wired whether or not bucketing is on.

    VariationSorting no longer takes the SAMPLES subset itself: a subset taken
    there is drawn without regard to buckets and can leave a bucket below
    batch_size, so the full population is passed on and the batch sorter draws it
    in whole batches. Not wiring the tags is therefore not "the old behaviour" --
    it is a SAMPLES budget of 10 training on all 100 images.
    """
    tagged = _StubVariationSorting(
        10, "SAMPLES", 100, group_out_name=BUCKET_GROUP_NAME, budget_out_name=BUCKET_BUDGET_NAME)
    tagged.start(0)

    assert tagged.length() == 100, "the full population must reach the sorter"
    assert list(tagged.group_budgets.values()) == [10], "and the budget must travel with it"

    untagged = _StubVariationSorting(10, "SAMPLES", 100)
    untagged.start(0)
    assert untagged.length() == 100, (
        "the subset is not taken here any more, so without a budget tag downstream "
        "the whole population trains and the SAMPLES setting is silently discarded")


def test_the_sorter_is_wired_to_the_module_that_emits_the_budget():
    """Naming an input no module produces is a pipeline error, so the sorter asks for
    the group/budget tags only once a VariationSorting exists to emit them."""
    config = _config()
    loader = _Loader()

    assert _Loader()._emits_bucket_balancing is False
    assert _sorter(_Loader(), config).group_in_name is None

    _only(_cache_modules(loader, config), VariationSorting)
    sorter = _sorter(loader, config)
    assert sorter.group_in_name == BUCKET_GROUP_NAME
    assert sorter.budget_in_name == BUCKET_BUDGET_NAME


def test_the_variation_sorter_emits_the_budget_even_with_bucketing_off():
    config = _config(aspect_ratio_bucketing=False)
    variation_sorting = _only(_cache_modules(_Loader(), config), VariationSorting)

    assert variation_sorting.group_out_name == BUCKET_GROUP_NAME
    assert variation_sorting.budget_out_name == BUCKET_BUDGET_NAME


# --- config parsing --------------------------------------------------------

def test_a_row_with_no_max_size_is_inert_and_the_editor_starts_that_way():
    """A tier with no population threshold cannot match anything, so an unfinished row
    must not fail the run -- and the tier editor's new-row default is exactly that
    shape, so this is the state a user reaches by clicking "add tier"."""
    from modules.ui.BucketTierParamsWindowController import BucketTierListController

    assert parse_bucket_tiers([{"max_size": "", "strategy": "", "mode": "move"}]) == []
    assert parse_bucket_tiers([{"max_size": "", "strategy": "drop", "mode": "move"}]) == []
    assert parse_bucket_tiers([BucketTierListController(None).create_new_element()]) == []

    with pytest.raises(ValueError, match="no strategy"):
        parse_bucket_tiers([{"max_size": "4", "strategy": "", "mode": "move"}])


def test_a_typo_in_a_tier_is_reported_rather_than_ignored():
    with pytest.raises(ValueError, match="whole number"):
        parse_bucket_tiers([{"max_size": "eight", "strategy": "drop"}])
    with pytest.raises(ValueError, match="unknown bucket strategy"):
        parse_bucket_tiers([{"max_size": "8", "strategy": "borow"}])
    with pytest.raises(ValueError, match="unknown aspect bucket tier mode"):
        parse_bucket_tiers([{"max_size": "8", "strategy": "borrow", "mode": "cpoy"}])


def test_an_omitted_mode_means_borrow_move():
    assert parse_bucket_tiers([{"max_size": "8", "strategy": "borrow"}]) == [
        BucketTier(max_size=8, strategy="borrow", mode="move")]


def test_a_typo_in_the_resolution_mode_is_reported():
    """mgds treats anything it does not recognise as "split", so a typo would turn
    the feature off without saying so."""
    assert parse_resolution_mode("Rotate") == "rotate"
    with pytest.raises(ValueError, match="unknown aspect_ratio_bucket_resolution_mode"):
        parse_resolution_mode("rotato")


def test_a_cap_off_the_quantization_grid_is_refused():
    """The cap is applied before the crop is quantized, so an off-grid cap can be
    rounded straight back above itself and the model still gets an oversized bucket."""
    config = _config()

    with pytest.raises(ValueError, match="multiple of the bucket quantization"):
        bucketing_params(config, quantization=64, batch_size=4, max_resolution=1900)
    with pytest.raises(ValueError, match="must be positive"):
        bucketing_params(config, quantization=64, batch_size=4, max_resolution=0)

    assert bucketing_params(config, quantization=64, batch_size=4, max_resolution=1920).max_resolution == 1920


def test_anima_caps_its_buckets_on_the_quantization_grid():
    from modules.util.bucket_limits import ANIMA_MAX_BUCKET_RESOLUTION, max_bucket_resolution_for
    from modules.util.enum.ModelType import ModelType

    assert max_bucket_resolution_for(ModelType.ANIMA) == ANIMA_MAX_BUCKET_RESOLUTION
    # Anima's dataloader passes quantization 64; an off-grid cap would be refused
    assert ANIMA_MAX_BUCKET_RESOLUTION % 64 == 0
    assert max_bucket_resolution_for(ModelType.STABLE_DIFFUSION_15) is None


# --- the VAE fine-tune loader ---------------------------------------------

def _vae_loader():
    from modules.dataLoader.StableDiffusionFineTuneVaeDataLoader import StableDiffusionFineTuneVaeDataLoader
    # BaseDataLoader.__init__ builds the whole dataset; the module builders under test
    # are plain methods that touch none of that state.
    return StableDiffusionFineTuneVaeDataLoader.__new__(StableDiffusionFineTuneVaeDataLoader)


class _StubModel:
    """Enough of a model for the dataset builders: they only reach for the VAE, and
    only to hand it to a module that is never run here."""

    vae = None
    train_dtype = DataType.FLOAT_32


def _built_modules(loader, config, **kwargs):
    """Every module the loader's real _create_dataset builds, flattened.

    Driving _create_dataset rather than the individual builders is what pins the
    arguments the loader chooses for itself -- its bucket quantization, its cap --
    which a test that passes its own values cannot see change.
    """
    captured = []

    def capture(_config, module_lists, _train_progress, _is_validation):
        captured.extend(m for group in module_lists if group for m in group)

    loader._create_mgds = capture
    loader.train_device = None
    loader.temp_device = None
    loader._create_dataset(config, _StubModel(), None, None, False, **kwargs)
    return captured


def test_the_vae_finetune_loader_gets_the_same_wiring():
    """It does not route through the text2image mixin, so nothing about it follows
    from the tests above -- it is the one loader that has to be checked separately."""
    config = _config(aspect_ratio_bucket_tolerance=0.1, aspect_ratio_bucket_resolution_mode="rotate",
                     aspect_ratio_bucket_min_tiers=_A_TIER)

    bucketing = _only(_built_modules(_vae_loader(), config), AspectBucketing)

    assert bucketing.quantization == 8, "the VAE fine-tune pipeline quantizes to 8"
    assert bucketing.bucket_aspect_tolerance == pytest.approx(0.1)
    assert bucketing.resolution_mode == "rotate"
    assert bucketing.min_bucket_tiers == parse_bucket_tiers(_A_TIER)


def test_the_vae_finetune_loader_asks_only_for_tags_that_exist():
    """Its VariationSorting exists only when latent caching is on, and its keep/repeat
    tags are cached as aggregates, so with caching off the sorter must not ask for a
    budget no module produces."""
    loader = _vae_loader()

    cached = _config(latent_caching=True, aspect_ratio_bucket_min_tiers=_A_TIER)
    uncached = _config(latent_caching=False, aspect_ratio_bucket_min_tiers=_A_TIER)

    cached_sorter = _only(
        loader._StableDiffusionFineTuneVaeDataLoader__output_modules(cached), AspectBatchSorting)
    uncached_sorter = _only(
        loader._StableDiffusionFineTuneVaeDataLoader__output_modules(uncached), InlineAspectBatchSorting)

    assert cached_sorter.group_in_name == BUCKET_GROUP_NAME
    assert cached_sorter.budget_in_name == BUCKET_BUDGET_NAME
    assert uncached_sorter.group_in_name is None
    assert uncached_sorter.budget_in_name is None
    # the tags themselves come straight from AspectBucketing either way
    assert cached_sorter.keep_in_name == uncached_sorter.keep_in_name == BUCKET_KEEP_NAME


# --- the whole pipeline ----------------------------------------------------

def test_a_loader_s_own_quantization_and_cap_reach_both_modules():
    """The per-model quantization and long-edge cap are chosen by the concrete
    loader, not by the config, so this is the only place they can be checked."""
    config = _config(aspect_ratio_bucket_min_tiers=[
        {"max_size": "8", "strategy": "borrow", "mode": BORROW_COPY}])

    modules = _built_modules(_Loader(), config, aspect_bucketing_quantization=32,
                             aspect_bucketing_max_resolution=1920)

    bucketing = _only(modules, AspectBucketing)
    rebalance = _only(modules, AspectBucketRebalance)

    assert bucketing.quantization == rebalance.quantization == 32
    assert bucketing.max_resolution == rebalance.max_resolution == 1920


def test_the_planner_runs_before_the_paths_it_renames_are_derived():
    """Ordering, not just presence: a ModifyPath ahead of the planner holds the
    pre-copy row count and IndexErrors on the first minted row."""
    config = _config(masked_training=True, aspect_ratio_bucket_min_tiers=[
        {"max_size": "8", "strategy": "borrow", "mode": BORROW_COPY}])

    modules = _built_modules(_Loader(), config, aspect_bucketing_quantization=64)
    kinds = [type(m) for m in modules]

    assert kinds.index(AspectBucketRebalance) < kinds.index(ModifyPath)
