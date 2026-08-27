"""Shared vocabulary for population-aware aspect-bucket rebalancing.

An aspect bucket holding fewer images than one batch is silently discarded by the
batch sorter's per-bucket drop-last: those images never train and nothing says so.
mgds' rebalancing planner can resolve such buckets instead (drop / donate / borrow /
repeat), and this module is the single place OneTrainer turns user configuration
into the arguments that planner needs.

Everything here is pure and torch-free so the dataloaders, the VAE fine-tune loader
and the UI can all import it without pulling in the training stack.
"""

from dataclasses import dataclass

from modules.util.config.TrainConfig import TrainConfig

from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.AspectBucketRebalance import AspectBucketRebalance
from mgds.util.bucketRebalancing import BORROW, BORROW_COPY, BORROW_MOVE, BucketTier

# Field names carrying the planner's per-item decisions through the pipeline.
# keep/repeat are treated exactly like crop_resolution -- cached as DiskCache
# aggregates and carried by VariationSorting -- so they reach the batch sorter in
# both the cached and the uncached mode.
BUCKET_KEEP_NAME = 'bucket_keep'
BUCKET_REPEAT_NAME = 'bucket_repeat'
# Per-item balancing-group id and sample budget, emitted by VariationSorting and
# read by the batch sorter so a SAMPLES subset is drawn in whole batches per bucket
# instead of bucket-blind (budget -1 == "take all whole batches").
BUCKET_GROUP_NAME = 'bucket_group'
BUCKET_BUDGET_NAME = 'bucket_budget'
# Per-row rung override emitted by AspectBucketRebalance and consumed by
# AspectBucketing. Unlike keep/repeat it is read mid-pipeline, before the crop, so
# it is neither cached nor carried to the sorter.
#
# It carries a rung *index*, never an aspect value: a rung renders to a different
# crop at every pixel budget, so re-deriving the rung from an aspect against a
# budget's own quantized ladder splits a rung's population across buckets and
# breaks the guarantee the planner just made. See mgds' bucketRebalancing docstring.
BUCKET_OVERRIDE_RUNG_NAME = 'bucket_override_rung'

# Strategies a tier may select. "keep" is the implicit behaviour of any bucket
# above every tier and is deliberately not offered as a rule.
STRATEGY_VALUES = ['drop', 'donate', 'borrow', 'repeat']
# Borrow sub-modes. Ignored by every other strategy.
MODE_VALUES = [BORROW_MOVE, BORROW_COPY]
# Multi-resolution budget selection, see TrainConfig.aspect_ratio_bucket_resolution_mode.
RESOLUTION_MODE_VALUES = ['split', 'rotate']


def parse_bucket_tiers(tier_dicts: list[dict[str, str]] | None) -> list[BucketTier]:
    """Convert the config's tier dicts into typed BucketTier rules.

    Values are strings because the tier list is edited through a ConfigList, which
    only stores strings.

    A row with no max_size names no population and so cannot apply to anything: it
    is a row the user added and has not filled in yet, and it is skipped. Anything
    else that cannot be read raises, so a typo in a hand-edited config is reported
    rather than quietly changing which images train -- which is the failure mode
    this whole feature exists to remove.
    """
    tiers: list[BucketTier] = []
    for tier_dict in tier_dicts or []:
        strategy = (tier_dict.get('strategy') or '').strip().lower()
        max_size_str = str(tier_dict.get('max_size') or '').strip()
        if not max_size_str:
            continue
        if not strategy:
            raise ValueError(f"aspect bucket tier {tier_dict!r} has a max_size but no strategy")
        try:
            max_size = int(max_size_str)
        except ValueError:
            raise ValueError(f"aspect bucket tier max_size must be a whole number, got {max_size_str!r}") from None
        mode = (tier_dict.get('mode') or '').strip().lower() or BORROW_MOVE
        if mode not in MODE_VALUES:
            raise ValueError(f"unknown aspect bucket tier mode: {mode!r} (expected one of {MODE_VALUES})")
        tiers.append(BucketTier(max_size=max_size, strategy=strategy, mode=mode))
    return tiers


def bucket_tags_enabled(config: TrainConfig) -> bool:
    """True when the pipeline carries the planner's keep/repeat tags.

    Read by the cache and the batch sorter, which see the config but not the derived
    geometry. Off unless bucketing is on *and* at least one tier exists, so a run
    that never configures a tier has the same names on the wire and the same values
    in its latent cache as it did before this feature existed.
    """
    return bool(config.aspect_ratio_bucketing) and bool(parse_bucket_tiers(config.aspect_ratio_bucket_min_tiers))


def has_copy_tier(tiers: list[BucketTier]) -> bool:
    """True if any tier asks for borrow-copy.

    borrow-copy is the one strategy that cannot be planned inside AspectBucketing:
    it mints an extra, differently cropped row per borrowed image, and rows can only
    be added before the images are loaded. It therefore moves planning upstream into
    AspectBucketRebalance and leaves AspectBucketing as a consumer of the result.
    """
    return any(tier.strategy == BORROW and tier.mode == BORROW_COPY for tier in tiers)


def parse_resolution_mode(resolution_mode: str | None) -> str:
    """Validate the multi-resolution budget mode.

    mgds treats anything it does not recognise as "split", so a typo would silently
    turn the feature off. Rejecting it here means the user is told instead.
    """
    mode = (resolution_mode or 'split').strip().lower()
    if mode not in RESOLUTION_MODE_VALUES:
        raise ValueError(
            f"unknown aspect_ratio_bucket_resolution_mode: {resolution_mode!r} "
            f"(expected one of {RESOLUTION_MODE_VALUES})")
    return mode


@dataclass(frozen=True)
class BucketingParams:
    """The bucket geometry AspectBucketing and AspectBucketRebalance must share.

    The two modules do not decide an image's rung from these values -- that is the
    budget-free canonical ladder -- but they do use them to work out which rungs are
    indistinguishable, i.e. render to the same crop at every budget and so should be
    planned as one unit. Given different values the two pool rungs differently and
    the plan degrades.

    They are bundled into one object, and both modules are built from that one
    object, so a mismatch is not something a caller can express rather than
    something a comment asks them to avoid.
    """

    quantization: int
    tolerance: float
    max_resolution: int | None
    batch_size: int
    resolution_mode: str
    tiers: list[BucketTier]

    @property
    def consumer_mode(self) -> bool:
        """True when planning belongs to AspectBucketRebalance, not AspectBucketing."""
        return has_copy_tier(self.tiers)

    @property
    def emits_tags(self) -> bool:
        """True when a keep/repeat decision can differ from "keep everything once".

        With no tiers the planner has no rule to apply, so the tags would be a
        constant True/1 on every row -- but they would still be two more names on the
        wire and two more values in every cached item, changing the on-disk cache for
        a feature nobody switched on. So they are emitted only once a tier exists.
        """
        return bool(self.tiers)


def bucketing_params(
        config: TrainConfig,
        quantization: int,
        batch_size: int,
        max_resolution: int | None = None,
) -> BucketingParams:
    """Derive the shared bucket geometry from a TrainConfig.

    ``max_resolution`` must be a multiple of ``quantization``: the cap is applied
    before the crop is quantized, so a cap that is not on the quantization grid can
    be rounded straight back above itself, and the model whose limit the cap exists
    to respect would still be handed an oversized bucket.
    """
    if max_resolution is not None:
        if max_resolution <= 0:
            raise ValueError(f"aspect bucket max_resolution must be positive, got {max_resolution}")
        if max_resolution % quantization != 0:
            raise ValueError(
                f"aspect bucket max_resolution {max_resolution} must be a multiple of the bucket "
                f"quantization {quantization}, or quantizing a capped edge can round it back above the cap")

    return BucketingParams(
        quantization=quantization,
        tolerance=config.aspect_ratio_bucket_tolerance,
        max_resolution=max_resolution,
        batch_size=batch_size,
        resolution_mode=parse_resolution_mode(config.aspect_ratio_bucket_resolution_mode),
        tiers=parse_bucket_tiers(config.aspect_ratio_bucket_min_tiers),
    )


def aspect_bucketing_module(
        params: BucketingParams,
        resolution_in_name: str,
        target_resolution_in_name: str,
        enable_target_resolutions_override_in_name: str,
        target_resolutions_override_in_name: str,
        target_frames_in_name: str,
        frame_dim_enabled: bool,
        scale_resolution_out_name: str,
        crop_resolution_out_name: str,
        possible_resolutions_out_name: str,
) -> AspectBucketing:
    """Build AspectBucketing from the shared geometry.

    In consumer mode the plan was already made by AspectBucketRebalance and arrives
    per row, so this module is given no tiers and emits no tags -- it only renders
    the supplied rung at the row's budget.
    """
    consumer_mode = params.consumer_mode
    emits_tags = params.emits_tags and not consumer_mode

    return AspectBucketing(
        quantization=params.quantization,
        resolution_in_name=resolution_in_name,
        target_resolution_in_name=target_resolution_in_name,
        enable_target_resolutions_override_in_name=enable_target_resolutions_override_in_name,
        target_resolutions_override_in_name=target_resolutions_override_in_name,
        target_frames_in_name=target_frames_in_name,
        frame_dim_enabled=frame_dim_enabled,
        scale_resolution_out_name=scale_resolution_out_name,
        crop_resolution_out_name=crop_resolution_out_name,
        possible_resolutions_out_name=possible_resolutions_out_name,
        bucket_aspect_tolerance=params.tolerance,
        batch_size=params.batch_size,
        min_bucket_tiers=[] if consumer_mode else params.tiers,
        keep_out_name=BUCKET_KEEP_NAME if emits_tags else None,
        repeat_out_name=BUCKET_REPEAT_NAME if emits_tags else None,
        override_rung_in_name=BUCKET_OVERRIDE_RUNG_NAME if consumer_mode else None,
        resolution_mode=params.resolution_mode,
        max_resolution=params.max_resolution,
    )


def aspect_bucket_rebalance_modules(
        params: BucketingParams,
        path_in_name: str,
        concept_in_name: str,
        target_resolution_in_name: str,
        enable_target_resolutions_override_in_name: str | None,
        target_resolutions_override_in_name: str | None,
        image_extensions: set[str],
) -> list[AspectBucketRebalance]:
    """The path-stage planner, or an empty list when nothing needs it.

    Only borrow-copy needs a module here: it appends re-cropped duplicate rows, and
    rows can only be appended before the images are loaded. Every other
    configuration -- including no tiers at all -- returns nothing, so the pipeline
    is left exactly as it was.
    """
    if not params.consumer_mode:
        return []

    return [AspectBucketRebalance(
        path_in_name=path_in_name,
        concept_in_name=concept_in_name,
        target_resolution_in_name=target_resolution_in_name,
        enable_target_resolutions_override_in_name=enable_target_resolutions_override_in_name,
        target_resolutions_override_in_name=target_resolutions_override_in_name,
        quantization=params.quantization,
        bucket_aspect_tolerance=params.tolerance,
        batch_size=params.batch_size,
        min_bucket_tiers=params.tiers,
        image_extensions=image_extensions,
        path_out_name=path_in_name,
        concept_out_name=concept_in_name,
        keep_out_name=BUCKET_KEEP_NAME,
        repeat_out_name=BUCKET_REPEAT_NAME,
        override_rung_out_name=BUCKET_OVERRIDE_RUNG_NAME,
        max_resolution=params.max_resolution,
    )]
