# Aspect-Bucket Population Rebalancing — Plan

**Status:** Phases 1 & 2 implemented; Phase 3 `borrow-copy` implemented (see §11).
Variation-aware `repeat` still deferred.
**Scope:** `mgds` (sibling editable clone at `../mgds`, origin=joshuafcole/mgds) + OneTrainer `modules/`
**Owner:** Joshua
**Last updated:** 2026-05-26

Working reference for making sparse-bucket handling explicit and configurable in the
**budget-based** resolution mode (comma-separated integer pixel budgets, e.g.
`"512, 1024"`).

**Explicitly out of scope (unchanged behavior):**
- Budget *selection* across multiple budgets. Today each image is assigned a budget
  uniform-randomly per access (`AspectBucketing.py:230`, `rand.choice`), independent of
  native size, so every concept image is rescaled across **all** trained budgets and
  grouped only by aspect ratio. This is the **desired** behavior and we are **not**
  changing it in this project. (A native-size-aware policy may be revisited later.)
- Intermixed explicit `WxH` rectangles in the resolution string (separate follow-on).

---

## 1. Background & current behavior (researched, with refs)

### 1.1 The master switch
`aspect_ratio_bucketing` (default **True**, `modules/util/config/TrainConfig.py:389,992`)
selects the per-item resolution module in
`modules/dataLoader/mixin/DataLoaderText2ImageMixin.py:183-186`:
- **True → `AspectBucketing`** — multi-aspect bucketing.
- **False → `SingleAspectCalculation`** — square-only (`(N, N)` center-crop).

### 1.2 A single budget number is a *pixel budget*, not a square
With bucketing on, a number `N` is expanded by
`AspectBucketing.__create_automatic_buckets` (`AspectBucketing.py:85-118`) into a
**family of buckets** at ≈`N²` pixels across a fixed aspect ladder
(1:1 … 1:4 + inverses, `AspectBucketing.py:18-28`), quantized to the model's
quantization (e.g. 64). Varied aspect ratios are preserved, not forced square.

### 1.3 Assignment is deterministic nearest-aspect, single bucket, no spread
`AspectBucketing.__get_bucket` (`AspectBucketing.py:154-157`) is
`argmin |bucket_aspect − image_aspect|`; the `rand` arg is **unused** → each image
lands in exactly one bucket, deterministically. All augmentation *variations* of a
source image share its native resolution → same rung; variations multiply a bucket's
population but never spread across buckets.

### 1.4 Budget assignment is uniform-random and native-size-agnostic (intended)
`AspectBucketing.get_item` (`AspectBucketing.py:228-231`):
```python
target_resolutions = [int(res.strip()) for res in target_resolutions.split(',')]
target_resolution = rand.choice(target_resolutions)          # uniform random budget
target_resolution = self.__get_bucket(rand, h, w, target_resolution)  # aspect snap
```
Native size influences only the aspect snap, never the budget choice. **This is
desired** (rescale every image across all budgets, group by aspect). Out of scope to
change. Key implication for this project: because the aspect ladder is the *same set of
rungs* at every budget, an image's nearest-aspect **rung index is budget-independent**.
So the dataset's aspect-rung population histogram is budget-invariant.

### 1.5 Sparse buckets are silently dropped (no reassignment, no partial batch)
Batching is drop-last **per bucket** (`batch_size = config.batch_size * world_size`):
- `AspectBatchSorting.__shuffle` (`AspectBatchSorting.py:27-60`): `batch_count =
  int(len(bucket)/batch_size)` (floor) + explicit remainder pop (lines 44-49).
- `InlineAspectBatchSorting.__fill_cache` (`InlineAspectBatchSorting.py:46-78`): a
  bucket emits only at exactly `batch_size`; leftovers discarded.

| Bucket population | Outcome |
|---|---|
| `< batch_size` | **0 batches → every image permanently dropped, every epoch** |
| `≥ batch_size`, not a multiple | rotating remainder dropped (reduced frequency) |
| exact multiple | fully used |

Drops are silent (only a commented-out `print` at `AspectBatchSorting.py:48`).

> Note: with multiple budgets, each epoch routes only ~`1/num_budgets` of the dataset
> to a given budget, so a sparse aspect is even sparser *per budget per epoch*. We
> therefore plan on the **full-dataset aspect-rung population** and apply each item's
> fate regardless of which budget it rolls into (see §3.5).

### 1.6 Architectural constraint: where length/resolution may change
Pipeline order (`DataLoaderText2ImageMixin._create_dataset:408-427`):
`AspectBucketing → ScaleCropImage → augmentation → DiskCache → AspectBatchSorting`.
- **Crop resolution is baked into the cached latent** → anything changing *which
  resolution* an image is cropped to must happen in `AspectBucketing` (before crop).
- **MGDS allows length changes only via carry-all modules** that re-emit every field
  read downstream; the blessed location is the **sorter** (end of pipeline) or the
  variations/cache system. A mid-pipeline length-changer must carry every field at that
  stage (brittle) → avoided for v1.

This split dictates where each strategy can cleanly live (§3.2).

### 1.7 Prior local work (uncommitted)
`AspectBucketing.py` carries an uncommitted local feature: `bucket_aspect_tolerance` +
`__collapse_close_aspects` (`AspectBucketing.py:120-152`), wired via
`config.aspect_ratio_bucket_tolerance` (`TrainConfig.py:390,993`; UI
`TrainUI.py:361-363`). It is a **static, data-agnostic** pre-pass that merges nearby
aspect rungs onto a shared crop. The work below is the **population-aware** complement.

---

## 2. Goals / non-goals
**Goals**
- Stop silently dropping data from sparse aspect buckets; make it explicit/configurable.
- Visibility: per-rung population histogram + a summary of actions taken.
- Keep everything **opt-in**; defaults reproduce today's behavior exactly.
- Extensible foundation (tiered rules, pluggable strategies).

**Non-goals**
- Budget selection policy (see out-of-scope note up top).
- Intermixed explicit `WxH` rectangles.
- Changing the fixed aspect ladder / making it data-driven.

---

## 3. Feature — Bucket population management (tiers + strategies)

### 3.1 Tier model
A tier = `(max_size, strategy, mode?)`. A rung whose population `p` satisfies
`p < max_size` is handled by that tier's strategy. Tiers are evaluated **smallest
`max_size` first**; first match wins; rungs at/above all tiers are untouched. Empty
config = today's behavior.

Example:
```
p < 2  → drop
p < 6  → donate
p < 10 → borrow (mode=move|copy)
```
(p=1→drop, p∈[2,5]→donate, p∈[6,9]→borrow, p≥10→keep.)

### 3.2 Strategies

| Strategy | Meaning | Changes | Lives in | Phase |
|---|---|---|---|---|
| **drop** | discard the rung's items | multiset (−) | sorter (skip via `keep` tag) | 2 |
| **donate** | reassign items to nearest surviving rung; rung dissolves | resolution remap | `AspectBucketing` | 2 |
| **borrow (move)** | rung survives; pull nearest neighbor items (re-cropped here) to fill target; donors capped to stay viable | resolution remap | `AspectBucketing` | 2 |
| **repeat (identical)** | oversample the rung's own members to fill target | multiset (+) | sorter (`repeat` tag) | 2 |
| **borrow (copy)** | like move, but neighbor also stays in its home rung (shared) | multiset (+) + re-crop | `AspectBucketRebalance` (path-stage row dup) | 3 ✅ |
| **repeat (varied)** | oversample with *distinct* augmentation variants | multiset (+) + re-aug | needs upstream dup / variation-system integration | 3 |

Rationale for the 2/3 split: **move**/**donate** are pure resolution remaps (no length
change) → clean in `AspectBucketing`. **drop**/**identical-repeat** are length changes
at the blessed end-of-pipeline location → small, backward-compatible sorter flags.
**copy** needs a re-cropped *duplicate*: realized as a genuine extra **row** minted at
the path stage (`AspectBucketRebalance`, before images load), so it flows through
load→crop→encode→cache as a first-class item and the disk cache encodes it at the borrow
crop — no cache changes, works at any `image_variations` (see §11). **varied-repeat**
still needs variation-system integration → remains deferred.

### 3.3 Fill target (saving a rung)
For **borrow**/**repeat**, fill up to the smallest multiple of `batch_size` that is ≥
the rung's current population (and ≥ `batch_size`). Every *surviving* rung is then
batch-aligned, so the sorter's drop-last is a no-op for it — **no sorter rewrite of the
drop logic needed**. (Confirmed: "the lowest tier should be `batch_size`.") The tier
`max_size` selects *which* strategy; batch-alignment sets *how much* to fill.
`AspectBucketing` needs the effective `batch_size` passed in.

### 3.4 Single brain, dumb executor
All counting + tier classification + planning happens **once** in
`AspectBucketing.start()` from the initial snapshot. It:
- applies **donate**/**borrow-move** itself (per-item rung override consulted by
  `get_item` after the budget roll + aspect snap),
- emits per-item fate tags **`bucket_keep` (bool)** and **`bucket_repeat` (int)**,
- logs diagnostics.

The sorter stays dumb: given optional `keep_in_name` / `repeat_in_name`, it skips
`keep=False` items and duplicates `repeat>1` items while building `bucket_dict`.
Defaults (names `None`) = current behavior. Keeps tier logic in one place; avoids
recompute drift between modules.

### 3.5 Budget-invariant, per-rung planning
Planning operates on **aspect rungs**, not per-budget buckets, because rung membership
is budget-independent (§1.4):
1. Counting pass over `original_resolution`: for each item compute aspect and nearest
   rung (using the canonical rung aspects); tally `pop[rung]`.
2. `fate[rung]` from the tiers.
3. Per item: store fate by its rung. In `get_item`, after the normal budget roll +
   aspect snap to rung `r`:
   - `donate` → snap to the neighbor rung's resolution **in the rolled budget** instead;
   - `borrow-move` (recipient side) → some neighbor items snap into `r` in the rolled
     budget;
   - `drop` → emit `keep=False`;
   - `repeat` → emit repeat factor.

A rung's fate thus applies consistently no matter which budget the item rolls into, so
multiple budgets need no special handling.

### 3.6 Diagnostics
At dataset build, **always** (whenever `aspect_ratio_bucketing` is enabled) log: an
ASCII histogram of rung → population (pre-resolution) and a one-line action summary
(e.g. `rung 1:2.0 (3 imgs) → donated to 1:1.75`; `dropped 5 imgs across 2 rungs`;
`repeated 1:1.75 6→8`). This makes the counting pass unconditional when bucketing is
on (it loads images once — latent caching loads them anyway). No flag.

---

## 4. Config schema (TrainConfig + UI)
New fields (all default to current behavior):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `aspect_ratio_bucket_min_tiers` | `list[dict[str,str]]` | `[]` | tiers, mirroring the `scheduler_params` list-of-dict pattern (`TrainConfig.py:397-399`) |

(The histogram prints unconditionally when bucketing is on — no diagnostics flag.)

Each tier dict: `{"max_size": "6", "strategy": "donate", "mode": ""}`. Parsed/validated
in the data loader into a typed `BucketTier` before being handed to the mgds module
(mgds stays OneTrainer-agnostic).

**UI:** add near the existing Bucket Tolerance control
(`TrainUI.create_data_tab`, `TrainUI.py:356-363`): a `ConfigList`-style tier editor +
diagnostics switch. Config + behavior can land before the UI pass.

---

## 5. Files to change
**mgds (`venv/src/mgds/src/mgds/`)**
- `util/bucketRebalancing.py` *(new)* — pure, numpy-only: `BucketTier`, per-rung
  planner, histogram formatter. No torch import → unit-testable.
- `pipelineModules/AspectBucketing.py` — counting pass in `start()`, apply rung
  remap in `get_item`, emit `bucket_keep`/`bucket_repeat`, log the histogram
  unconditionally. New `__init__` params: `batch_size`, `min_bucket_tiers`,
  `keep_out_name`, `repeat_out_name`.
- `pipelineModules/AspectBatchSorting.py` and `InlineAspectBatchSorting.py` — optional
  `keep_in_name` / `repeat_in_name`; honor them when building buckets.

**mgds tests (`venv/src/mgds/tests/`)**
- `test_bucket_rebalancing.py` *(new)* — pure-logic unit tests (no VAE/CUDA).

**OneTrainer (`modules/`)**
- `util/config/TrainConfig.py` — 2 new fields + defaults.
- `dataLoader/mixin/DataLoaderText2ImageMixin.py` — thread `batch_size` + new config
  into `AspectBucketing`; pass `keep_in_name`/`repeat_in_name` into the sorters in
  `_output_modules_from_out_names`. Parse tier dicts → `BucketTier`.
- `dataLoader/StableDiffusionFineTuneVaeDataLoader.py` — same wiring for the VAE loader.
- `ui/TrainUI.py` — tier editor + diagnostics switch.

---

## 6. Planner pseudocode (per rung)
```
plan(item_aspects, rung_aspects, tiers, batch_size):
    assign[i] = argmin_r |rung_aspects[r] - item_aspects[i]|     # nearest rung
    pop[r]    = count of i with assign[i] == r
    fate[r]   = first tier with pop[r] < tier.max_size, else KEEP # ascending
    surviving = { r : fate[r] in (KEEP, BORROW, REPEAT) }

    DROP   r: keep[i in r] = False
    DONATE r: reassign each i in r -> nearest surviving rung (by aspect)
    BORROW r: target = ceil(max(pop[r],batch)/batch)*batch
              pull nearest neighbor items into r until target,
              capping donors at their viability floor
    REPEAT r: target = ceil(max(pop[r],batch)/batch)*batch
              repeat[i in r] to reach target (distribute remainder)
    return keep[], rung_override[], repeat[], diagnostics
```
**Algorithm details to finalize in impl:** donor selection order & viability floor for
borrow; whether donate recipients are re-aligned (default: no, accept bounded sorter
remainder loss); seeded tie-breaking.

---

## 7. Test plan (Phase-2 core, pure)
- Tier selection boundaries (`p == max_size` not matched; `p == max_size−1` matched);
  ascending first-match; no tiers = identity.
- drop: sub-threshold rungs flagged `keep=False`; others untouched.
- donate: items land in nearest surviving rung; dissolved rung emptied.
- borrow-move: recipient reaches a batch multiple; donors not pushed below floor;
  deterministic given seed.
- repeat: factors reach the batch-aligned target; sane distribution.
- fill target = batch alignment across `batch_size ∈ {1,2,4,8}`.
- histogram formatting snapshot.
- Integration smoke (manual, mirrors `tests/test_main.py`, CUDA+VAE): multi-budget run
  with tiers completes and trains formerly-dropped images.

---

## 8. Phasing
- **Phase 1:** counting pass + always-on diagnostics histogram (visibility only; no
  *training* behavior change). Establishes the foundation the strategies build on.
- **Phase 2 (core):** tier framework + `drop`, `donate`, `borrow-move`,
  `repeat-identical`, batch-aligned fill, `keep`/`repeat` sorter tags, full config +
  wiring + unit tests.
- **Phase 3 (follow-on):** `borrow-copy` ✅ done via path-stage row duplication
  (`AspectBucketRebalance`); variation-aware `repeat` still pending (needs the
  variations/augmentation system).

---

## 9. Backward compatibility
- Empty tiers → no change to *training* behavior (same RNG usage, same crops, same
  drops). The only difference with bucketing on is the always-printed histogram and the
  counting pass it requires. New sorter params default to `None`; new `AspectBucketing`
  params have safe defaults.

---

## 10. Resolved decisions
1. **Borrow donor viability floor:** a donor may be depleted **down to `batch_size`**
   (never below one full batch).
2. **Fill target:** round up to the **smallest multiple of `batch_size` ≥ current
   population** (and ≥ `batch_size`).
3. **Diagnostics:** **always** print the histogram when `aspect_ratio_bucketing` is
   enabled (counting pass is unconditional); no flag.

---

## 11. Implementation status (Phases 1 & 2 done)

**mgds (`../mgds`, branch `feat/bucket-rebalancing`):**
- `util/bucketRebalancing.py` — pure planner (`BucketTier`, `RebalancePlan`,
  `plan_rebalance`, `batch_aligned_target`, `format_histogram`, strategy/mode
  constants). Stdlib-only; 23 unit tests in `tests/test_bucket_rebalancing.py`.
- `pipelineModules/AspectBucketing.py` — new `__init__` params `batch_size`,
  `min_bucket_tiers`, `keep_out_name`, `repeat_out_name`. Counting + planning runs
  once in `start()` over the **budget-based** items (fixed-WxH items are exempt);
  `get_item` applies the rung override (donate / borrow-move) and emits
  `bucket_keep`/`bucket_repeat`; the rung histogram + action log print whenever
  bucketing is on.
- `pipelineModules/AspectBatchSorting.py` + `InlineAspectBatchSorting.py` — optional
  `keep_in_name` / `repeat_in_name`; skip `keep=False`, duplicate `repeat>1`
  (inline expands the index list up-front to keep the exact-batch trigger intact).
  Defaults `None` => upstream behavior.

**OneTrainer (`modules/`):**
- `util/config/TrainConfig.py` — `aspect_ratio_bucket_min_tiers: list[dict[str,str]]`
  (default `[]`).
- `dataLoader/mixin/DataLoaderText2ImageMixin.py` — module-level `BUCKET_KEEP_NAME`,
  `BUCKET_REPEAT_NAME`, `parse_bucket_tiers()`. `_aspect_bucketing_in` threads
  `batch_size` (= `config.batch_size * world_size`) + parsed tiers + tag out-names.
  `_cache_modules_from_names` carries the tags exactly like `crop_resolution`
  (image-cache aggregates + `sort_names`). `_output_modules_from_out_names` passes
  the tag in-names to both sorters. All gated on `config.aspect_ratio_bucketing`.
- `dataLoader/StableDiffusionFineTuneVaeDataLoader.py` — same wiring (tags as
  cache-only aggregates; no VariationSorting in its non-cached path).
- `ui/BucketTierParamsWindow.py` + `ui/TrainUI.py` — "configure tiers" button on the
  data tab opens a `ConfigList` tier editor (max_size / strategy / mode).

**Verified:** unit tests (23/23), `py_compile` of all touched files, OneTrainer
import + config round-trip + `parse_bucket_tiers` + no circular import. **Not yet
run:** end-to-end GPU training (field-flow through DiskCache, histogram output,
formerly-dropped images now trained) — the manual CUDA+VAE smoke test from §7.

### Phase 3: borrow-copy (implemented)

The architectural blocker for copy was that an extra re-cropped instance of a donor
needs a *length increase before the crop*, where MGDS only blesses the sorter
(post-crop, too late to re-crop) or the variations/cache system. The realization
makes the copy a **genuine extra dataset row** minted at the path stage, so the disk
cache encodes it at the borrow crop with **no cache changes** and **independent of
`image_variations`** (the open question that gated the design).

**mgds (`../mgds`, branch `feat/bucket-rebalancing`):**
- `util/bucketRebalancing.py` — `plan_rebalance` gains `RebalancePlan.copies`
  (`(item, aspect)` pairs). `borrow`+`mode=copy` fills the borrow rung from the
  nearest KEEP rungs *without* depleting them (no viability floor, no override on the
  donor) — each donor original is re-cropped into the borrow rung as a duplicate.
  The bucket-ladder construction (`build_bucket_resolutions`, `collapse_close_aspects`,
  `quantize_resolution`, `ASPECT_LADDER`, `reference_rung_aspects`) is extracted here
  as the single source of truth shared with `AspectBucketing`.
- `pipelineModules/AspectBucketRebalance.py` *(new)* — path-stage planner + row
  duplicator. Reads EXIF-corrected aspects from file headers (cheap, no decode),
  runs `plan_rebalance`, emits per-row `keep`/`repeat`/`override_aspect` tags, and
  **appends a duplicate (`image_path`,`concept`) row per copy** tagged with the borrow
  aspect. Non-image/unreadable files opt out and pass through.
- `pipelineModules/AspectBucketing.py` — gains a **consumer mode**
  (`override_aspect_in_name`): when set it does no planning and just snaps each row to
  the supplied aspect. Bucket building now delegates to the shared util.

**OneTrainer (`modules/`):**
- `dataLoader/mixin/DataLoaderText2ImageMixin.py` — `BUCKET_OVERRIDE_NAME`,
  `has_copy_tier()`. `_bucket_rebalance_modules()` inserts `AspectBucketRebalance`
  (right after path enumeration, before image load) **only when a copy tier is
  configured**; `_aspect_bucketing_in` then runs `AspectBucketing` in consumer mode.
  Non-copy configs are byte-for-byte unchanged (module not inserted, inline planning).
- `dataLoader/StableDiffusionFineTuneVaeDataLoader.py` — same wiring (quantization 8).
- UI / rehearsal schema needed **no change**: `mode` already offered `move`/`copy`.

**Design notes:** planning here reads file-header aspects (EXIF-corrected to mirror
`LoadImage`) rather than decoded tensors; they agree for all normal images, and any
rung-boundary disagreement is marginal and never fatal. Copy donors are distinct
within a borrow rung (max diversity); donors are never depleted. Video items are not
rebalanced under copy mode (image-dimension planner) — a documented scope limit.

**Verified:** mgds unit tests (26/26, incl. 4 new copy tests), e2e tests (6/6, incl.
a full `AspectBucketRebalance → AspectBucketing(consumer) → AspectBatchSorting` chain
over real image files proving copies fill the borrow batch without depleting donors),
`pyright` clean on the planner + new module, `ruff` clean, OneTrainer import +
`has_copy_tier` gating. **Not yet run:** end-to-end GPU training with a copy tier.

### Multi-budget effectiveness caveat
Budget assignment is uniform-random per item (§1.4), so a rung's items **split
across** per-budget buckets each epoch. Batch-aligning the *total* rung population
therefore fully prevents drops only with a **single budget**. With multiple budgets:
**drop** and **donate** stay effective (they reduce rung count / fatten survivors,
shrinking per-budget remainders), while **repeat** and **borrow** only *reduce*
per-budget remainder drops rather than eliminate them. Guaranteeing per-budget
batches would require changing the out-of-scope random budget assignment.

### cinema-studio rehearsal integration
The rehearsal launch config (cinema-studio) exposes this surface so a run can be
configured without editing JSON by hand:
- `rehearsal-schema/models.py` — `BucketTier` CamelModel (typed `max_size: int`;
  `strategy` / `mode` literals). `RunConfigSummary.bucket_tiers` (read) and
  `TrainingPlan.bucket_tiers` (write).
- `scanner/config_read.py::read_bucket_tiers` — parses OT's
  `aspect_ratio_bucket_min_tiers` (string→int boundary), tolerant of garbage.
- `launcher/config_builder.py` — `apply_overrides` writes the tiers **wholesale**
  (`None` inherits the base; a list, incl. `[]`, replaces it — tiers are anonymous
  rules the loader re-sorts, so no sparse per-index channel); diff labels read
  "Tier N · …".
- `launcher/config_introspect.py` — `aspect_ratio_bucketing` + `…_tolerance`
  scalars added to the relevance map (group `data`, warm) so they ride the generic
  override channel.
- Frontend `rehearsal/components/bucket-tier-row.tsx` + the "Aspect-ratio buckets"
  section in `configure-panel.tsx` (toggle + tolerance via the generic field
  renderer; tiers via a whole-list override editor). TS types regenerated through
  freight-depot.
- Verified: 253 rehearsal-agent tests (10 new), pyright/ruff clean, frontend
  typecheck + 357 studio tests, codegen `--check` matches.
