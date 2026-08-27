# Aspect Bucket Rebalancing

Aspect ratio bucketing sorts your images by shape and trains each shape in its own
batches, so nothing has to be squashed to a square. The part that is rarely
mentioned is what happens to a shape you only have a few of.

Batches are built per bucket and the leftovers are dropped. If your batch size is
4 and one aspect bucket holds 3 images, that bucket produces no batches at all:
**those three images never train, every epoch, and nothing says so.** It is worst
on small datasets, which is most datasets, and on the odd shapes you deliberately
included because they were interesting.

Everything on this page is off by default. With no tiers configured and the
tolerance at 0, aspect bucketing builds the same buckets it always has.

## Seeing it first

Turn aspect ratio bucketing on and start a run. Before training begins, the
console prints the population of every aspect rung:

```
[AspectBucketing] aspect-rung populations (128 budget images across 7 rungs):
     1.00:1        61  ########################################
     1:1.25        24  ################
     1.25:1        19  ############
     1:1.50        12  ########
     1.50:1         9  ######
     1:2.00         2  #
     2.00:1         1  #
```

With a batch size of 4, the bottom two rows are 3 images that contribute nothing.
Read the histogram before changing anything: it tells you whether you have a
problem worth solving and which rungs it is in.

## Bucket Aspect Tolerance

*Data tab → Bucket Aspect Tolerance. Default `0`.*

The simplest fix is to have fewer, fuller buckets. Two buckets at 1.33:1 and
1.31:1 are not meaningfully different crops, but they are different batches.
The tolerance collapses buckets whose aspect ratios differ by no more than that
amount onto a single crop resolution.

`0` is the exact bucket set aspect bucketing has always produced. `0.1` is a
reasonable first try. Larger values fill batches more easily and crop some images
a little further from their true shape; a group is bounded by the tolerance you
asked for, so `0.1` never spans more than 0.1 of aspect.

## Min Bucket Tiers

*Data tab → Min Bucket Tiers → configure tiers.*

Where the tolerance changes the buckets, tiers change what happens to a bucket
that is still too small. Each tier is a rule: *buckets holding fewer than
`Max Size` images get this strategy*. Tiers are checked smallest `Max Size` first
and the first match wins, so

| Max Size | Strategy |
|---|---|
| 2 | drop |
| 6 | donate |
| 10 | borrow |

reads as: a bucket with 1 image is dropped, 2–5 donate, 6–9 borrow, and 10 or
more are left alone.

| Strategy | What it does |
|---|---|
| `drop` | Discard the bucket. This is what already happens; naming it makes the choice explicit and stops a larger tier from catching those buckets. |
| `donate` | Move the images into the nearest surviving bucket. They train at a slightly wrong crop rather than not at all. |
| `borrow` | Pull images in from neighbouring buckets until this one fills a whole number of batches. Donors give up at most one batch each. |
| `repeat` | Oversample the bucket's own images until it fills a batch. |

### Borrow: move or copy

`Mode` applies to `borrow` only.

- **move** — the donor hands the image over. It trains at the borrowing bucket's
  crop instead of its own. Your epoch length does not change.
- **copy** — the donor keeps its image *and* a second, re-cropped copy is added to
  the borrowing bucket. The image trains at both crops. This costs a longer epoch
  and a larger latent cache, and it gives the borrowing bucket real images rather
  than taking them from somewhere else.

`copy` is the only strategy that changes how many rows your dataset has, so it
does its planning before images are loaded and mints the duplicates there. The
copies flow through cropping and caching as ordinary items.

### Which to pick

- A handful of images in one odd shape, and you care about that shape: `borrow`
  with `copy`.
- The odd shapes are incidental and you would rather not train on stretched
  crops: `drop`, so it is at least a decision.
- Everything is close to square already: raise the tolerance and you may need no
  tiers at all.

## Resolution Mode

*Data tab → Resolution Mode. Default `Split`.*

Only relevant when your training resolution lists more than one value
(`512, 768`, say).

- **Split** — each image picks one of the resolutions at random. One aspect ends
  up scattered across every resolution, which is precisely how a bucket that
  looked big enough turns into several that are not. The number of steps in an
  epoch also moves with the seed.
- **Rotate** — the whole epoch trains at one resolution, and the resolution
  rotates as the run advances. Nothing fragments, the step count per epoch is
  constant, and you still get multi-scale coverage — across the run rather than
  inside each epoch.

## Caches

Changing any of these changes which crop an image lands in, and a cached latent
belongs to its crop. Clear the cache after changing the tolerance, the tiers, or
the resolution mode.
