"""Block groups, fitted to the layers an adapter actually contains (cinema-studio 455).

455 ablates *groups of layers* — scale one group's contribution while holding
the rest at 1.0 — to find which weights drive quality along a probe prompt. This
module decides what those groups are.

Fitted, not declared
--------------------
The groups are derived from the layer prefixes **present in the checkpoint**,
not matched into a fixed taxonomy. That inversion is the whole design, and it
follows from one measurement: a real anima run trains
``layer_filter_preset="attn-only"``, so six of the twelve groups an exhaustive
band × part taxonomy defines had **zero members**, and an ablation grid over it
would have spent half its renders on groups that cannot affect anything.
Different models train different things under different filters; the coordinate
system has to be the adapter's, not the architecture's.

The consequence worth stating plainly: **coverage is total by construction.**
Every prefix lands in exactly one group because the groups are built *from* the
prefixes. "Unassigned" and "multiply-assigned" are not failure modes that got
fixed — they are no longer expressible.

The config is a naming layer
----------------------------
``block_groups.json`` no longer enumerates the taxonomy. It supplies *labels*:
patterns that recognize a leaf path and give it a human name (``attn1.* →
attn-self``), per granularity. A leaf no rule recognizes still gets a group,
named by its raw leaf path — so an unfamiliar architecture works immediately and
adding a naming section is a readability improvement, never a prerequisite.
Naming is cosmetic; membership is exact.

Granularity is the user's knob
------------------------------
``fine`` names each distinct leaf path separately (``attn1.to_q``,
``attn1.to_k``, …); ``coarse`` merges by the config's rules (``attn-self``,
``attn-cross``, …). For an attention-only adapter, ``fine`` is where the
resolution is — 3 bands × 8 leaves = 24 live groups against ``coarse``'s 6.
Deliberately explicit rather than fitted to a target count: a heuristic default
is a second thing to explain when a profile looks wrong, and can be added later
once there is evidence about the resolution an ablation actually needs.

Depth bands
-----------
"Early/mid/late" only means something against the block count of the checkpoint
in hand, so the config declares how many bands and names them and
``band_for_index`` derives the ranges at resolve time (see its rounding rule).
Layers outside the configured block prefix — a text-encoder LoRA, say — are real
and are kept, under the ``unblocked`` band, rather than being dropped to
preserve a tidy invariant.

Groups are named ``<band>.<part>``. Usable as a library (``load_groups`` /
``fit`` / ``FittedGroups.patterns_for``) by 456's search, and as a script::

    python scripts/util/block_groups.py CHECKPOINT.safetensors [--granularity fine]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from safetensors import safe_open

DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"
# LoKr names one layer under a different key family, so a LoKr checkpoint would
# otherwise report zero layers rather than an error -- an empty table that looks
# like an answer. Kept in step with lora_soup's LOKR_NAMES by hand: this module
# deliberately imports nothing from there, since it is the *taxonomy*, not the
# merge engine, and the two are used independently.
LAYER_SUFFIXES: tuple[str, ...] = (
    DOWN_SUFFIX,
    UP_SUFFIX,
    ALPHA_SUFFIX,
    ".lokr_w1", ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b",
    ".lokr_t2",
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("block_groups.json")

#: Band for layers that carry no ``<block_prefix>.<i>.`` head at all.
UNBLOCKED_BAND = "unblocked"


class BlockGroupError(Exception):
    """A malformed config, an unknown granularity, or an unresolvable group."""


@dataclasses.dataclass(frozen=True)
class NameRule:
    """One naming rule: recognize a leaf path, label it.

    Order matters — the first rule whose pattern matches wins, so a config lists
    specific rules before general ones. Unlike the taxonomy this replaced, an
    overlap is not a defect to be caught: it is a documented precedence.
    """

    label: str
    pattern: re.Pattern[str]


@dataclasses.dataclass(frozen=True)
class GroupConfig:
    model: str
    block_prefix: str
    band_count: int
    band_names: tuple[str, ...]
    granularities: dict[str, tuple[NameRule, ...]]
    default_granularity: str
    block_regex: re.Pattern[str]

    def rules_for(self, granularity: str) -> tuple[NameRule, ...]:
        if granularity not in self.granularities:
            known = ", ".join(sorted(self.granularities))
            raise BlockGroupError(f"unknown granularity {granularity!r} (known: {known})")
        return self.granularities[granularity]


def load_groups(path: str | Path | None = None) -> GroupConfig:
    """Load and validate a naming config. Defaults to the file beside this module."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = json.loads(config_path.read_text())

    try:
        model = data["model"]
        block_prefix = data["block_prefix"]
        band_count = int(data["depth_bands"]["count"])
        band_names = tuple(data["depth_bands"]["names"])
        raw_granularities = data["granularities"]
    except KeyError as e:
        raise BlockGroupError(f"{config_path}: missing key {e}") from e

    if len(band_names) != band_count:
        raise BlockGroupError(
            f"{config_path}: depth_bands.count={band_count} but {len(band_names)} name(s) given"
        )
    if band_count < 1:
        raise BlockGroupError(f"{config_path}: depth_bands.count must be >= 1, got {band_count}")
    if not raw_granularities:
        raise BlockGroupError(f"{config_path}: granularities must be non-empty")
    if UNBLOCKED_BAND in band_names:
        raise BlockGroupError(
            f"{config_path}: {UNBLOCKED_BAND!r} is reserved for layers outside the block prefix"
        )

    granularities = {
        name: tuple(
            NameRule(label=label, pattern=re.compile(pattern))
            for label, pattern in rules.items()
        )
        for name, rules in raw_granularities.items()
    }
    default_granularity = data.get("default_granularity") or next(iter(granularities))
    if default_granularity not in granularities:
        raise BlockGroupError(
            f"{config_path}: default_granularity {default_granularity!r} is not a declared granularity"
        )

    return GroupConfig(
        model=model,
        block_prefix=block_prefix,
        band_count=band_count,
        band_names=band_names,
        granularities=granularities,
        default_granularity=default_granularity,
        block_regex=re.compile(rf"^{re.escape(block_prefix)}\.(\d+)\.(.+)$"),
    )


def band_for_index(index: int, block_count: int, band_count: int) -> int:
    """Which band (0-based) block ``index`` falls in, out of ``band_count``
    bands over ``block_count`` blocks.

    Deterministic and total: every ``0 <= index < block_count`` lands in
    exactly one band. Rounding rule -- the first ``band_count - 1`` bands
    each get ``block_count // band_count`` (floor) blocks; the **last** band
    absorbs everything from there to the end, i.e. the floor division's
    remainder plus one full band's worth. E.g. 29 blocks / 3 bands: sizes
    [9, 9, 11], not [9, 9, 9] with block 28 falling off the end. When
    ``block_count < band_count`` every earlier band is empty (size 0) and
    every index lands in the last band -- still total, just concentrated.
    """
    if block_count < 1:
        raise BlockGroupError(f"block_count must be >= 1, got {block_count}")
    if not (0 <= index < block_count):
        raise BlockGroupError(f"index {index} out of range for block_count={block_count}")
    base = block_count // band_count
    last_band_start = base * (band_count - 1)
    if index >= last_band_start:
        return band_count - 1
    return index // base


def observed_block_count(prefixes: Iterable[str], config: GroupConfig) -> int:
    """``max(index) + 1`` over every prefix that names a block index, or ``0``
    if none do (an all-stray input is a legitimate, if useless, one)."""
    indices = [int(m.group(1)) for prefix in prefixes if (m := config.block_regex.match(prefix))]
    return max(indices) + 1 if indices else 0


def _label_for(leaf: str, rules: Sequence[NameRule]) -> tuple[str, bool]:
    """``(part name, was it recognized)``.

    An unrecognized leaf keeps its own path as the part name. That is what makes
    an unfamiliar architecture usable on sight — the names are ugly, the
    membership is still exact.
    """
    for rule in rules:
        if rule.pattern.search(leaf):
            return rule.label, True
    return leaf, False


@dataclasses.dataclass(frozen=True)
class FittedGroups:
    """The coordinate system for one checkpoint at one granularity.

    ``groups`` covers every input prefix exactly once — see the module
    docstring. ``unrecognized_parts`` names the parts whose label is a raw leaf
    path *despite the granularity having rules that could have named them*: not
    an error, but the signal that the naming layer has a gap for this
    architecture.

    It is empty for a rule-less granularity (``fine``), where a raw leaf name is
    the intended output rather than a miss. Reporting all sixteen leaves as
    "unrecognized" on every identity fit would make the field noise, and a
    signal that fires always carries nothing.
    """

    granularity: str
    block_count: int
    groups: dict[str, tuple[str, ...]]
    unrecognized_parts: tuple[str, ...]

    @property
    def total_layers(self) -> int:
        return sum(len(members) for members in self.groups.values())

    def patterns_for(self, group: str) -> list[str]:
        """``fnmatch`` patterns for ``lora_soup --block-scale`` selecting exactly
        this group's members.

        These are the member prefixes themselves, escaped — not globs inferred
        from a naming rule. The taxonomy this replaced had to carry a regex *and*
        a glob per part (fnmatch cannot express the regex's negative lookahead),
        which could drift apart silently, and a pattern that looks right while
        selecting the wrong layers is the whole failure mode. Membership is
        known here, so nothing has to be inferred.
        """
        if group not in self.groups:
            raise BlockGroupError(f"not a group of this fit: {group!r}")
        return [_literal_pattern(prefix) for prefix in self.groups[group]]


def _literal_pattern(prefix: str) -> str:
    """``prefix`` as an fnmatch pattern matching itself and nothing else."""
    return "".join("[[]" if ch == "[" else f"[{ch}]" if ch in "*?" else ch for ch in prefix)


def fit(
    prefixes: Sequence[str], config: GroupConfig, granularity: str | None = None
) -> FittedGroups:
    """Fit the coordinate system to the layers ``prefixes`` actually contains."""
    level = granularity or config.default_granularity
    rules = config.rules_for(level)
    block_count = observed_block_count(prefixes, config)

    groups: dict[str, list[str]] = {}
    unrecognized: dict[str, None] = {}
    for prefix in sorted(prefixes):
        match = config.block_regex.match(prefix)
        if match is None or not (0 <= int(match.group(1)) < block_count):
            # Outside the block prefix entirely (a text-encoder LoRA, say). Kept
            # rather than dropped: a layer the coordinate system cannot place is
            # still a layer the adapter trains.
            band, leaf = UNBLOCKED_BAND, prefix
        else:
            band = config.band_names[
                band_for_index(int(match.group(1)), block_count, config.band_count)
            ]
            leaf = match.group(2)
        part, recognized = _label_for(leaf, rules)
        if rules and not recognized:
            unrecognized[part] = None
        groups.setdefault(f"{band}.{part}", []).append(prefix)

    return FittedGroups(
        granularity=level,
        block_count=block_count,
        groups={name: tuple(members) for name, members in sorted(groups.items())},
        unrecognized_parts=tuple(unrecognized),
    )


def _layer_prefixes_from_keys(keys: Iterable[str]) -> list[str]:
    """Every adapter key collapsed to its shared layer prefix (de-duplicated: a
    layer names one prefix across all of its suffixes).

    LoRA's three suffixes and LoKr's seven both land here -- the taxonomy is
    over *layers*, and which factorization a layer was trained with is not one
    of its coordinates. Matched with ``endswith`` on the dotted suffix, so
    ``.lokr_w1`` cannot swallow ``.lokr_w1_a``.
    """
    prefixes: set[str] = set()
    for key in keys:
        for suffix in LAYER_SUFFIXES:
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return sorted(prefixes)


def read_layer_prefixes(path: str | Path) -> list[str]:
    """The layer-prefix set of a ``.safetensors`` LoRA/LoKr file, read from its
    key set only -- ``safe_open`` never materializes a tensor for this."""
    with safe_open(str(path), framework="pt") as f:
        return _layer_prefixes_from_keys(f.keys())  # noqa: SIM118 -- safe_open is not a Mapping


def _print_report(fitted: FittedGroups) -> None:
    name_width = max((len(name) for name in fitted.groups), default=0)
    for name, members in fitted.groups.items():
        print(f"  {name:<{name_width}}  {len(members)}")
    print()
    print(
        f"granularity: {fitted.granularity} — {len(fitted.groups)} group(s) over "
        f"{fitted.total_layers} layer(s), {fitted.block_count} block(s)"
    )
    # Totality is structural here, so it is stated rather than checked.
    print("coverage: total by construction — every layer is in exactly one group")
    if fitted.unrecognized_parts:
        print()
        print(f"unnamed parts ({len(fitted.unrecognized_parts)}) — no rule matched, raw leaf used:")
        for part in fitted.unrecognized_parts:
            print(f"    {part}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group -> member-count table for a LoRA/LoKr safetensors file, "
                    "fitted to the layers it actually contains."
    )
    parser.add_argument("checkpoint", help="path to a .safetensors LoRA/LoKr file")
    parser.add_argument(
        "--granularity", default=None, metavar="LEVEL",
        help="naming granularity (default: the config's default_granularity)",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="block_groups.json to use (default: the file beside this script)",
    )
    args = parser.parse_args()

    try:
        config = load_groups(args.config)
    except (OSError, json.JSONDecodeError, BlockGroupError) as e:
        sys.exit(f"block_groups: {e}")

    path = Path(args.checkpoint)
    if not path.is_file():
        sys.exit(f"block_groups: no such file: {path}")
    try:
        _print_report(fit(read_layer_prefixes(path), config, args.granularity))
    except BlockGroupError as e:
        sys.exit(f"block_groups: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
