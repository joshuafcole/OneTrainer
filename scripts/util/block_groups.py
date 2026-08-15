"""The block-group taxonomy for anima's Cosmos-Predict2 DiT (cinema-studio phase 455).

455 ablates *groups of blocks* -- scale one group's adapter contribution while
holding the rest at 1.0 -- to find which weights drive quality along a probe
prompt. The groups it ablates over, and the ones 456's CMA-ES later searches
over, are defined here as a resolver over a config file
(``block_groups.json`` by default), **not** as a hardcoded vocabulary: the
taxonomy is data, this module only derives group membership from it.

Anatomy (verified against ``BaseAnimaSetup.LAYER_PRESETS`` and the diffusers
``CosmosTransformerBlock``)
-----------------------------------------------------------------------------
A LoRA-adapted layer's key takes the form
``transformer.transformer_blocks.<i>.<remainder>.lora_{down,up}.weight`` (or
``.alpha``); *layer prefix* means everything before those suffixes, which is
what ``lora_soup.block_scale_for`` matches its ``fnmatch`` patterns against.

Within one ``CosmosTransformerBlock`` the LoRA-decomposable sublayers are:

- ``attn1.to_{q,k,v}`` / ``attn1.to_out.0``  -- self-attention
- ``attn2.to_{q,k,v}`` / ``attn2.to_out.0``  -- cross-attention (anima runs
  Cosmos without image-context conditioning, so ``attn2`` never grows
  ``add_*_proj`` sublayers; only the ``to_*`` convention applies)
- ``ff.net.0.proj`` / ``ff.net.2``           -- the feedforward/MLP
- ``norm{1,2,3}.linear_{1,2}``               -- the three adaLN modulation
  blocks' Linear sublayers

``attn1.norm_q`` / ``attn1.norm_k`` (the attention module's own QK-norm) are
**not** LoRA-decomposable -- they are plain (RMS/Layer)Norm, not ``nn.Linear``
-- so they never appear in a real adapted key set. The exclusion below is
carried anyway, reproducing ``LAYER_PRESETS``'s ``^(?=.*attn)(?!.*norm).*``
intent verbatim, so a stray norm-flavored key is refused rather than
silently absorbed into an attention group if one ever does turn up (a
manually edited file, a future qk_norm=False path, ...).

Depth bands
-----------
"Early/mid/late thirds" only means something against the block *count* of
the checkpoint in hand, and that varies by model config -- so the config
declares how many bands and names them, and ``band_for_index`` derives the
actual index ranges at resolve time. See its docstring for the rounding
rule.

Groups are the cross product of band x part, named ``<band>.<part>``, e.g.
``"early.attn-self"``. A group with zero members in a given checkpoint is not
an error -- ``group_layers`` still reports it, with an empty list.

Usable as a library (``load_groups`` / ``assign`` / ``group_layers`` /
``patterns_for`` / ``coverage``) by 456's search, and as a script::

    python scripts/util/block_groups.py CHECKPOINT.safetensors
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

DEFAULT_CONFIG_PATH = Path(__file__).with_name("block_groups.json")


class BlockGroupError(Exception):
    """A malformed config or an unresolvable group name."""


@dataclasses.dataclass(frozen=True)
class PartRule:
    """One ``parts`` entry: how to recognize, and how to select, one within-
    block part (e.g. ``attn-self``)."""

    name: str
    # Matched against the *remainder* -- the prefix with the
    # ``<block_prefix>.<i>.`` head stripped -- via re.search (anchored ``^``
    # patterns are how the config declares "whole remainder", not this
    # module).
    regex: re.Pattern[str]
    # The fnmatch glob fragment, over the same remainder, that ``patterns_for``
    # splices after ``<block_prefix>.<i>.`` for lora_soup's --block-scale.
    # Kept in the config rather than derived from ``regex`` because fnmatch
    # and re are different languages: the regex's negative lookahead
    # (exclude *norm*) has no fnmatch equivalent, so the glob instead relies
    # on the ``to_*`` / ``linear_*`` naming convention already covering
    # exactly the LoRA-decomposable sublayers and nothing else.
    glob: str


@dataclasses.dataclass(frozen=True)
class GroupConfig:
    model: str
    block_prefix: str
    band_count: int
    band_names: tuple[str, ...]
    parts: tuple[PartRule, ...]
    block_regex: re.Pattern[str]


def load_groups(path: str | Path | None = None) -> GroupConfig:
    """Load and validate a block-group config. Defaults to the JSON file
    shipped beside this module."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = json.loads(config_path.read_text())

    try:
        model = data["model"]
        block_prefix = data["block_prefix"]
        band_count = int(data["depth_bands"]["count"])
        band_names = tuple(data["depth_bands"]["names"])
        raw_parts = data["parts"]
    except KeyError as e:
        raise BlockGroupError(f"{config_path}: missing key {e}") from e

    if len(band_names) != band_count:
        raise BlockGroupError(
            f"{config_path}: depth_bands.count={band_count} but {len(band_names)} name(s) given"
        )
    if band_count < 1:
        raise BlockGroupError(f"{config_path}: depth_bands.count must be >= 1, got {band_count}")
    if not raw_parts:
        raise BlockGroupError(f"{config_path}: parts must be non-empty")

    parts = tuple(
        PartRule(name=name, regex=re.compile(rule["regex"]), glob=rule["glob"])
        for name, rule in raw_parts.items()
    )
    block_regex = re.compile(rf"^{re.escape(block_prefix)}\.(\d+)\.(.+)$")

    return GroupConfig(
        model=model,
        block_prefix=block_prefix,
        band_count=band_count,
        band_names=band_names,
        parts=parts,
        block_regex=block_regex,
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
    every index lands in the last band -- still total, just concentrated;
    ``group_layers`` is what makes an empty band legible rather than a bug.
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


def _matching_part_names(remainder: str, config: GroupConfig) -> list[str]:
    return [part.name for part in config.parts if part.regex.search(remainder)]


def assign(prefix: str, config: GroupConfig, block_count: int) -> str | None:
    """The group ``prefix`` belongs to, or ``None``.

    ``None`` covers two distinct misses, both meaning "not part of the
    coordinate system": ``prefix`` is not inside the configured transformer
    blocks at all (no ``<block_prefix>.<i>.`` head, or an index outside
    ``[0, block_count)``), or it is, but its remainder matches none of the
    configured parts (e.g. a stray non-LoRA-decomposable sublayer). Neither
    case is silently bucketed into some catch-all group -- ``coverage``
    is what surfaces them.

    If a remainder matches more than one part's regex, the first part listed
    in the config wins here (a stable, arbitrary tie-break for callers that
    just want *a* group) -- but that overlap is exactly what ``coverage``
    exists to catch and name; a well-formed config should never produce one.
    """
    match = config.block_regex.match(prefix)
    if match is None:
        return None
    index = int(match.group(1))
    if not (0 <= index < block_count):
        return None
    remainder = match.group(2)
    parts = _matching_part_names(remainder, config)
    if not parts:
        return None
    band = config.band_names[band_for_index(index, block_count, config.band_count)]
    return f"{band}.{parts[0]}"


def all_group_names(config: GroupConfig) -> list[str]:
    """Every group the cross product defines, band-major then part-major --
    the full ~``band_count * len(parts)`` roster, independent of any
    particular checkpoint's key set."""
    return [f"{band}.{part.name}" for band in config.band_names for part in config.parts]


def observed_block_count(prefixes: Iterable[str], config: GroupConfig) -> int:
    """``max(index) + 1`` over every prefix that names a block index, or ``0``
    if none do (an all-stray input is a legitimate, if useless, one)."""
    indices = [int(m.group(1)) for prefix in prefixes if (m := config.block_regex.match(prefix))]
    return max(indices) + 1 if indices else 0


def group_layers(prefixes: Sequence[str], config: GroupConfig) -> dict[str, list[str]]:
    """Every configured group to its member prefixes, ``block_count`` derived
    from the observed indices. Groups with no members still appear, mapped to
    an empty list -- that is a reportable fact about this checkpoint, not an
    error."""
    block_count = observed_block_count(prefixes, config)
    groups: dict[str, list[str]] = {name: [] for name in all_group_names(config)}
    for prefix in sorted(prefixes):
        group = assign(prefix, config, block_count)
        if group is not None:
            groups[group].append(prefix)
    return groups


def patterns_for(group: str, prefixes: Sequence[str], config: GroupConfig) -> list[str]:
    """The explicit ``fnmatch`` patterns for ``lora_soup --block-scale`` that
    select exactly ``group``'s members (for the ``block_count`` observed in
    ``prefixes``) and nothing else.

    Bands are index-derived, not spelled anywhere in a real key -- a single
    glob like ``'*early*'`` would pretend the band name is a name prefix and
    match nothing. So this enumerates one pattern per block index in the
    band instead, each pattern splicing the part's glob after that index:
    ``'transformer.transformer_blocks.5.attn1.to_*'``, one per index, not one
    for the whole band.
    """
    band_name, _, part_name = group.partition(".")
    if band_name not in config.band_names or not part_name:
        raise BlockGroupError(f"not a valid group name: {group!r}")
    part = next((p for p in config.parts if p.name == part_name), None)
    if part is None:
        raise BlockGroupError(f"not a valid group name: {group!r}")

    block_count = observed_block_count(prefixes, config)
    band_index = config.band_names.index(band_name)
    indices = [i for i in range(block_count) if band_for_index(i, block_count, config.band_count) == band_index]
    return [f"{config.block_prefix}.{i}.{part.glob}" for i in indices]


@dataclasses.dataclass(frozen=True)
class CoverageReport:
    """The exact-once check: every prefix should resolve to exactly one
    group. ``unassigned`` and ``multiply_assigned`` name the prefixes that
    don't, rather than only counting them."""

    total: int
    covered: int
    unassigned: tuple[str, ...]
    multiply_assigned: dict[str, tuple[str, ...]]

    @property
    def exact_once(self) -> bool:
        return not self.unassigned and not self.multiply_assigned


def coverage(prefixes: Sequence[str], config: GroupConfig) -> CoverageReport:
    """Check every prefix in ``prefixes`` resolves to exactly one group.

    Unlike ``assign`` (which tie-breaks an overlap to its first-listed part
    so callers always get *a* answer), this reports an overlap as
    ``multiply_assigned`` instead of resolving it -- the whole point of an
    exact-once check is to not paper over that case.
    """
    block_count = observed_block_count(prefixes, config)
    unassigned: list[str] = []
    multiply_assigned: dict[str, tuple[str, ...]] = {}
    covered = 0

    for prefix in prefixes:
        match = config.block_regex.match(prefix)
        index = int(match.group(1)) if match else None
        if match is None or index is None or not (0 <= index < block_count):
            unassigned.append(prefix)
            continue
        remainder = match.group(2)
        parts = _matching_part_names(remainder, config)
        if not parts:
            unassigned.append(prefix)
        elif len(parts) > 1:
            multiply_assigned[prefix] = tuple(parts)
        else:
            covered += 1

    return CoverageReport(
        total=len(prefixes),
        covered=covered,
        unassigned=tuple(sorted(unassigned)),
        multiply_assigned=multiply_assigned,
    )


def _layer_prefixes_from_keys(keys: Iterable[str]) -> list[str]:
    """Every ``lora_down``/``lora_up``/``alpha`` key collapsed to its shared
    layer prefix (de-duplicated: a layer names one prefix across all three
    suffixes)."""
    prefixes: set[str] = set()
    for key in keys:
        for suffix in (DOWN_SUFFIX, UP_SUFFIX, ALPHA_SUFFIX):
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return sorted(prefixes)


def read_layer_prefixes(path: str | Path) -> list[str]:
    """The layer-prefix set of a ``.safetensors`` LoRA file, read from its
    key set only -- ``safe_open`` never materializes a tensor for this."""
    with safe_open(str(path), framework="pt") as f:
        return _layer_prefixes_from_keys(f.keys())  # noqa: SIM118 -- safe_open is not a Mapping


def _print_report(prefixes: Sequence[str], config: GroupConfig) -> None:
    groups = group_layers(prefixes, config)
    name_width = max((len(name) for name in groups), default=0)
    for name in sorted(groups):
        print(f"  {name:<{name_width}}  {len(groups[name])}")

    report = coverage(prefixes, config)
    print()
    if report.exact_once:
        print(f"coverage: OK -- {report.covered}/{report.total} layer(s) covered exactly once")
    else:
        print(f"coverage: FAILED -- {report.covered}/{report.total} layer(s) covered exactly once")
        if report.unassigned:
            print(f"  unassigned ({len(report.unassigned)}):")
            for prefix in report.unassigned:
                print(f"    {prefix}")
        if report.multiply_assigned:
            print(f"  multiply-assigned ({len(report.multiply_assigned)}):")
            for prefix, parts in sorted(report.multiply_assigned.items()):
                print(f"    {prefix}: {', '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group -> member-count table (plus coverage verdict) for a LoRA safetensors file's block taxonomy."
    )
    parser.add_argument("checkpoint", help="path to a .safetensors LoRA file")
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
    if not path.exists():
        sys.exit(f"block_groups: no such file: {path}")
    prefixes = read_layer_prefixes(path)
    if not prefixes:
        sys.exit(f"block_groups: no LoRA layer prefixes found in {path}")

    print(f"model:      {config.model}")
    print(f"checkpoint: {path}  ({len(prefixes)} layer(s))")
    print()
    _print_report(prefixes, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
