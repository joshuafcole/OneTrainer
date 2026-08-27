"""Caption coordinate parsing for coordinate-labeled image sliders.

A coordinate-labeled slider dataset is an *ordinary* OneTrainer dataset: no
paired files, no new format. The user annotates each image's caption with where
that image sits on the slider axis, written as an a1111-style weighted token::

    a photo of a car on a road, (distance:-2)

The training pipeline pulls the *declared-axis* coordinates out of the caption
before tokenization, for two reasons:

  * the coordinate becomes the training-time adapter multiplier
    ``m = gain_k * value``, and
  * removing it from the caption keeps the conditioning orthogonal to the axis.
    That is the one load-bearing constraint of the whole regime: if the caption
    still said "distance", the base could read the attribute off the prompt and
    the adapter would have nothing to learn.

Only tokens whose name matches a *declared* axis are touched. Ordinary a1111
emphasis -- ``(red car:1.2)``, ``(blurry:0.7)`` -- passes through untouched,
which is the compatibility property that makes an existing captioned dataset
usable as-is. The value is a plain signed decimal; ordinal spacing is
recommended but not enforced, so continuous labels (a measured distance, a
head-pose angle) work as authored and are rescaled by the per-axis gain at
training time rather than in the dataset.

It also owns the other half of that contract: which axes a config has declared,
and which one of them supplies the multiplier. The data loader (which strips the
tokens) and the model setup (which reads the coordinate) must agree on both, and
they are far enough apart in the pipeline that agreeing by construction beats
agreeing by convention.

This module is deliberately torch-free, so all of it can be unit-tested without
the model / MGDS import chain.
"""

import re

from modules.util.config.SliderConfig import SliderAxisConfig

# A signed decimal: 2, -2, +1.5, .75, 3. -- but not "1e-3", which a1111 does not
# accept either. The name half of the pattern is built per call from the declared
# axis names, so an emphasis token the user did not declare is never consumed.
_VALUE = r"[-+]?(?:\d+\.?\d*|\.\d+)"


def _axis_pattern(axis_names: list[str]) -> "re.Pattern[str] | None":
    # Order-preserving dedup, so the compiled pattern is the same every run. The
    # order does not affect which axis wins: alternation backtracks, so declaring
    # "age" before "age_group" still resolves "(age_group:1)" to age_group -- the
    # shorter branch matches the name but then fails on the ":" and is retried.
    # (test_a_prefix_of_an_axis_name_does_not_swallow_the_longer_one pins that.)
    names = list(dict.fromkeys(n.strip() for n in axis_names if n and n.strip()))
    if not names:
        return None
    alternation = "|".join(re.escape(name) for name in names)
    return re.compile(
        rf"\(\s*(?P<name>{alternation})\s*:\s*(?P<val>{_VALUE})\s*\)",
        re.IGNORECASE,
    )


def parse_slider_coordinates(prompt: str, axis_names: list[str]) -> tuple[str, dict[str, float]]:
    """Extract the declared-axis coordinate tokens from ``prompt``.

    Returns ``(cleaned_prompt, {axis_name_lower: value})``. Matched tokens are
    removed from the caption and the separators they leave behind are tidied, so
    the cleaned caption reads as if the coordinate had never been written. Axis
    names match case-insensitively and are returned lower-cased; if an axis
    appears more than once the last occurrence wins, which is what re-annotating
    a caption by appending plainly means.
    """
    coords: dict[str, float] = {}
    if not prompt or not axis_names:
        return prompt, coords

    pattern = _axis_pattern(axis_names)
    if pattern is None:
        return prompt, coords

    def _take(match: "re.Match[str]") -> str:
        coords[match.group("name").lower()] = float(match.group("val"))
        return ""

    cleaned = pattern.sub(_take, prompt)

    # Tidy the holes the removals leave: collapse runs of separators that are now
    # adjacent, then drop a separator stranded at either end and squeeze the
    # doubled spaces. Done in this order so ", , " becomes ", " rather than " ".
    cleaned = re.sub(r"\s*,(?:\s*,)+\s*", ", ", cleaned)
    cleaned = re.sub(r"^\s*,\s*|\s*,\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, coords


# ---------------------------------------------------------------------------
# which axes are declared, and which one drives the multiplier
# ---------------------------------------------------------------------------

def declared_axis_names(axes: list[SliderAxisConfig]) -> list[str]:
    """Every enabled axis name, in declaration order.

    These are the tokens stripped from the caption -- *all* of them, not just the
    target. Declaring a confounder the run is not training is how the user keeps
    it out of the conditioning, so a declared axis is stripped whether or not it
    is the target.
    """
    return [a.name.strip() for a in axes if a.enabled and a.name and a.name.strip()]


def resolve_target_axis(axes: list[SliderAxisConfig]) -> SliderAxisConfig:
    """The one enabled axis whose coordinate becomes the adapter multiplier.

    v1 trains a single axis per run, so "exactly one" is a real constraint and not
    a convenience: with two target axes there is no single multiplier to set, and
    quietly picking the first would train one axis while the user watched the
    other. Every failure here is a config mistake with a control to change, so the
    messages name the tab.
    """
    named = [a for a in axes if a.enabled and a.name and a.name.strip()]
    if not named:
        raise RuntimeError(
            "Coordinate image-slider training needs at least one declared axis with a name "
            "(see the Slider tab). The axis name is the caption token key -- 'distance' for "
            "a caption reading '(distance:-2)'."
        )

    targets = [a for a in named if a.is_target]
    if len(targets) == 1:
        return targets[0]
    if not targets:
        raise RuntimeError(
            "No slider axis is flagged as the target axis, so no coordinate drives the "
            f"adapter multiplier. Flag exactly one of {_quoted_names(named)} as the target "
            "on the Slider tab."
        )
    raise RuntimeError(
        f"Exactly one slider axis may be the target axis; {_quoted_names(targets)} are all "
        "flagged. A run trains one axis, and there is only one adapter multiplier to set."
    )


def _quoted_names(axes: list[SliderAxisConfig]) -> str:
    return ", ".join(repr(a.name.strip()) for a in axes)
