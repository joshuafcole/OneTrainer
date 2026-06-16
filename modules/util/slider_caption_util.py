"""Caption coordinate parsing for coordinate-labeled image sliders (docs §10).

A coordinate-labeled slider dataset uses vanilla OneTrainer concepts: the user
annotates each image's caption with the image's position on the slider axis using
an a1111-style weighted token, e.g.

    a photo of a car on a road, (distance:-2)

The training pipeline extracts the *declared-axis* coordinates out of the caption
*before* tokenization, for two reasons:

  * the coordinate becomes the training-time adapter multiplier ``m = k * value``
    (coordinate-scaled reconstruction), and
  * removing it from the conditioning keeps the caption orthogonal to the axis --
    the base never reads the attribute from the prompt, which is the one
    load-bearing slider constraint (see docs §10.0).

Only tokens whose name matches a *declared* axis are touched; ordinary a1111
emphasis tokens like ``(red car:1.2)`` pass through untouched. The value is a
plain float (ordinal recommended but not enforced; continuous / lidar-derived
labels work as-is, scaled by the per-axis gain ``k`` at training time).

This module is deliberately torch-free so it can be unit-tested without the
model/MGDS import chain.
"""

import re

# (name : value) where value is an int/float, optionally signed. The name group
# is built per-call from the declared axis names so unrelated a1111 emphasis
# tokens are never consumed.
_VALUE = r"[-+]?(?:\d+\.?\d*|\.\d+)"


def _axis_pattern(axis_names: list[str]) -> "re.Pattern[str] | None":
    names = [re.escape(n.strip()) for n in axis_names if n and n.strip()]
    if not names:
        return None
    alternation = "|".join(names)
    return re.compile(
        rf"\(\s*(?P<name>{alternation})\s*:\s*(?P<val>{_VALUE})\s*\)",
        re.IGNORECASE,
    )


def parse_slider_coordinates(prompt: str, axis_names: list[str]) -> tuple[str, dict[str, float]]:
    """Extract declared-axis coordinate tokens from ``prompt``.

    Returns ``(cleaned_prompt, {axis_name_lower: value})``. Matched tokens are
    removed from the caption and the leftover comma/whitespace tidied so the
    cleaned prompt reads naturally. Axis names are matched case-insensitively and
    returned lower-cased. The last occurrence wins if an axis is repeated.
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

    # Tidy the holes the removals leave behind: collapse the doubled separators
    # (", ,"), then strip leading/trailing separators and runs of whitespace.
    cleaned = re.sub(r"\s*,(?:\s*,)+\s*", ", ", cleaned)
    cleaned = re.sub(r"^\s*[,]\s*|\s*[,]\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, coords
