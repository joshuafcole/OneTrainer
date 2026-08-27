"""The caption contract for coordinate-labeled image sliders.

This is the surface a user touches by hand: they type coordinates into captions
and trust that (a) the coordinate is read the way they wrote it and (b) nothing
else in the caption is disturbed. Both halves are pinned here.

``slider_caption_util`` is torch-free, so this file imports nothing heavier than
a config class -- no model, no MGDS, no CUDA. Run with
``python -m pytest tests/test_slider_caption.py``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.config.SliderConfig import SliderAxisConfig  # noqa: E402
from modules.util.slider_caption_util import (  # noqa: E402
    declared_axis_names,
    parse_slider_coordinates,
    resolve_target_axis,
)

AXES = ["distance"]


# ---------------------------------------------------------------------------
# reading the coordinate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,expected", [
    ("(distance:-2)", -2.0),
    ("(distance:2)", 2.0),
    ("(distance:+2)", 2.0),
    ("(distance:0)", 0.0),
    ("(distance:1.5)", 1.5),
    ("(distance:-0.25)", -0.25),
    ("(distance:.5)", 0.5),
    ("( distance : -2 )", -2.0),
    ("(DISTANCE:-2)", -2.0),
    ("(Distance:-2)", -2.0),
])
def test_the_coordinate_is_read_as_written(token, expected):
    _, coords = parse_slider_coordinates(f"a photo of a car, {token}", AXES)
    assert coords == {"distance": expected}


def test_the_axis_name_is_reported_lower_cased():
    """The caption may be written in any case; every consumer keys on one."""
    _, coords = parse_slider_coordinates("a car, (DiStAnCe:1)", ["DISTANCE"])
    assert coords == {"distance": 1.0}


def test_a_repeated_axis_takes_the_last_value():
    """Re-annotating by appending is what a user plainly means."""
    cleaned, coords = parse_slider_coordinates("a car, (distance:-2), (distance:1)", AXES)
    assert coords == {"distance": 1.0}
    assert cleaned == "a car"


def test_several_axes_are_read_in_one_pass():
    cleaned, coords = parse_slider_coordinates(
        "a portrait, (distance:-2), (age:30)", ["distance", "age"])
    assert coords == {"distance": -2.0, "age": 30.0}
    assert cleaned == "a portrait"


def test_a_prefix_of_an_axis_name_does_not_swallow_the_longer_one():
    """"age" and "age_group" declared together: the token must resolve to the
    axis actually written, not to whichever alternation branch is tried first."""
    _, coords = parse_slider_coordinates("x, (age_group:3)", ["age", "age_group"])
    assert coords == {"age_group": 3.0}


# ---------------------------------------------------------------------------
# leaving the rest of the caption alone
# ---------------------------------------------------------------------------

def test_ordinary_a1111_emphasis_passes_through_untouched():
    """The compatibility property: an already-captioned dataset keeps working,
    and only the axes the user declared are ever consumed."""
    prompt = "a photo of a (red car:1.2) on a (wet road:0.8), (distance:-2)"
    cleaned, coords = parse_slider_coordinates(prompt, AXES)
    assert cleaned == "a photo of a (red car:1.2) on a (wet road:0.8)"
    assert coords == {"distance": -2.0}


def test_an_undeclared_axis_is_left_in_the_caption():
    """A typo in the axis name must not silently eat a caption token -- the
    caption is the evidence the user has that something is wrong."""
    cleaned, coords = parse_slider_coordinates("a car, (distence:-2)", AXES)
    assert cleaned == "a car, (distence:-2)"
    assert coords == {}


def test_a_bare_parenthesised_word_is_not_a_coordinate():
    cleaned, coords = parse_slider_coordinates("a car, (distance)", AXES)
    assert cleaned == "a car, (distance)"
    assert coords == {}


def test_a_non_numeric_value_is_not_a_coordinate():
    cleaned, coords = parse_slider_coordinates("a car, (distance:far)", AXES)
    assert cleaned == "a car, (distance:far)"
    assert coords == {}


@pytest.mark.parametrize("prompt,expected", [
    ("a car, (distance:-2)", "a car"),
    ("(distance:-2), a car", "a car"),
    ("a car, (distance:-2), on a road", "a car, on a road"),
    ("a car (distance:-2) on a road", "a car on a road"),
    ("(distance:-2)", ""),
    ("a car,(distance:-2),on a road", "a car, on a road"),
])
def test_the_hole_the_token_leaves_is_tidied(prompt, expected):
    """A stranded comma or a doubled space is not wrong, exactly -- but the
    cleaned caption is what gets tokenized, and it should read as if the
    coordinate had never been written."""
    cleaned, _ = parse_slider_coordinates(prompt, AXES)
    assert cleaned == expected


# ---------------------------------------------------------------------------
# the degenerate inputs the pipeline will actually hand it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis_names", [[], [""], ["   "], [None]])
def test_no_declared_axis_is_a_no_op(axis_names):
    """The IMAGE regime refuses to start without an axis, but the parser is also
    reachable from a half-filled config; it must not turn that into a crash or,
    worse, into a caption it silently rewrote."""
    prompt = "a car, (distance:-2)"
    cleaned, coords = parse_slider_coordinates(prompt, axis_names)
    assert cleaned == prompt
    assert coords == {}


def test_an_empty_caption_is_a_no_op():
    assert parse_slider_coordinates("", AXES) == ("", {})


def test_an_axis_name_with_regex_metacharacters_is_matched_literally():
    """Axis names come from a text box. "." must not become "any character"."""
    cleaned, coords = parse_slider_coordinates("a car, (a.b:1)", ["a.b"])
    assert coords == {"a.b": 1.0}
    assert cleaned == "a car"

    cleaned, coords = parse_slider_coordinates("a car, (axb:1)", ["a.b"])
    assert coords == {}
    assert cleaned == "a car, (axb:1)"


# ---------------------------------------------------------------------------
# which axes are declared, and which one drives the multiplier
# ---------------------------------------------------------------------------

def _axis(name, gain_k=1.0, is_target=True, enabled=True):
    axis = SliderAxisConfig.default_values()
    axis.name, axis.gain_k, axis.is_target, axis.enabled = name, gain_k, is_target, enabled
    return axis


def test_every_declared_axis_is_stripped_not_only_the_target():
    """Declaring a confounder is how a user keeps it out of the conditioning, so
    a non-target axis is still a name the caption pipeline removes."""
    axes = [_axis("distance"), _axis("age", is_target=False)]
    assert declared_axis_names(axes) == ["distance", "age"]


def test_a_disabled_or_unnamed_axis_declares_nothing():
    axes = [_axis("distance"), _axis("age", enabled=False), _axis("  ", is_target=False)]
    assert declared_axis_names(axes) == ["distance"]


def test_the_target_axis_is_the_one_flagged():
    axes = [_axis("age", is_target=False), _axis("distance", gain_k=0.5)]
    target = resolve_target_axis(axes)
    assert target.name == "distance" and target.gain_k == 0.5


@pytest.mark.parametrize("axes,expected", [
    ([], "at least one declared axis"),
    ([_axis("distance", enabled=False)], "at least one declared axis"),
    ([_axis("   ")], "at least one declared axis"),
    ([_axis("distance", is_target=False)], "No slider axis is flagged as the target"),
    ([_axis("distance"), _axis("age")], "Exactly one slider axis may be the target"),
])
def test_an_unusable_axis_set_is_refused_with_the_control_to_change(axes, expected):
    """Every one of these is a config mistake, and the only way a user fixes it is
    on the Slider tab -- so the message has to name it rather than describe an
    internal invariant."""
    with pytest.raises(RuntimeError, match=expected):
        resolve_target_axis(axes)


def test_the_refusal_names_the_axes_it_is_talking_about():
    with pytest.raises(RuntimeError, match="'distance', 'age'"):
        resolve_target_axis([_axis("distance"), _axis("age")])


def test_a_disabled_axis_does_not_count_as_a_second_target():
    """Switching an axis off is how a user parks it; it must not keep colliding
    with the axis they switched on."""
    axes = [_axis("distance"), _axis("age", enabled=False)]
    assert resolve_target_axis(axes).name == "distance"
