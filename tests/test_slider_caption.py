"""The caption contract for coordinate-labeled image sliders.

This is the surface a user touches by hand: they type coordinates into captions
and trust that (a) the coordinate is read the way they wrote it and (b) nothing
else in the caption is disturbed. Both halves are pinned here.

``slider_caption_util`` is torch-free, so this file imports nothing but the
parser -- no model, no MGDS, no CUDA. Run with
``python -m pytest tests/test_slider_caption.py``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.slider_caption_util import parse_slider_coordinates  # noqa: E402

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
