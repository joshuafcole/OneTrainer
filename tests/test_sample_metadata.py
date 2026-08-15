"""Behavioural tests for sample provenance (modules/util/sample_metadata.py).

The module is pure stdlib + PIL (no torch/diffusers), so these run without the
training stack: ``python tests/test_sample_metadata.py`` or under pytest. They
guard the PNG text-chunk contract that lets a sample image be identified,
outside the workspace, by the training state that produced it.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.sample_metadata import SampleProvenance, build_png_info, hash_text, provenance_fields

from PIL import Image


def _prov(**overrides):
    base = {
        "global_step": 100,
        "epoch": 2,
        "epoch_step": 5,
        "seed": 42,
        "prompt_hash": hash_text("a cat"),
        "sample_config_hash": hash_text("some config"),
        "last_save_filename": "prefix-100-2-5.safetensors",
    }
    base.update(overrides)
    return SampleProvenance(**base)


def test_hash_text_is_stable_and_16_hex():
    a = hash_text("a cat sitting on a mat")
    b = hash_text("a cat sitting on a mat")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_hash_text_distinguishes_input():
    assert hash_text("a cat") != hash_text("a dog")


def test_provenance_fields_stringifies_and_namespaces():
    fields = provenance_fields(_prov())
    assert fields["ot.global_step"] == "100"
    assert fields["ot.epoch"] == "2"
    assert fields["ot.epoch_step"] == "5"
    assert fields["ot.seed"] == "42"
    assert fields["ot.last_save_filename"] == "prefix-100-2-5.safetensors"
    assert all(key.startswith("ot.") for key in fields)


def test_provenance_fields_omits_none_rather_than_stringifying_it():
    fields = provenance_fields(_prov(seed=None, last_save_filename=None))
    assert "ot.seed" not in fields
    assert "ot.last_save_filename" not in fields
    # unaffected fields are still present
    assert fields["ot.global_step"] == "100"


def test_png_round_trip_carries_every_field():
    prov = _prov()
    info = build_png_info(prov)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "sample.png")
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        image.save(path, format="PNG", pnginfo=info)

        reopened = Image.open(path)
        reopened.load()  # tEXt chunks are only populated after a full load

        # A literal expected map, not `provenance_fields(prov)` -- deriving
        # "expected" from the function under test would let a key it silently
        # drops disappear from both sides at once and still pass.
        expected = {
            "ot.global_step": "100",
            "ot.epoch": "2",
            "ot.epoch_step": "5",
            "ot.seed": "42",
            "ot.prompt_hash": prov.prompt_hash,
            "ot.sample_config_hash": prov.sample_config_hash,
            "ot.last_save_filename": "prefix-100-2-5.safetensors",
        }
        for key, value in expected.items():
            assert reopened.text.get(key) == value, f"{key} missing or wrong after round-trip"


def test_png_round_trip_omits_none_fields():
    prov = _prov(seed=None, last_save_filename=None)
    info = build_png_info(prov)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "sample.png")
        image = Image.new("RGB", (4, 4), color=(1, 2, 3))
        image.save(path, format="PNG", pnginfo=info)

        reopened = Image.open(path)
        reopened.load()

        assert "ot.seed" not in reopened.text
        assert "ot.last_save_filename" not in reopened.text


def test_non_png_save_is_unaffected_by_provenance():
    # JPEG (and any other format) is saved exactly as before: no pnginfo kwarg,
    # no re-encoding to attach metadata. build_png_info is never required to
    # perform a plain save -- a caller can save without calling it at all.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "sample.jpg")
        image = Image.new("RGB", (4, 4), color=(4, 5, 6))
        image.save(path, format="JPEG")  # no pnginfo kwarg -- JPEG doesn't accept one

        reopened = Image.open(path)
        reopened.load()
        assert reopened.format == "JPEG"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
