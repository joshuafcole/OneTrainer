"""Behavioural tests for sample provenance (modules/util/sample_metadata.py).

The module is pure stdlib + PIL (no torch/diffusers), so these run without the
training stack: ``python tests/test_sample_metadata.py`` or under pytest. They
guard the contract that lets a sample image be identified, outside the
workspace, by the training state that produced it -- in *both* containers, PNG
text chunks and JPEG EXIF, since JPG is the format real runs actually use.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.sample_metadata import (
    SampleProvenance,
    build_exif,
    build_png_info,
    hash_text,
    provenance_fields,
    read_provenance_fields,
)

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


def test_jpeg_round_trip_carries_every_field():
    # JPG is sample_image_format's default and what every real workspace config
    # sets, so this is the path that actually runs -- a PNG-only stamp would be
    # inert on the runs this exists for.
    prov = _prov()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "sample.jpg")
        image = Image.new("RGB", (4, 4), color=(4, 5, 6))
        image.save(path, format="JPEG", exif=build_exif(prov))

        reopened = Image.open(path)
        reopened.load()
        assert reopened.format == "JPEG"

        # Literal expected map, for the same reason as the PNG round-trip.
        expected = {
            "ot.global_step": "100",
            "ot.epoch": "2",
            "ot.epoch_step": "5",
            "ot.seed": "42",
            "ot.prompt_hash": prov.prompt_hash,
            "ot.sample_config_hash": prov.sample_config_hash,
            "ot.last_save_filename": "prefix-100-2-5.safetensors",
        }
        assert read_provenance_fields(reopened) == expected


def test_jpeg_round_trip_omits_none_fields():
    prov = _prov(seed=None, last_save_filename=None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "sample.jpg")
        image = Image.new("RGB", (4, 4), color=(4, 5, 6))
        image.save(path, format="JPEG", exif=build_exif(prov))

        fields = read_provenance_fields(Image.open(path))
        assert "ot.seed" not in fields
        assert "ot.last_save_filename" not in fields
        assert fields["ot.global_step"] == "100"


def test_reader_recovers_the_same_map_from_both_formats():
    # The two containers are an encoding detail; a consumer downstream should
    # not have to know which one it is holding.
    prov = _prov()
    with tempfile.TemporaryDirectory() as tmp_dir:
        png_path = os.path.join(tmp_dir, "sample.png")
        jpg_path = os.path.join(tmp_dir, "sample.jpg")
        image = Image.new("RGB", (4, 4), color=(7, 8, 9))
        image.save(png_path, format="PNG", pnginfo=build_png_info(prov))
        image.save(jpg_path, format="JPEG", exif=build_exif(prov))

        from_png = read_provenance_fields(Image.open(png_path))
        from_jpg = read_provenance_fields(Image.open(jpg_path))
        assert from_png == from_jpg
        assert from_png["ot.global_step"] == "100"


def test_reader_returns_empty_for_an_unstamped_image():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, fmt in (("plain.png", "PNG"), ("plain.jpg", "JPEG")):
            path = os.path.join(tmp_dir, name)
            Image.new("RGB", (4, 4), color=(1, 1, 1)).save(path, format=fmt)
            assert read_provenance_fields(Image.open(path)) == {}


def test_reader_ignores_a_foreign_image_description():
    # An unrelated caption in ImageDescription reads as "no provenance", not as
    # garbage -- sample images get handled by other tools too.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "captioned.jpg")
        exif = Image.Exif()
        exif[0x010E] = "a photo of a cat, by some other tool"
        Image.new("RGB", (4, 4), color=(2, 2, 2)).save(path, format="JPEG", exif=exif)
        assert read_provenance_fields(Image.open(path)) == {}


def test_exif_bytes_are_stable_for_identical_provenance():
    # Two samples of identical training state produce identical provenance
    # bytes -- the property that makes drift detectable by comparison.
    assert build_exif(_prov()).tobytes() == build_exif(_prov()).tobytes()


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
