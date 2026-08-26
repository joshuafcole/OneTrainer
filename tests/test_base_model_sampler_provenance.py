"""Integration tests for the provenance *wiring* in BaseModelSampler.

``tests/test_sample_metadata.py`` proves the field map and both container
builders/readers are correct in isolation, but it never calls
``BaseModelSampler.save_sampler_output`` -- so it cannot catch a defect in how
the sampler actually invokes ``build_png_info``/``build_exif``. That gap is
exactly the shape of the bug ``9b28d426`` fixed upstream (JPEG wired up in the
field map but never wired into the save path): a change that makes the image
save path ignore provenance entirely, or that no-ops one container while
leaving the other intact, would pass every test in that file. These tests
exist to close that gap.

Imports ``torch`` (for ``torch.device``) and constructs a minimal concrete
``BaseModelSampler`` subclass -- unlike ``test_sample_metadata.py``, this file
is not torch-free. Importing ``modules.modelSampler.BaseModelSampler`` directly
(rather than ``modules.modelSampler`` or anything that pulls in
``modules.util.create``) avoids the package-wide sampler import sweep, so this
does not require a GPU or a matching diffusers version.
"""

import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.sample_metadata import SampleProvenance, hash_text, read_provenance_fields

from PIL import Image


class _FakeSampler(BaseModelSampler):
    """The abstract `sample` method is never exercised here -- only
    `save_sampler_output`, which is what carries provenance."""

    def sample(self, *args, **kwargs):
        raise NotImplementedError


def _sampler() -> _FakeSampler:
    return _FakeSampler(torch.device("cpu"), torch.device("cpu"))


def _prov(**overrides) -> SampleProvenance:
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


def _save_image(sampler: BaseModelSampler, dest: str, image_format: ImageFormat) -> Image.Image:
    output = ModelSamplerOutput(FileType.IMAGE, Image.new("RGB", (4, 4), color=(1, 2, 3)))
    sampler.save_sampler_output(output, dest, image_format, None, None)
    reopened = Image.open(dest + image_format.extension())
    reopened.load()
    return reopened


def test_png_save_stamps_provenance_when_set():
    sampler = _sampler()
    sampler.set_provenance(_prov())
    with tempfile.TemporaryDirectory() as tmp_dir:
        reopened = _save_image(sampler, os.path.join(tmp_dir, "sample"), ImageFormat.PNG)
        fields = read_provenance_fields(reopened)
        assert fields["ot.global_step"] == "100"
        assert fields["ot.seed"] == "42"


def test_jpeg_save_stamps_provenance_when_set():
    # JPG is sample_image_format's default -- the path every real run takes.
    # This is the exact case 9b28d426 fixed: a PNG-only stamp reads green on
    # test_png_save_stamps_provenance_when_set while this one stays unstamped.
    sampler = _sampler()
    sampler.set_provenance(_prov())
    with tempfile.TemporaryDirectory() as tmp_dir:
        reopened = _save_image(sampler, os.path.join(tmp_dir, "sample"), ImageFormat.JPG)
        fields = read_provenance_fields(reopened)
        assert fields["ot.global_step"] == "100"
        assert fields["ot.seed"] == "42"


def test_no_provenance_set_saves_plainly():
    sampler = _sampler()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for image_format in (ImageFormat.PNG, ImageFormat.JPG):
            reopened = _save_image(sampler, os.path.join(tmp_dir, f"sample-{image_format}"), image_format)
            assert read_provenance_fields(reopened) == {}


def test_broken_png_provenance_falls_through_to_a_plain_save():
    # A broken provenance chunk must never cost a training run its sample.
    sampler = _sampler()
    sampler.set_provenance(_prov())
    with tempfile.TemporaryDirectory() as tmp_dir, \
            patch("modules.modelSampler.BaseModelSampler.build_png_info", side_effect=RuntimeError("boom")):
        reopened = _save_image(sampler, os.path.join(tmp_dir, "sample"), ImageFormat.PNG)
        assert read_provenance_fields(reopened) == {}


def test_broken_jpeg_provenance_falls_through_to_a_plain_save():
    sampler = _sampler()
    sampler.set_provenance(_prov())
    with tempfile.TemporaryDirectory() as tmp_dir, \
            patch("modules.modelSampler.BaseModelSampler.build_exif", side_effect=RuntimeError("boom")):
        reopened = _save_image(sampler, os.path.join(tmp_dir, "sample"), ImageFormat.JPG)
        assert read_provenance_fields(reopened) == {}


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
