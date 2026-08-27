"""The dataset fingerprints (modules/util/dataset_key.py).

These guard the property that lets a non-cleared cache be reused safely: the
fingerprint must move when -- and only when -- the dataset content behind a
cached tensor moves, and the media/caption split must hold, so a caption edit
never drags a whole dataset back through the VAE.

Pure stdlib plus a temp directory: no model, no torch, no GPU.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.dataset_key import dataset_fingerprints


def _concept(path, **overrides) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.path = path
    for key, value in overrides.items():
        setattr(concept, key, value)
    return concept


def _write(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(content, bytes) else "w") as fh:
        fh.write(content)


def _dataset(tmp_dir, images=("a.png", "b.png"), captions=None):
    """A concept dir with images and, by default, a caption per image."""
    concept_dir = os.path.join(tmp_dir, "concept")
    for i, name in enumerate(images):
        _write(os.path.join(concept_dir, name), bytes([i]) * (10 + i))
    if captions is None:
        captions = {f"{os.path.splitext(n)[0]}.txt": f"caption {n}" for n in images}
    for name, text in captions.items():
        _write(os.path.join(concept_dir, name), text)
    return concept_dir


def _fingerprint(concept_dir, **overrides):
    return dataset_fingerprints([_concept(concept_dir, **overrides)])


def test_a_dataset_with_no_rows_fingerprints_the_same_however_it_got_there():
    # No concepts, and concepts that contribute nothing, hold the same tensors --
    # so they must not be handed separate cache directories.
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert dataset_fingerprints([]) == _fingerprint(_dataset(tmp_dir), enabled=False)


def test_fingerprints_are_stable_and_16_hex():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        media, captions = _fingerprint(concept_dir)
        assert (media, captions) == _fingerprint(concept_dir), "same dataset, same fingerprint"
        for value in (media, captions):
            assert len(value) == 16 and all(c in "0123456789abcdef" for c in value)


def test_media_and_captions_do_not_collapse_into_one():
    with tempfile.TemporaryDirectory() as tmp_dir:
        media, captions = _fingerprint(_dataset(tmp_dir))
        assert media != captions


def test_an_added_image_moves_both_fingerprints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "c.png"), b"new image")
        after = _fingerprint(concept_dir)
        assert before[0] != after[0], "an added image must move the media fingerprint"
        assert before[1] != after[1], "an added image adds a row, so captions move too"


def test_a_removed_image_moves_both_fingerprints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        os.remove(os.path.join(concept_dir, "b.png"))
        after = _fingerprint(concept_dir)
        assert before[0] != after[0] and before[1] != after[1]


def test_a_replaced_image_moves_only_the_media_fingerprint():
    """Same filename, different bytes -- the case a row count can never reveal."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a.png"), b"different content entirely")
        after = _fingerprint(concept_dir)
        assert before[0] != after[0], "replaced image bytes must move the media fingerprint"
        assert before[1] == after[1], "its caption did not change, so text must not re-encode"


def test_a_same_length_caption_reword_moves_only_the_caption_fingerprint():
    """"cat" -> "dog": an ordinary edit a size check cannot see, and one that must
    not push the dataset back through the VAE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",), captions={"a.txt": "a cat"})
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a.txt"), "a dog")
        after = _fingerprint(concept_dir)
        assert before[1] != after[1], "a same-length reword must move the caption fingerprint"
        assert before[0] == after[0], "a caption edit must never move the media fingerprint"


def test_a_same_size_image_replacement_moves_the_media_fingerprint():
    """Media are identified by size and mtime, not by their bytes -- hashing every
    image on every launch would cost more than the re-encode the cache saves. The
    mtime half is what covers a replacement that happens to keep the same size, so
    it is set explicitly here rather than left to filesystem timestamp resolution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        image = os.path.join(concept_dir, "a.png")
        os.utime(image, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
        before = _fingerprint(concept_dir)

        _write(image, b"z" * os.path.getsize(image))  # same size, different bytes
        os.utime(image, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
        assert _fingerprint(concept_dir)[0] != before[0]


def test_a_mask_image_moves_the_media_fingerprint():
    """Mask and conditioning images sit in the concept dir under a postfix and are
    cached as tensors too (latent_mask, latent_conditioning_image), so editing one
    must invalidate the latent cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        _write(os.path.join(concept_dir, "a-masklabel.png"), b"mask")
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a-masklabel.png"), b"different mask!")
        assert _fingerprint(concept_dir)[0] != before[0]


def test_a_non_training_file_is_ignored():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "notes.md"), "not a training input")
        assert _fingerprint(concept_dir) == before


def test_subdirectories_are_walked_only_when_the_concept_says_so():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        flat_before = _fingerprint(concept_dir, include_subdirectories=False)
        deep_before = _fingerprint(concept_dir, include_subdirectories=True)
        _write(os.path.join(concept_dir, "nested", "d.png"), b"nested image")
        assert _fingerprint(concept_dir, include_subdirectories=False) == flat_before, \
            "a nested file must be invisible when the concept does not recurse"
        assert _fingerprint(concept_dir, include_subdirectories=True) != deep_before, \
            "a nested file must be visible when the concept recurses"


def test_a_disabled_concept_is_neither_walked_nor_counted():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        enabled = _fingerprint(concept_dir, enabled=True)
        disabled = _fingerprint(concept_dir, enabled=False)
        assert enabled != disabled, "enabling a concept adds rows, so it must move"

        # It produces no rows, so editing its files must be invisible ...
        _write(os.path.join(concept_dir, "a.png"), b"changed while disabled")
        assert _fingerprint(concept_dir, enabled=False) == disabled

        # ... and so must be indistinguishable from not listing it at all, which is
        # the cache with the same contents.
        assert disabled == dataset_fingerprints([])
        other_dir = _dataset(tmp_dir + "-other")
        assert _fingerprint(other_dir, enabled=False) == disabled


def test_an_unlistable_path_is_distinct_from_another_concepts_empty_directory():
    """A path typo must not reuse a different concept's cache. The path is in the
    header, so an unlistable directory needs no marker of its own to stay apart."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_dir = os.path.join(tmp_dir, "empty")
        os.makedirs(empty_dir)
        missing = _fingerprint(os.path.join(tmp_dir, "does-not-exist"))
        assert missing != _fingerprint(empty_dir)
        assert missing != dataset_fingerprints([])


def test_concept_order_is_significant():
    # Rows are cached positionally per concept, so the concept list is ordered
    # data, not a set: swapping two concepts renumbers every row after the first.
    with tempfile.TemporaryDirectory() as tmp_dir:
        first = _dataset(tmp_dir, images=("a.png",))
        second = os.path.join(tmp_dir, "second")
        _write(os.path.join(second, "b.png"), b"other")
        forward = dataset_fingerprints([_concept(first), _concept(second)])
        reverse = dataset_fingerprints([_concept(second), _concept(first)])
        assert forward != reverse


def test_the_walk_is_not_memoized_across_calls():
    """OneTrainer's UI is long-lived and can start a second run after the dataset
    was edited. A process-level memo would answer that second run with the first
    run's fingerprint, which is the exact failure this module exists to prevent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a.png"), b"edited between two runs")
        assert _fingerprint(concept_dir) != before
