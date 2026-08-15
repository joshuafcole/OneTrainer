"""Behavioural tests for the dataset fingerprints (modules/util/dataset_key.py).

These guard the property that lets a non-cleared cache be reused safely: the
fingerprint must move when (and only when) the dataset content behind a cached
tensor moves — and the media/caption split must hold, so a caption edit never drags
a whole dataset back through the VAE.

Pure stdlib + a temp directory: ``python tests/test_dataset_key.py`` or under pytest.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.dataset_key import dataset_fingerprints, reset_memo


class _Concept:
    """Duck-typed stand-in for ConceptConfig — only the three fields that decide
    which files a concept contributes."""

    def __init__(self, path, enabled=True, include_subdirectories=False):
        self.path = path
        self.enabled = enabled
        self.include_subdirectories = include_subdirectories


def _write(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(content)


def _dataset(tmp_dir, images=("a.png", "b.png"), captions=None):
    """A concept dir with images and (by default) a caption per image."""
    concept_dir = os.path.join(tmp_dir, "concept")
    for i, name in enumerate(images):
        _write(os.path.join(concept_dir, name), bytes([i]) * (10 + i))
    captions = {f"{os.path.splitext(n)[0]}.txt": f"caption {n}" for n in images} \
        if captions is None else captions
    for name, text in captions.items():
        _write(os.path.join(concept_dir, name), text)
    return concept_dir


def _fingerprint(concept_dir, **kwargs):
    reset_memo()  # these tests edit a dataset in place; the process memo must not hide it
    return dataset_fingerprints([_Concept(concept_dir, **kwargs)])


def test_no_concepts_yields_no_fingerprint():
    # A config with no dataset information must salt byte-identically to one built
    # before dataset fingerprinting existed — no spurious re-cache.
    assert dataset_fingerprints(None) == (None, None)
    assert dataset_fingerprints([]) == (None, None)


def test_fingerprints_are_stable_and_16_hex():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        media, captions = _fingerprint(concept_dir)
        assert (media, captions) == _fingerprint(concept_dir), "same dataset, same fingerprint"
        for value in (media, captions):
            assert len(value) == 16 and all(c in "0123456789abcdef" for c in value)


def test_media_and_captions_differ():
    with tempfile.TemporaryDirectory() as tmp_dir:
        media, captions = _fingerprint(_dataset(tmp_dir))
        assert media != captions, "the two fingerprints must not collapse into one"


def test_added_image_moves_both_fingerprints():
    """The reported crash: an image added to a concept whose path did not change."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "c.png"), b"new image")
        after = _fingerprint(concept_dir)
        assert before[0] != after[0], "an added image must move the media fingerprint"
        assert before[1] != after[1], "an added image adds a row, so captions move too"


def test_removed_image_moves_both_fingerprints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        os.remove(os.path.join(concept_dir, "b.png"))
        after = _fingerprint(concept_dir)
        assert before[0] != after[0] and before[1] != after[1]


def test_replaced_image_moves_only_media():
    """Same filename, different bytes — the case a row count can never reveal."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a.png"), b"different content entirely")
        after = _fingerprint(concept_dir)
        assert before[0] != after[0], "replaced image bytes must move the media fingerprint"
        assert before[1] == after[1], "its caption did not change, so text must not re-encode"


def test_same_length_caption_reword_moves_only_captions():
    """"cat" -> "dog": an ordinary edit that a size check cannot see, and one that
    must NOT push the dataset back through the VAE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",), captions={"a.txt": "a cat"})
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "a.txt"), "a dog")
        after = _fingerprint(concept_dir)
        assert before[1] != after[1], "a same-length reword must move the caption fingerprint"
        assert before[0] == after[0], "a caption edit must never move the media fingerprint"


def test_unrelated_file_is_ignored():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        before = _fingerprint(concept_dir)
        _write(os.path.join(concept_dir, "notes.md"), "not a training input")
        assert _fingerprint(concept_dir) == before, "a non-input file must not move either salt"


def test_subdirectories_respect_the_concept_flag():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        flat_before = _fingerprint(concept_dir, include_subdirectories=False)
        deep_before = _fingerprint(concept_dir, include_subdirectories=True)
        _write(os.path.join(concept_dir, "nested", "d.png"), b"nested image")
        assert _fingerprint(concept_dir, include_subdirectories=False) == flat_before, \
            "a nested file must be invisible when the concept does not recurse"
        assert _fingerprint(concept_dir, include_subdirectories=True) != deep_before, \
            "a nested file must be visible when the concept recurses"


def test_disabled_concept_is_not_walked_but_still_counts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir)
        enabled = _fingerprint(concept_dir, enabled=True)
        disabled = _fingerprint(concept_dir, enabled=False)
        assert enabled != disabled, "toggling a concept must move the salt"
        # a disabled concept contributes no file content, so editing it is invisible
        _write(os.path.join(concept_dir, "a.png"), b"changed while disabled")
        assert _fingerprint(concept_dir, enabled=False) == disabled


def test_missing_path_is_distinct_from_empty_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_dir = os.path.join(tmp_dir, "empty")
        os.makedirs(empty_dir)
        missing = _fingerprint(os.path.join(tmp_dir, "does-not-exist"))
        empty = _fingerprint(empty_dir)
        assert missing != empty, "a path typo must not reuse an empty concept's cache"


def test_concept_order_is_significant():
    # Rows are cached positionally per concept, so the concept list is ordered data,
    # not a set — swapping two concepts renumbers every row after the first.
    with tempfile.TemporaryDirectory() as tmp_dir:
        first = _dataset(tmp_dir, images=("a.png",))
        second = os.path.join(tmp_dir, "second")
        _write(os.path.join(second, "b.png"), b"other")
        reset_memo()
        forward = dataset_fingerprints([_Concept(first), _Concept(second)])
        reset_memo()
        reverse = dataset_fingerprints([_Concept(second), _Concept(first)])
        assert forward != reverse


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
