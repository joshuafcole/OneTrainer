"""The dataset fingerprints (modules/util/dataset_key.py).

These guard the property that lets a non-cleared cache be reused safely: the
fingerprint must move when -- and only when -- the dataset content behind a
cached tensor moves, and the media/caption split must hold, so a caption edit
never drags a whole dataset back through the VAE.

Pure stdlib plus a temp directory: no model, no torch, no GPU.
"""

import builtins
import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.dataset_key import _RECORD_NAME, dataset_fingerprints


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


def _fingerprint(concept_dir, cache_dir=None, **overrides):
    """``cache_dir=None`` is "keep no digest record", so every media file is hashed
    on every call. That is the right default for the tests below that ask *what*
    the fingerprint sees; the tests that ask *how often it reads* pass a real
    directory."""
    return dataset_fingerprints([_concept(concept_dir, **overrides)], cache_dir)


def _record(cache_dir):
    """The record as it sits on disk, unparsed by the module under test."""
    with open(os.path.join(cache_dir, _RECORD_NAME), encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def _watching_opens():
    """Every path handed to ``open`` while the block runs.

    A record hit has to be shown to perform no *read*, and the only honest way to
    say that is to count opens. Timing would measure the page cache.
    """
    opened = []
    real_open = builtins.open

    def counted(file, *args, **kwargs):
        if not isinstance(file, int):
            opened.append(os.path.abspath(os.fspath(file)))
        return real_open(file, *args, **kwargs)

    builtins.open = counted
    try:
        yield opened
    finally:
        builtins.open = real_open


def test_a_dataset_with_no_rows_fingerprints_the_same_however_it_got_there():
    # No concepts, and concepts that contribute nothing, hold the same tensors --
    # so they must not be handed separate cache directories.
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert dataset_fingerprints([], None) == _fingerprint(_dataset(tmp_dir), enabled=False)


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


def test_a_same_size_replacement_moves_the_media_fingerprint_with_the_mtime_frozen():
    """Media are identified by their *bytes*, so neither half of the old
    ``(size, mtime)`` proxy can hide a replacement. Both halves are pinned here on
    purpose: the replacement keeps the byte count, and the timestamp is set back to
    what it was -- which is what ``touch -r``, a restore tool, or a filesystem with
    coarse mtime granularity does for free.

    This is the un-recorded call. With a record in play the same edit is the one
    case the record cannot see; see
    ``test_a_record_hit_reads_nothing_and_writes_nothing``."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        image = os.path.join(concept_dir, "a.png")
        frozen = 1_000_000_000_000_000_000
        os.utime(image, ns=(frozen, frozen))
        before = _fingerprint(concept_dir)

        _write(image, b"z" * os.path.getsize(image))  # same size, different bytes
        os.utime(image, ns=(frozen, frozen))
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
        assert disabled == dataset_fingerprints([], None)
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
        assert missing != dataset_fingerprints([], None)


def test_concept_order_is_significant():
    # Rows are cached positionally per concept, so the concept list is ordered
    # data, not a set: swapping two concepts renumbers every row after the first.
    with tempfile.TemporaryDirectory() as tmp_dir:
        first = _dataset(tmp_dir, images=("a.png",))
        second = os.path.join(tmp_dir, "second")
        _write(os.path.join(second, "b.png"), b"other")
        forward = dataset_fingerprints([_concept(first), _concept(second)], None)
        reverse = dataset_fingerprints([_concept(second), _concept(first)], None)
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


# --- the digest record -----------------------------------------------------
#
# Media are hashed, but a hash of every file on every launch is ~170x a stat, so
# each digest is recorded under cache_dir keyed on (size, mtime_ns) and re-read
# only when that key moves. These tests are about the record: that it makes the
# right call cheap, that it never makes a wrong call cheap, and that it cannot
# take a run down with it.


def test_a_bumped_mtime_over_identical_bytes_does_not_move_the_media_fingerprint():
    """The point of the whole record.

    ``git checkout`` of a dataset repo, an rsync without ``--times``, a restore
    from a backup: every one of them rewrites timestamps over bytes that did not
    change. While mtime was part of the dataset's identity that meant a fresh salt
    and a full re-encode of an unchanged dataset. It must now be invisible -- both
    with a record (the bumped stat forces a re-read, which finds the same bytes)
    and without one (nothing ever looked at the timestamp)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        concept_dir = _dataset(tmp_dir, images=("a.png", "b.png"))
        recorded_before = _fingerprint(concept_dir, cache_dir=cache_dir)
        plain_before = _fingerprint(concept_dir)

        for name in os.listdir(concept_dir):
            os.utime(os.path.join(concept_dir, name), ns=(2**60, 2**60))

        assert _fingerprint(concept_dir, cache_dir=cache_dir)[0] == recorded_before[0], \
            "a timestamp is not content: the salt must not move"
        assert _fingerprint(concept_dir)[0] == plain_before[0]


def test_a_record_hit_reads_nothing_and_writes_nothing():
    """Steady state is a stat per media file and nothing else.

    Asserted against a counting ``open``, not against a stopwatch: a warm page
    cache makes a real read fast enough to pass a timing test. The caption is the
    control -- captions carry no record by design, so one *must* appear in the same
    list, or the counter is not watching the right thing.

    The record file itself must also not be rewritten when nothing changed. A
    launch that writes nothing cannot lose a race with a concurrent one."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        image = os.path.abspath(os.path.join(concept_dir, "a.png"))
        caption = os.path.abspath(os.path.join(concept_dir, "a.txt"))

        cold = _fingerprint(concept_dir, cache_dir=cache_dir)
        written_at = os.stat(os.path.join(cache_dir, _RECORD_NAME)).st_mtime_ns

        with _watching_opens() as opened:
            warm = _fingerprint(concept_dir, cache_dir=cache_dir)

        assert warm == cold, "the recorded digest must be the digest of the bytes"
        assert image not in opened, "a record hit must not open the file it vouches for"
        assert caption in opened, "captions have no record and are read every launch"
        assert os.stat(os.path.join(cache_dir, _RECORD_NAME)).st_mtime_ns == written_at, \
            "an unchanged dataset must not rewrite the record"


def test_changed_bytes_move_the_media_fingerprint_through_the_record():
    """The record may only ever save a read, never an invalidation. An ordinary
    edit moves the stat key, so the file is re-hashed and the new digest reaches
    the salt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        image = os.path.abspath(os.path.join(concept_dir, "a.png"))
        before = _fingerprint(concept_dir, cache_dir=cache_dir)
        recorded_before = _record(cache_dir)["files"][image]

        _write(os.path.join(concept_dir, "a.png"), b"entirely different pixels")
        after = _fingerprint(concept_dir, cache_dir=cache_dir)

        assert after[0] != before[0]
        assert _record(cache_dir)["files"][image][2] != recorded_before[2], \
            "the new digest must be written back, or every launch re-hashes it"


def test_a_size_change_with_the_mtime_preserved_is_re_hashed():
    """The record keys on *both* halves of the stat, so a size change is caught even
    when the timestamp is put back -- which a copy tool that preserves mtime does
    on every file it writes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        image = os.path.join(concept_dir, "a.png")
        frozen = 1_000_000_000_000_000_000
        os.utime(image, ns=(frozen, frozen))
        before = _fingerprint(concept_dir, cache_dir=cache_dir)

        _write(image, b"x" * (os.path.getsize(image) + 1))  # one byte longer
        os.utime(image, ns=(frozen, frozen))

        assert _fingerprint(concept_dir, cache_dir=cache_dir)[0] != before[0]


def test_an_unusable_record_is_rebuilt_instead_of_raising():
    """Missing, truncated mid-write, not JSON at all, or written by a version that
    does not exist yet: every one of them has to mean "hash everything once", never
    a run that cannot start. A record is an optimisation, and an optimisation may
    not be load-bearing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = _dataset(tmp_dir, images=("a.png",))
        expected = _fingerprint(concept_dir)

        corruptions = [
            None,                                              # never written
            "",                                                # created, never filled
            '{"version": 1, "files": {"/a.png": [1, 2, "de',   # torn mid-write
            "not json at all",
            '{"version": 99, "files": {}}',                    # a shape from the future
            '{"version": 1, "files": "not a mapping"}',
            '{"version": 1, "files": {"/a.png": ["nonsense"]}}',
        ]
        for i, content in enumerate(corruptions):
            cache_dir = os.path.join(tmp_dir, f"cache-{i}")
            os.makedirs(cache_dir)
            if content is not None:
                _write(os.path.join(cache_dir, _RECORD_NAME), content)

            assert _fingerprint(concept_dir, cache_dir=cache_dir) == expected, content
            # ... and the damaged record is replaced by a good one, so the cost is
            # paid once rather than on every launch from here on.
            assert _record(cache_dir)["version"] == 1
            assert _fingerprint(concept_dir, cache_dir=cache_dir) == expected


def test_two_writers_never_leave_a_torn_record():
    """Two runs can share one cache_dir -- a training run and a sample window, two
    configs launched back to back. The record is written to a temp file in the same
    directory and moved into place with os.replace, so a reader sees the old file or
    the new one and never a half-written one. The loser of a race loses only its own
    entries, and re-establishes them next launch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        failures = []

        def churn(tag):
            concept_dir = _dataset(os.path.join(tmp_dir, tag), images=(f"{tag}.png",))
            try:
                for i in range(25):
                    # a fresh size every round, so every pass misses the record and
                    # writes a new one
                    _write(os.path.join(concept_dir, f"{tag}.png"), bytes([i]) * (i + 1))
                    _fingerprint(concept_dir, cache_dir=cache_dir)
                    _record(cache_dir)  # raises if it ever catches a torn file
            except Exception as error:  # noqa: BLE001 -- re-raised on the main thread
                failures.append(error)

        threads = [threading.Thread(target=churn, args=(tag,)) for tag in ("one", "two")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not failures, failures
        assert os.listdir(cache_dir) == [_RECORD_NAME], "no temp file may be left behind"


def test_walking_one_concept_does_not_evict_anothers_entries():
    """Entries have to be pruned or the record grows forever, but only against the
    files walked *for the concepts walked this run*. Prune globally and two configs
    that alternate -- a small smoke-test concept and the real dataset -- would evict
    each other every launch and re-hash every launch, which is the whole cost this
    record exists to remove."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        first = _dataset(os.path.join(tmp_dir, "first"), images=("a.png", "gone.png"))
        second = _dataset(os.path.join(tmp_dir, "second"), images=("b.png",))
        kept = os.path.abspath(os.path.join(second, "b.png"))

        _fingerprint(first, cache_dir=cache_dir)
        _fingerprint(second, cache_dir=cache_dir)

        # A run that walks only the first concept ...
        os.remove(os.path.join(first, "gone.png"))
        _fingerprint(first, cache_dir=cache_dir)
        files = _record(cache_dir)["files"]

        # ... prunes what it walked and found gone ...
        assert os.path.abspath(os.path.join(first, "gone.png")) not in files
        # ... and leaves the concept it never looked at alone.
        assert kept in files
        with _watching_opens() as opened:
            _fingerprint(second, cache_dir=cache_dir)
        assert kept not in opened, "the surviving entry must still be a record hit"
