"""Content identity of the *dataset* behind a latent / text-embedding cache.

``modules/util/cache_key.py`` salts the cache directories by everything a cached
tensor depends on that the DiskCache group key does not carry. This module
supplies one of those dimensions: the dataset itself.

Why it is needed
----------------
mgds keys a cached sample by its per-concept *group key* (``concept.path``,
``concept.seed``, ``concept.include_subdirectories`` and the ``image``/``text``
sub-config) and stores each group as a **positional list**. None of that carries
the concept's *contents*. So a cache built when a concept held one set of files is
reused verbatim after that set changed, serving row *i*'s tensors under row *i*'s
new identity. mgds' own guard catches the case where the row *count* moved (a
length mismatch rebuilds instead of ``IndexError``-ing); a same-count change -- an
image swapped, a caption reworded -- is invisible to it and can only be caught
here, by moving the salt so the run lands in a fresh directory.

Do not be tempted to rely on the concept *path* changing instead. A tool that
prepares datasets may deliberately reuse one path across re-exports (to get
incremental sync); that is a property of the caller, not something this cache may
assume.

Why media are hashed, but only once
-----------------------------------
A media file is identified by ``(relative path, digest of its bytes)``. Size and
mtime are deliberately **not** part of that identity. They are the key of a
*record* -- ``<cache_dir>/.dataset_fingerprints.json`` -- whose only job is to let
a digest be reused without reading the file again. A launch stats every media
file and reads only the ones whose ``(size, mtime_ns)`` no longer matches what the
record vouches for.

It is built this way round because of where the cost actually is. Per file, a hash
is cheap against the thing the cache exists to avoid: sha256 over a 2 MB jpeg is
~0.35 ms, while merely *decoding* it with PIL is ~5 ms -- 14x, before any resize
and before the VAE runs at all. What is expensive is hashing *every* file on
*every* launch, because the floor there is touching the bytes: even an infinitely
fast digest is ~170x a ``stat``, which reads an inode already in RAM, and no
cheaper algorithm closes that gap (crc32 is 2.4x sha256 and is a 32-bit checksum;
sha1, blake2 and md5 are all slower than sha256 on any box with SHA-NI). So the
trade is not "hash or stat" but "hash the files that moved, stat the rest".

The two ways this can be wrong are **not** symmetric, and must not be described as
if they were:

- A tool that rewrites timestamps over identical bytes -- ``git checkout`` of a
  dataset repo, an rsync without ``--times``, a restore from a backup that drops
  metadata -- misses the record and pays **one read per file**. It cannot produce
  a wrong answer: identical bytes digest identically, so the dataset identity does
  not move and nothing is re-encoded. Conservative, and merely slow once.
- A content change that leaves **both** size and mtime alone -- ``touch -r``, an
  in-place same-size edit inside one filesystem tick, a coarse-granularity mount
  such as exFAT or some SMB shares -- **hits** the record, reuses a digest that is
  no longer true, and serves a stale cache. That is unsound, not wasteful, and
  this module does not fix it. Closing it means hashing every file on every
  launch, which is the cost ruled out above. The hole is narrow -- an ordinary
  write moves mtime -- but it is real, and it is the one thing a reader of this
  module should know it is not protected from.

The record lives at the *root* of ``cache_dir``, never inside the salted
subdirectory: it is an input to computing that salt, so storing it under the
directory the salt names would be circular. It also survives
``clear_cache_before_training``, which removes only ``image``, ``text``, ``vae``
and ``epoch-*`` -- the record describes the *dataset*, not the cache, so clearing
the cache should cost a re-encode but not a re-hash.

Captions carry no record at all. They are tiny, so a record entry would cost more
than the read it saves, and a same-length reword ("cat" -> "dog") is an entirely
ordinary edit that only their bytes reveal. Their bytes are hashed every launch.

Why media and captions are separated
------------------------------------
A caption edit changes no VAE latent. Folding captions into the image salt would
push an entire dataset back through the VAE because one word changed, so both
fingerprints come from a single walk and are consumed by different salts.

Known gap: a prompt that comes from ``concept.text.prompt_path`` points outside
the concept directory and is not walked. Such a prompt file is per-concept, so
mgds' group key covers a change of *path* but not of contents.
"""

import contextlib
import hashlib
import json
import os
import tempfile
from typing import TypeAlias

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig

# The bytes a cached latent is encoded from. Mask and conditioning images live in
# the same directory under a ``-masklabel`` / ``-condlabel`` postfix and carry
# ordinary image extensions, so they are covered here too -- and they must be:
# ``latent_mask`` and ``latent_conditioning_image`` are cached tensors as much as
# ``latent_image`` is.
_MEDIA_EXTENSIONS = path_util.supported_image_extensions() | path_util.supported_video_extensions()
# The text a cached embedding is encoded from.
_CAPTION_EXTENSIONS = path_util.supported_caption_extensions()

# Stand-in for a file that cannot be stat'd or read. It has to digest to
# *something*, and a constant is the conservative choice: two files that both fail
# read as the same, which at worst costs a spurious re-cache once they can be read.
# It is not hex, so it can never collide with a real digest.
_UNREADABLE = "?"

# The digest record. Its name starts with a dot so it does not look like one of the
# cache's own subdirectories to a human reading ``cache_dir``, and the version is
# checked on load so a future shape can be introduced by bumping it rather than by
# teaching every old checkout to parse it -- an unrecognised version is simply
# re-hashed.
_RECORD_NAME = ".dataset_fingerprints.json"
_RECORD_VERSION = 1

# ``{absolute path: (size, mtime_ns, digest)}``. Absolute, because one ``cache_dir``
# serves every concept of every run in that workspace and two concepts can hold the
# same relative name.
_Record: TypeAlias = dict[str, tuple[int, int, str]]

# The leaf/branch shape fed to _digest: a tree of strings, ints and sequences.
# Spelled as a string so the recursion resolves without needing 3.12's `type`
# statement (the project supports 3.10).
_Node: TypeAlias = "str | int | tuple[_Node, ...] | list[_Node]"


def _digest(payload: _Node) -> str:
    """Stable short hex digest of a nested list/tuple/str/int structure."""
    hasher = hashlib.sha256()

    def feed(node: _Node) -> None:
        if isinstance(node, (list, tuple)):
            hasher.update(b"(")
            for child in node:
                feed(child)
                hasher.update(b",")
            hasher.update(b")")
        else:
            hasher.update(str(node).encode("utf-8", "surrogatepass"))
        hasher.update(b"\x00")

    feed(payload)
    return hasher.hexdigest()[:16]


def _bytes_digest(path: str) -> str:
    """Hash a file's bytes, or ``_UNREADABLE`` if it cannot be read.

    Truncated to the same 16 hex digits as everything else here. It is a leaf of a
    tree that is itself hashed, so the width only has to make an accidental
    collision within one dataset implausible, and it keeps the record roughly a
    third of the size a full sha256 would.
    """
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                hasher.update(chunk)
    except OSError:
        return _UNREADABLE
    return hasher.hexdigest()[:16]


def _media_fingerprint(path: str, known: _Record, seen: _Record) -> str:
    """The content digest of one media file, reading it only when it may have moved.

    ``known`` is the record as the last launch left it and is never mutated;
    ``seen`` accumulates what this launch established, and is what gets written
    back. A file whose ``(size, mtime_ns)`` still matches the record is not opened
    at all.

    An unreadable file is not recorded. It has no digest to vouch for, and leaving
    it out means the next launch retries instead of pinning ``_UNREADABLE`` to a
    stat that will still match once the permissions are fixed.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return _UNREADABLE

    size, mtime_ns = stat.st_size, stat.st_mtime_ns
    entry = known.get(path)
    if entry is not None and entry[0] == size and entry[1] == mtime_ns:
        digest = entry[2]
    else:
        digest = _bytes_digest(path)
        if digest == _UNREADABLE:
            return digest

    seen[path] = (size, mtime_ns, digest)
    return digest


def _load_record(cache_dir: str | None) -> _Record:
    """Last launch's record, or empty if there is nothing trustworthy to read.

    Every failure mode -- absent, unreadable, truncated mid-write, not JSON, not
    the shape this version writes -- lands in the same place: an empty record, so
    the launch hashes everything once and then writes a good one. A record is an
    optimisation; it may never be the reason a run cannot start.
    """
    if not cache_dir:
        return {}

    try:
        with open(os.path.join(cache_dir, _RECORD_NAME), encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return {}

    if not isinstance(blob, dict) or blob.get("version") != _RECORD_VERSION:
        return {}
    files = blob.get("files")
    if not isinstance(files, dict):
        return {}

    record: _Record = {}
    for path, entry in files.items():
        # Per entry, not per file: one mangled line costs one re-hash, not the
        # whole record.
        if not isinstance(path, str) or not isinstance(entry, list) or len(entry) != 3:
            continue
        size, mtime_ns, digest = entry
        if isinstance(size, int) and isinstance(mtime_ns, int) and isinstance(digest, str):
            record[path] = (size, mtime_ns, digest)
    return record


def _save_record(cache_dir: str | None, record: _Record) -> None:
    """Write the record, atomically, and never fail the run over it.

    Two runs can share a ``cache_dir``, so the file is written to a temp name in
    that same directory and moved into place with ``os.replace``: a reader sees
    either the old record or the new one, never a half-written one, and the loser
    of a race merely loses its own entries until the next launch re-establishes
    them. The temp file is created in ``cache_dir`` rather than in the system temp
    directory because ``os.replace`` is only atomic within one filesystem.
    """
    if not cache_dir:
        return

    payload = {
        "version": _RECORD_VERSION,
        # Sorted so the file is diffable and so two runs over the same dataset
        # produce byte-identical records.
        "files": {path: list(entry) for path, entry in sorted(record.items())},
    }

    temp_path: str | None = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        handle, temp_path = tempfile.mkstemp(prefix=_RECORD_NAME, suffix=".tmp", dir=cache_dir)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(temp_path, os.path.join(cache_dir, _RECORD_NAME))
        temp_path = None
    except OSError:
        # A read-only or not-yet-creatable cache_dir costs a re-hash next launch and
        # nothing else. It must not stop a run that is otherwise ready to train.
        if temp_path is not None:
            with contextlib.suppress(OSError):
                os.remove(temp_path)


def _prune(known: _Record, seen: _Record, scopes: list[tuple[str, bool]]) -> _Record:
    """``seen``, plus every entry of ``known`` this run had no business judging.

    Entries have to be dropped somewhere or the record grows forever, but the set
    to prune against is *the files walked for the concepts walked this run* -- not
    the record as a whole. Pruning globally would make two configs that alternate
    (a small smoke-test concept and the real dataset, say) evict each other's
    entries every single launch, so each would re-hash on every run: precisely the
    cost this record exists to remove.

    ``scopes`` is one ``(absolute concept path, recursive)`` per concept walked. A
    non-recursive concept only claims the files directly in its directory, so a
    nested concept's entries survive a launch that walked only the parent.
    """
    kept = {
        path: entry
        for path, entry in known.items()
        if not any(
            path.startswith(root + os.sep) if recursive else os.path.dirname(path) == root
            for root, recursive in scopes
        )
    }
    kept.update(seen)
    return kept


def _iter_files(root: str, include_subdirectories: bool) -> list[tuple[str, str]] | None:
    """``(relative posix path, absolute path)`` for every file under ``root``, sorted.

    ``None`` when ``root`` cannot be listed at all. Sorting is what makes the fingerprint
    independent of filesystem enumeration order, so the same dataset on two
    machines digests identically.
    """
    found: list[tuple[str, str]] = []
    try:
        if include_subdirectories:
            for dir_path, dir_names, file_names in os.walk(root):
                dir_names.sort()  # in-place: fixes recursion order too
                for name in sorted(file_names):
                    full = os.path.join(dir_path, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    found.append((rel, full))
        else:
            with os.scandir(root) as entries:
                found.extend(
                    (entry.name, entry.path)
                    for entry in sorted(entries, key=lambda e: e.name)
                    if entry.is_file()
                )
    except OSError:
        return None
    return found


def _concept_header(concept: ConceptConfig) -> tuple[str, bool, bool]:
    """The concept fields that decide *which* files it contributes."""
    return (
        str(concept.path or ""),
        bool(concept.enabled),
        bool(concept.include_subdirectories),
    )


def dataset_fingerprints(concepts: list[ConceptConfig], cache_dir: str | None) -> tuple[str, str]:
    """``(media_fingerprint, caption_fingerprint)`` over every enabled concept.

    - **media** moves when an image/video is added, removed, or has its bytes
      changed. It is what the *image* (VAE latent) cache must key on.
    - **caption** moves when a caption's text changes, and also when the media file
      list changes -- embeddings are cached positionally per row, so which rows
      exist is part of the identity even though their pixels are not.

    ``cache_dir`` is where the digest record is kept, at its root, as
    ``.dataset_fingerprints.json``. It is required rather than defaulted because
    the record is the difference between stat'ing a dataset and reading it, and no
    caller should acquire that cost by forgetting an argument. Pass ``None`` (or
    ``""``, which is what ``TrainConfig.cache_dir`` holds when caching is
    configured off) to run **without** a record: every media file is then hashed on
    every call. That is the correct answer for a caller that has nowhere durable to
    write, and it is never *wrong* -- only slower.

    A disabled concept is skipped entirely: it produces no rows, so a cache built
    with it disabled is interchangeable with one built without it. Enabling it
    still moves both fingerprints, because its files then enter the walk. Its
    record entries are left alone, so disabling a concept for one run does not cost
    a re-hash when it comes back.

    One walk serves both fingerprints. The walk is the expensive part -- a stat per
    media file plus a read of the ones that moved, a read per caption -- and the two
    salts are computed together, so the caller is expected to call this once and
    hand each half to the salt that consumes it. There is deliberately no
    process-level memo: OneTrainer's UI is long-lived and can start a second run
    after the dataset was edited, which is exactly the case a memo would answer
    wrongly. The record is not a memo -- it is re-validated against the filesystem,
    file by file, on every call.
    """
    media_leaves: list[_Node] = []
    caption_leaves: list[_Node] = []

    known = _load_record(cache_dir)
    seen: _Record = {}
    scopes: list[tuple[str, bool]] = []

    for concept in concepts:
        header = _concept_header(concept)
        path, enabled, include_subdirectories = header

        # A disabled concept produces no rows -- mgds' CollectPaths skips it
        # outright -- so it contributes nothing here either. Not even its header:
        # a cache built with it disabled holds exactly what a cache built without
        # it at all would hold, and telling the two apart would only re-encode a
        # dataset for a difference no tensor can see. Walking it would be worse
        # still: editing a disabled concept would push the enabled ones back
        # through the VAE.
        if not enabled:
            continue

        # An unlistable path (a typo, a permissions error, a Huggingface dataset
        # name that is not a local dir) contributes no files. The path itself is in
        # the header, so this cannot be confused with a different concept's empty
        # directory.
        files = _iter_files(path, include_subdirectories) or []
        scopes.append((os.path.abspath(path), include_subdirectories))

        media: list[_Node] = []
        captions: list[_Node] = []
        for rel, full in files:
            extension = os.path.splitext(rel)[1].lower()
            if extension in _MEDIA_EXTENSIONS:
                # The record is keyed absolutely (two concepts can hold the same
                # relative name) while the leaf stays relative: moving a whole
                # dataset directory must not re-encode it, and the concept path is
                # already in the header.
                media.append((rel, _media_fingerprint(os.path.abspath(full), known, seen)))
                # the row exists for the text cache too, but its pixels are irrelevant
                captions.append(rel)
            elif extension in _CAPTION_EXTENSIONS:
                captions.append((rel, _bytes_digest(full)))

        media_leaves.append((header, media))
        caption_leaves.append((header, captions))

    record = _prune(known, seen, scopes)
    if record != known:
        # A steady-state launch changes nothing, and a launch that writes nothing
        # cannot lose a race with a concurrent one.
        _save_record(cache_dir, record)

    return _digest(media_leaves), _digest(caption_leaves)
