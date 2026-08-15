"""Content identity of the *dataset* behind a latent / text-embedding cache.

``modules/util/cache_key.py`` salts the cache directories by the *global* inputs a
cached tensor depends on — the VAE, the resolution / bucketing, the text encoders.
This module supplies the one dimension it was missing: the dataset itself.

Why it is needed
----------------
MGDS keys a cached sample by its per-concept *group key* (``concept.path``,
``concept.seed``, ``concept.include_subdirectories``, the ``image``/``text``
sub-config) and stores each group as a **positional list**. None of that carries the
concept's *contents*. So a cache built when a concept held one set of files is reused
verbatim after that set changed, serving row *i*'s tensors under row *i*'s new
identity. MGDS's own guard catches the case where the row *count* moved (a length
mismatch rebuilds instead of ``IndexError``-ing); a same-count change — an image
swapped, a caption reworded — is invisible to it and can only be caught here, by
moving the salt so the run lands in a fresh directory.

Do not be tempted to rely on the concept *path* changing instead. A staging layer may
deliberately reuse one path across re-exports to get incremental blob sync; that is a
property of the caller, not something this cache may assume.

Why it is stat-based
--------------------
Hashing every image on every launch would cost more than the re-encode the cache
exists to avoid, so media files are identified by ``(relative path, size, mtime)`` —
the standard cheap proxy, and the same trade cache_key.py already makes for
checkpoints. Captions are different: they are tiny, so hashing them is free, and a
same-length reword ("cat" → "dog") is an entirely ordinary edit that a size check
cannot see. Their text is hashed outright.

The stat proxy assumes a materializer that preserves mtime for unchanged content
(hardlinking, or a copy that carries metadata). One that rewrites every file with a
fresh timestamp costs a spurious re-cache — conservative, never wrong.

Why media and captions are separated
------------------------------------
A caption edit changes no VAE latent. Folding captions into the image salt would push
an entire dataset back through the VAE because one word changed, so both fingerprints
come from a single walk and are consumed by different salts.

Dependency-light on purpose (stdlib + ``path_util``) so the pure cache salt and its
isolated unit test can import it without pulling in the training stack.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from modules.util import path_util

# The bytes a cached latent is encoded from.
_MEDIA_EXTENSIONS = path_util.supported_image_extensions() | path_util.supported_video_extensions()
# The text a cached embedding is encoded from.
_CAPTION_EXTENSIONS = path_util.supported_caption_extensions()

# Stand-in for a concept whose directory cannot be listed (a missing path, a
# permissions error, a Huggingface dataset name that is not a local dir). Recorded
# rather than skipped so "unreadable" and "empty" stay distinguishable — the two mean
# very different things, and silently collapsing them would let a path typo reuse
# another concept's cache.
_UNREADABLE = "?"


def _digest(payload: Any) -> str:
    """Stable short hex digest of a nested list/tuple/str/int structure."""
    hasher = hashlib.sha256()

    def feed(node: Any) -> None:
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


def _caption_digest(path: str) -> str:
    """Hash a caption file's bytes, or ``_UNREADABLE`` if it cannot be read."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                hasher.update(chunk)
    except OSError:
        return _UNREADABLE
    return hasher.hexdigest()[:16]


def _media_stat(path: str) -> tuple[int, int] | str:
    """``(size, mtime_ns)`` for a media file, or ``_UNREADABLE``."""
    try:
        stat = os.stat(path)
    except OSError:
        return _UNREADABLE
    return (stat.st_size, stat.st_mtime_ns)


def _iter_files(root: str, include_subdirectories: bool) -> list[tuple[str, str]] | None:
    """``(relative posix path, absolute path)`` for every file under ``root``, sorted.

    ``None`` when ``root`` cannot be listed. Sorting is what makes the fingerprint
    independent of filesystem enumeration order, so the same dataset on two machines
    digests identically.
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


def _concept_identity(concept: Any) -> tuple[str, bool, bool]:
    """The concept fields that decide *which* files it contributes."""
    return (
        str(getattr(concept, "path", "") or ""),
        bool(getattr(concept, "enabled", True)),
        bool(getattr(concept, "include_subdirectories", False)),
    )


# One walk serves both salts. ``image_cache_salt`` and ``text_cache_salt`` are
# computed back to back from the same concepts, and the walk is the expensive part
# (a stat per media file, a read per caption), so the result is memoized for the
# process on the concept headers. Scoped to a launch by construction: a dataset
# cannot change between the two salt calls, and the salts are computed once, while
# the dataloader is being built.
_MEMO: dict[tuple, tuple[str | None, str | None]] = {}


def reset_memo() -> None:
    """Drop the memo. Only tests need this — they edit a dataset in place and
    re-fingerprint it within one process, which is the one thing the memo assumes
    never happens."""
    _MEMO.clear()


def dataset_fingerprints(concepts: Any) -> tuple[str | None, str | None]:
    """``(media_fingerprint, caption_fingerprint)`` over every enabled concept.

    Both are ``None`` when there is nothing to fingerprint (no concepts at all), so a
    caller with no dataset information produces a salt byte-identical to one built
    before this module existed — no spurious re-cache for a config that cannot benefit.

    - **media** moves when an image/video is added, removed, replaced, or resized. It
      is what the *image* (VAE latent) cache must key on.
    - **caption** moves when a caption's text changes, and also when the media file
      list changes — captions are cached positionally per row, so which rows exist is
      part of the identity even though their pixels are not.

    A disabled concept contributes only its name and disabled flag: its files are not
    walked (they produce no rows) but re-enabling it must still move the salt.
    """
    if not concepts:
        return None, None

    headers = tuple(_concept_identity(concept) for concept in concepts)
    memoized = _MEMO.get(headers)
    if memoized is not None:
        return memoized

    media_leaves: list[Any] = []
    caption_leaves: list[Any] = []

    for header in headers:
        path, enabled, include_subdirectories = header

        if not enabled:
            media_leaves.append(header)
            caption_leaves.append(header)
            continue

        files = _iter_files(path, include_subdirectories)
        if files is None:
            media_leaves.append((header, _UNREADABLE))
            caption_leaves.append((header, _UNREADABLE))
            continue

        media: list[Any] = []
        captions: list[Any] = []
        for rel, full in files:
            extension = os.path.splitext(rel)[1].lower()
            if extension in _MEDIA_EXTENSIONS:
                media.append((rel, _media_stat(full)))
                # the row exists for the text cache too, but its pixels are irrelevant
                captions.append(rel)
            elif extension in _CAPTION_EXTENSIONS:
                captions.append((rel, _caption_digest(full)))

        media_leaves.append((header, media))
        caption_leaves.append((header, captions))

    fingerprints = (_digest(media_leaves), _digest(caption_leaves))
    _MEMO[headers] = fingerprints
    return fingerprints
