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

Why it is stat-based
--------------------
Hashing every image on every launch would cost more than the re-encode the cache
exists to avoid, so media files are identified by ``(relative path, size,
mtime)`` -- the standard cheap proxy, and the same trade ``cache_key`` already
makes for model checkpoints. Captions are different: they are tiny, so hashing
them is free, and a same-length reword ("cat" -> "dog") is an entirely ordinary
edit that a size check cannot see. Their bytes are hashed outright.

The stat proxy assumes a materializer that preserves mtime for unchanged content
(a hardlink, or a copy that carries metadata). One that rewrites every file with a
fresh timestamp -- ``git checkout`` of a dataset repo, an rsync without
``--times`` -- costs a spurious re-cache. Conservative, never wrong. A filesystem
with coarse mtime granularity degrades the same way round: two edits inside one
tick are only distinguished if the size moved, which is why captions are hashed
rather than stat'd.

Why media and captions are separated
------------------------------------
A caption edit changes no VAE latent. Folding captions into the image salt would
push an entire dataset back through the VAE because one word changed, so both
fingerprints come from a single walk and are consumed by different salts.

Known gap: a prompt that comes from ``concept.text.prompt_path`` points outside
the concept directory and is not walked. Such a prompt file is per-concept, so
mgds' group key covers a change of *path* but not of contents.
"""

import hashlib
import os
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
_UNREADABLE = "?"

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


def dataset_fingerprints(concepts: list[ConceptConfig]) -> tuple[str, str]:
    """``(media_fingerprint, caption_fingerprint)`` over every enabled concept.

    - **media** moves when an image/video is added, removed, replaced or resized.
      It is what the *image* (VAE latent) cache must key on.
    - **caption** moves when a caption's text changes, and also when the media file
      list changes -- embeddings are cached positionally per row, so which rows
      exist is part of the identity even though their pixels are not.

    A disabled concept is skipped entirely: it produces no rows, so a cache built
    with it disabled is interchangeable with one built without it. Enabling it
    still moves both fingerprints, because its files then enter the walk.

    One walk serves both fingerprints. The walk is the expensive part -- a stat per
    media file, a read per caption -- and the two salts are computed together, so
    the caller is expected to call this once and hand each half to the salt that
    consumes it. There is deliberately no process-level memo: OneTrainer's UI is
    long-lived and can start a second run after the dataset was edited, which is
    exactly the case a memo would answer wrongly.
    """
    media_leaves: list[_Node] = []
    caption_leaves: list[_Node] = []

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

        media: list[_Node] = []
        captions: list[_Node] = []
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

    return _digest(media_leaves), _digest(caption_leaves)
