"""Lightweight, opt-in diagnostics for textual-inversion (embedding) training.

The goal is to answer two questions on a remote training box without a debugger:

  1. Does the trigger token actually reach the model? (i.e. is the placeholder's
     token id present in the batch the train step consumes — if it never appears,
     no gradient can reach the trained vector and the embedding stays frozen.)
  2. Is the trained vector receiving a gradient and actually moving? (grad norm +
     cumulative drift from its seed — distinguishes "no signal" from "signal but
     pinned by norm-preservation/optimizer".)

All output is gated: nothing prints unless ``OT_TI_DEBUG`` is set (truthy) or the
run's ``config.debug_mode`` is on. Logging is throttled so a long run doesn't
flood stdout. Output is a single greppable ``[TI-DEBUG]`` line per probe, flushed
so it interleaves correctly with the agent's captured subprocess stdout.
"""

from __future__ import annotations

import os

_PREFIX = "[TI-DEBUG]"
_TRUTHY = {"1", "true", "yes", "on"}


def ti_debug_enabled(config=None) -> bool:
    """True when the env flag is set or the run is in debug_mode."""
    if os.environ.get("OT_TI_DEBUG", "").strip().lower() in _TRUTHY:
        return True
    return bool(config is not None and getattr(config, "debug_mode", False))


def ti_should_log(step: int) -> bool:
    """Dense at the start (catch a cold pipeline immediately), then periodic.

    Cadence is tunable: ``OT_TI_DEBUG_EVERY=N`` logs every N steps (default 50);
    the first 5 steps always log so the very first batches are visible."""
    try:
        every = int(os.environ.get("OT_TI_DEBUG_EVERY", "50") or "50")
    except ValueError:
        every = 50
    return step < 5 or (every > 0 and step % every == 0)


def ti_log(msg: str) -> None:
    print(f"{_PREFIX} {msg}", flush=True)
