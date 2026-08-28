"""stdout is the result. Keep everything else off it.

These ``block_*`` scripts have a machine-readable contract: one JSON document on
stdout, consumed by the rehearsal agent, which parses it with pydantic. That
contract is only as good as this process's exclusive claim on fd 1 — and it does
not have one. The CUDA support libraries ``torch`` loads write banners straight
to fd 1 at load time, from C, so they arrive *before* the payload and neither
``print`` nor ``contextlib.redirect_stdout`` sees them::

    nvCOMP: low-level batched C API is deprecated ...{"granularity": ...}

which reaches the agent as ``Invalid JSON: expected ident at line 1 column 2``
and is reported as an out-of-date checkout, because from the agent's side an
unparseable payload looks exactly like an older script's payload.

So: **claim the channel before the noisy import**. ``claim()`` dups the real
stdout somewhere private and points fd 1 at stderr, which is descriptor-level
and therefore catches C libraries, subprocesses and stray ``print`` calls alike.
Anything written to stdout after that lands in the log where it belongs, and
``emit_json`` writes the payload to the descriptor nobody else can reach.

The alternative already in the tree is the marker frame the config-introspection
seam uses (``<<<OT_UNIVERSE_BEGIN>>>``), which works and is the right shape when
the payload is genuinely *embedded* in a log. It is the wrong shape here: it
would make every one of these scripts unusable with ``| jq`` and would break
every agent older than the markers, whereas a clean stdout is what every reader
— old agent, new agent, human — already expects.

Usage, and the order matters::

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import result_channel  # noqa: E402

    result_channel.claim()

    import torch  # noqa: E402
    ...
    result_channel.emit_json(payload)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# The dup of the original fd 1, or ``None`` when the channel was never claimed
# (or could not be). ``None`` is a supported state, not a failure: `emit_json`
# then writes to stdout the ordinary way, which is exactly what these scripts
# did before this module existed.
_result_fd: int | None = None


def claim() -> None:
    """Take fd 1 for the payload and give the noise to stderr.

    Idempotent, and best-effort: a process whose stdout cannot be duplicated
    (closed, or a platform without ``dup2``) keeps the un-claimed behaviour
    rather than failing a measurement over its logging.
    """
    global _result_fd
    if _result_fd is not None:
        return
    try:
        sys.stdout.flush()
        _result_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        _result_fd = None


def emit_json(payload: Any) -> None:
    """Write the payload — and only the payload — to the real stdout.

    ``os.write`` on the saved descriptor rather than through ``sys.stdout``,
    which after :func:`claim` points at stderr. The loop is not decoration: a
    write to a pipe is allowed to be partial, and a Gram over a large selection
    is comfortably past the 64 KiB a pipe will take in one go.
    """
    data = (json.dumps(payload, indent=None) + "\n").encode("utf-8")
    if _result_fd is None:
        sys.stdout.write(data.decode("utf-8"))
        sys.stdout.flush()
        return
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.write(_result_fd, view[written:])
