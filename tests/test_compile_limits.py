"""Both dynamo recompile caps must be raised, not just the per-frame one.

The plan doc for this slice said no test was natural here, because the
behaviour is torch-version-dependent global config. That is true of torch's
side of it and not of ours: what this pins is our own invariant -- that
``init_compile()`` raises *both* caps -- which is exactly the thing that was
wrong (only the per-frame cap was set, leaving the accumulated one at its
default of 256).

It also pins the alias, which is the claim this slice was blocked on. If a
future torch really does drop ``cache_size_limit``, this says so directly
instead of a run silently training under a cap of 8.
"""

import torch

from modules.util.compile_util import init_compile


def _snapshot():
    c = torch._dynamo.config
    return c.recompile_limit, c.accumulated_recompile_limit


def _restore(state):
    c = torch._dynamo.config
    c.recompile_limit, c.accumulated_recompile_limit = state


def test_init_compile_raises_both_recompile_caps():
    state = _snapshot()
    try:
        torch._dynamo.config.recompile_limit = 8
        torch._dynamo.config.accumulated_recompile_limit = 256

        init_compile()

        assert torch._dynamo.config.recompile_limit > 8, "per-frame cap not raised"
        assert torch._dynamo.config.accumulated_recompile_limit > 256, (
            "accumulated cap left at its default -- a bucketed run stops at 256 "
            "total recompiles even though every frame is under its own limit"
        )
    finally:
        _restore(state)


def test_cache_size_limit_is_still_an_alias_of_recompile_limit():
    # Upstream's comment at compile_util.py:80 asserts this; our fork's commit
    # message asserted the opposite for a cu130 nightly, and the PR was
    # unreviewable until one of them was pinned. Measured true on
    # 2.9.1+rocm6.3 and 2.12.0+rocm7.2.
    state = _snapshot()
    try:
        torch._dynamo.config.recompile_limit = 4242
        assert torch._dynamo.config.cache_size_limit == 4242, (
            f"cache_size_limit is no longer an alias on torch {torch.__version__} -- "
            "setting only it would be a silent no-op"
        )
    finally:
        _restore(state)
