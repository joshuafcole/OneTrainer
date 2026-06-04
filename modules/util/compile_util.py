import os

import torch

from sympy import S


#code from https://github.com/pytorch/pytorch/blob/ed82d5fcfd80110565f69130f286c7bfec6db2dc/torch/utils/_sympy/functions.py#L481
#but accepts negative numbers, to avoid https://github.com/Nerogar/OneTrainer/issues/1126
#can be removed once https://github.com/pytorch/pytorch/pull/169726 is merged into a torch version we use
@classmethod
def Mod_patched_eval(cls, p, q):
    # This was adapted from: sympy/core/mod.py

    # Triggered by
    # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
    # assert p.is_integer, p
    # assert q.is_integer, q

    if q.is_zero:
        raise ZeroDivisionError("Modulo by zero")

    # Three cases:
    #   1. p == 0
    #   2. p is either q or -q
    #   3. p is integer and q == 1
    if p is S.Zero or p in (q, -q) or q == 1:
        return S.Zero

    # Evaluate if they are both literals.
    if q.is_Number and p.is_Number:
        if p < 0:
            #raise AssertionError(p)
            pass
        if q < 1:
            raise AssertionError(q)
        return p % q

    # If q == 2, it's a matter of whether p is odd or even.
    if q.is_Number and q == 2:
        if p.is_even:
            return S.Zero
        if p.is_odd:
            return S.One

    # If p is a multiple of q.
    r = p / q
    if r.is_integer:
        return S.Zero

    # If p < q and its ratio is positive, then:
    #   - floor(p / q) = 0
    #   - p % q = p - floor(p / q) * q = p
    less = p < q
    if less.is_Boolean and bool(less) and r.is_positive:
        return p

    return None


def init_compile():
    torch._dynamo.config.cache_size_limit = 8192
    torch.utils._sympy.functions.Mod.eval = Mod_patched_eval


def init_attention_backend():
    """Env-gated SDPA backend preference for A/B perf testing (perf-instrumentation branch).

    The default PyTorch SDPA selection lands on the cutlass *memory-efficient* path, whose
    kernels are sm80-tagged -- a compatibility fallback on newer (e.g. Blackwell/sm120) GPUs.
    This lets us force a faster backend for the (unmasked, expensive) self-attention while
    leaving mem-efficient enabled as a fallback for ops the preferred backend rejects (e.g.
    the masked cross-attention), so a run never hard-crashes from an unsupported mask.

        OT_ATTN=cudnn      -> prefer cuDNN attention (native on Blackwell), mem-eff fallback
        OT_ATTN=flash      -> prefer Flash attention, mem-eff fallback
        OT_ATTN=mem/unset  -> leave PyTorch defaults (baseline: sm80 cutlass mem-efficient)
    """
    mode = os.environ.get("OT_ATTN", "").strip().lower()

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm{cap[0]}{cap[1]}"
        else:
            name, arch = "cpu", "n/a"
    except Exception:
        name, arch = "unknown", "n/a"

    b = torch.backends.cuda
    if mode == "cudnn":
        b.enable_mem_efficient_sdp(True)   # fallback for masked cross-attn
        b.enable_flash_sdp(False)
        b.enable_cudnn_sdp(True)
        b.enable_math_sdp(True)
    elif mode == "flash":
        b.enable_mem_efficient_sdp(True)
        b.enable_flash_sdp(True)
        b.enable_cudnn_sdp(False)
        b.enable_math_sdp(True)
    # else (mem / unset): leave defaults untouched for a clean baseline

    try:
        enabled = (
            f"flash={b.flash_sdp_enabled()} mem_eff={b.mem_efficient_sdp_enabled()} "
            f"cudnn={b.cudnn_sdp_enabled()} math={b.math_sdp_enabled()}"
        )
    except Exception:
        enabled = "(could not query backend flags)"
    print(f"[attn] OT_ATTN={mode or 'unset'} on {name} {arch} | SDPA allowed: {enabled}")
