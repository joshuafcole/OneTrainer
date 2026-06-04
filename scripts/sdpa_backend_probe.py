#!/usr/bin/env python3
"""Force-test each SDPA backend at Anima's real attention shapes.

Why: setting torch.backends.cuda.enable_cudnn_sdp(True) only makes a backend
*eligible*; the dispatcher (and, under torch.compile, inductor's lowering) still
picks by its own priority + constraint checks and will silently fall back to the
sm80 cutlass mem-efficient kernel -- which is what OT_ATTN=cudnn did in training
(trace: 100% _efficient_attention, 0 cuDNN kernels). This probe HARD-SELECTS one
backend via sdpa_kernel([backend]), so an unsupported config raises with the real
reason instead of silently falling back, and times fwd+bwd + peak VRAM.

Run on the GPU box (the 5090), not the dev box:
    venv\\Scripts\\python.exe scripts\\sdpa_backend_probe.py

Override the shape grid if Anima's heads/head_dim differ from the guesses:
    ...sdpa_backend_probe.py --heads 24 --head-dim 128 --batch 1
"""
import argparse
import re
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

# seq lengths = latent_tokens straight out of the perf logs; HIGH-RES FIRST so the
# rows we actually care about land even if a later (math/OOM) cell wedges the run.
SEQ_LENS = [25344, 24576, 2560, 2240]

# cudnn/mem_eff first (the real candidates); math + flash last (math OOMs at high
# seq; flash is not compiled into the cu128 windows wheel).
BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}

# math materializes an seq*seq*heads score matrix -> skip it past this point.
MATH_MAX_SEQ = 4096

reasons = {}  # short token -> full first line, printed once at the end


def short_reason(e):
    msg = (str(e).splitlines() or ["?"])[0].strip()
    # collapse the common ones to a stable token
    low = msg.lower()
    if "no available kernel" in low or "no kernel" in low:
        tok = "no-kernel"
    elif "out of memory" in low:
        tok = "oom"
    elif "not compiled" in low:
        tok = "not-built"
    else:
        tok = msg[:12]
    reasons[tok] = msg[:120]
    return tok


def bench(backend, b, h, s, d, dtype, masked):
    dev = "cuda"
    iters = 20 if s < 5000 else 6
    q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    mask = None
    if masked:
        mask = torch.zeros(b, 1, 1, s, device=dev, dtype=dtype)
        mask[..., s // 2:] = float("-inf")
    grad = torch.randn(b, h, s, d, device=dev, dtype=dtype)

    def one():
        with sdpa_kernel([backend]):  # single backend => raises if unsupported
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out.backward(grad)

    try:
        for _ in range(3):  # warmup / capability check (raises -> caller catches)
            one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000.0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        return ms, peak_gb
    finally:
        del q, k, v, grad, mask
        torch.cuda.empty_cache()


# substrings that mark a real GPU *device* kernel (vs an aten:: dispatcher row)
_KERNEL_HINTS = ("cudnn_generated", "cutlass", "fmha", "wmma", "flash", "_sdpa", "gemm")


def _kernel_arch(name):
    """Pull the target arch out of a kernel name: 'sm80'/'sm120' or cutlass '_80_'."""
    low = name.lower()
    m = re.search(r"sm(\d{2,3})", low) or re.search(r"cutlass_(\d{2,3})", low)
    return f"sm{m.group(1)}" if m else "?"


def capture_kernels(fn, label):
    """Run fn under the CUDA profiler and report which arch the dispatched device
    kernels target. This is the decisive sm80(Ampere-fallback)-vs-sm120(native
    Blackwell) readout -- inferred from kernel *names*, not from latency."""
    try:
        for _ in range(3):  # warmup; also surfaces capability errors
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  {label:>34}: x:{short_reason(e)}")
        return
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

    archs = {}  # arch tag -> set of (truncated) kernel names
    for ev in prof.key_averages():
        nm = ev.key
        low = nm.lower()
        if low.startswith("aten::") or not any(h in low for h in _KERNEL_HINTS):
            continue
        archs.setdefault(_kernel_arch(nm), set()).add(nm[:74])

    if not archs:
        print(f"  {label:>34}: (no arch-tagged device kernels captured)")
        return
    print(f"  {label:>34}: [{','.join(sorted(archs))}]")
    for arch in sorted(archs):
        for nm in sorted(archs[arch]):
            print(f"      {arch:>6}  {nm}")


def kernel_identity(batch, dtype):
    """Dump the actual dispatched kernels for cuDNN/mem_eff attention and a
    representative bf16 projection GEMM (the 7000+ kernels that dominate a step)."""
    print("\n== kernel identity (arch the dispatched device kernels target) ==")
    print("#  sm80 = Ampere compat fallback; sm100/sm120 = native Blackwell\n")
    h, d, s = 24, 128, 24576  # heads=24 head_dim=128 -> model_dim 3072, like Anima's heavy block
    F = torch.nn.functional

    q = torch.randn(batch, h, s, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, h, s, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch, h, s, d, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(batch, h, s, d, device="cuda", dtype=dtype)

    def mk_attn(be):
        def run():
            with sdpa_kernel([be]):
                out = F.scaled_dot_product_attention(q, k, v)
            out.backward(g)
        return run

    capture_kernels(mk_attn(SDPBackend.CUDNN_ATTENTION), f"cuDNN attn h{h} d{d} s{s}")
    capture_kernels(mk_attn(SDPBackend.EFFICIENT_ATTENTION), f"mem_eff attn h{h} d{d} s{s}")
    del q, k, v, g
    torch.cuda.empty_cache()

    dim = h * d
    a = torch.randn(s, dim, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(dim, dim, device="cuda", dtype=dtype, requires_grad=True)

    def mk_gemm():
        (a @ w).sum().backward()

    capture_kernels(mk_gemm, f"bf16 GEMM {s}x{dim} @ {dim}x{dim}")
    del a, w
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=0, help="0 => sweep {16,24}")
    ap.add_argument("--head-dim", type=int, default=0, help="0 => sweep {128,64}")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--no-kernel-id", action="store_true",
                    help="skip the dispatched-kernel-arch identity dump at the end")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device -- run this on the GPU box"); return
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"# {name} sm{cap[0]}{cap[1]}  torch {torch.__version__}  dtype={args.dtype}")
    print(f"# cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}  "
          f"arch_list {torch.cuda.get_arch_list()}")
    print("# cells = fwd+bwd ms (peak GB).  x:<reason> = backend rejected the shape.\n")

    head_dims = [args.head_dim] if args.head_dim else [128, 64]
    heads_list = [args.heads] if args.heads else [16, 24]

    for d in head_dims:
        for h in heads_list:
            print(f"== heads={h} head_dim={d} (model_dim={h * d}) batch={args.batch} ==")
            print(f"{'seq':>7} {'kind':>5} " + " ".join(f"{b:>16}" for b in BACKENDS))
            for s in SEQ_LENS:
                for masked in (False, True):
                    cells = []
                    for label, be in BACKENDS.items():
                        if label == "math" and s > MATH_MAX_SEQ:
                            cells.append(f"{'x:oom-skip':>16}"); continue
                        try:
                            ms, gb = bench(be, args.batch, h, s, d, dtype, masked)
                            cells.append(f"{ms:>10.2f}({gb:4.1f})")
                        except Exception as e:
                            torch.cuda.empty_cache()
                            cells.append(f"{'x:' + short_reason(e):>16}")
                    tag = "mask" if masked else "self"
                    print(f"{s:>7} {tag:>5} " + " ".join(cells), flush=True)
            print()

    if not args.no_kernel_id:
        kernel_identity(args.batch, dtype)

    if reasons:
        print("\nrejection reasons:")
        for tok, full in reasons.items():
            print(f"  {tok:>10}: {full}")


if __name__ == "__main__":
    main()
