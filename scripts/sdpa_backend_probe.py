#!/usr/bin/env python3
"""Force-test each SDPA backend at Anima's real attention shapes.

Why: setting torch.backends.cuda.enable_cudnn_sdp(True) only makes a backend
*eligible*; the dispatcher still picks by its own priority + constraint checks and
will silently fall back to mem-efficient (cutlass sm80) -- which is exactly what an
OT_ATTN=cudnn training run did (trace showed 100% _efficient_attention, 0 cuDNN
kernels). This probe instead HARD-SELECTS one backend via sdpa_kernel(), so an
unsupported config raises with the real reason instead of silently falling back,
and times fwd+bwd for the ones that work.

Run this on the GPU box (the 5090), not the dev box:
    venv\\Scripts\\python.exe scripts\\sdpa_backend_probe.py

Override the shape grid if Anima's heads/head_dim differ from the guesses:
    ...sdpa_backend_probe.py --heads 24 --head-dim 128 --batch 1
"""
import argparse
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# seq lengths = latent_tokens straight out of the perf logs (low-res + high-res buckets)
SEQ_LENS = [2240, 2560, 24576, 25344]

BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def bench(backend, b, h, s, d, dtype, masked, iters=10):
    dev = "cuda"
    q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    mask = None
    if masked:
        # a padding-style additive mask like Cosmos cross-attn uses
        mask = torch.zeros(b, 1, 1, s, device=dev, dtype=dtype)
        mask[..., s // 2:] = float("-inf")
    grad = torch.randn(b, h, s, d, device=dev, dtype=dtype)

    def one():
        with sdpa_kernel([backend]):  # single backend => raises if unsupported
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out.backward(grad)

    # warmup / compile / capability check (may raise -> caller reports "unsupported")
    one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/iter (fwd+bwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=0, help="0 => sweep {16,24,32}")
    ap.add_argument("--head-dim", type=int, default=0, help="0 => sweep {64,128}")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device -- run this on the GPU box"); return
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"# {name} sm{cap[0]}{cap[1]}  torch {torch.__version__}  dtype={args.dtype}\n")

    head_dims = [args.head_dim] if args.head_dim else [64, 128]
    heads_list = [args.heads] if args.heads else [16, 24, 32]

    for d in head_dims:
        for h in heads_list:
            print(f"== heads={h} head_dim={d} (model_dim={h * d}) batch={args.batch} ==")
            print(f"{'seq':>7} {'mask':>5} " + " ".join(f"{b:>10}" for b in BACKENDS))
            for s in SEQ_LENS:
                for masked in (False, True):  # self-attn unmasked, cross-attn masked
                    cells = []
                    for label, be in BACKENDS.items():
                        try:
                            ms = bench(be, args.batch, h, s, d, dtype, masked)
                            cells.append(f"{ms:>9.1f}m")
                        except Exception as e:
                            msg = str(e).splitlines()[0][:9] if str(e) else "reject"
                            cells.append(f"{'x:' + msg:>10}")
                    tag = "mask" if masked else "self"
                    print(f"{s:>7} {tag:>5} " + " ".join(cells))
            print()


if __name__ == "__main__":
    main()
