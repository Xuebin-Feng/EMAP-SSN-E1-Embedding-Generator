# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmark of the global (block-causal, cross-sequence) attention used by E1,
on a real RA input shape (median-token query), plus a GEMM roofline for this GPU.

Compares, for one global layer (H=20, D=64, bf16):
  flex_e1    : E1's path = create_block_mask (per forward) + compiled flex_attention
  flex_only  : the compiled flex_attention call alone (mask prebuilt)
  flash_loop : flash-attn, one call per sequence i: q_i attends to k[:end_i] (exact)
  flash_paged: flash_attn_with_kvcache, sequences as batch, shared paged K/V,
               cache_seqlens = end_i (exact, one launch)
Errors are measured against an fp32 reference of the same block-causal attention.
Nothing in E1/ is modified.
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
from flash_attn import flash_attn_func, flash_attn_with_kvcache  # noqa: E402

import E1.model.flex_attention as e1flex  # noqa: E402

torch.manual_seed(42)
dev = "cuda"
H, D = 20, 64

# Real sequence lengths of the median-token query
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    hmap = json.load(fh)["homolog_map"]
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402

fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
seq_by_h = dict(zip(fh_, fs_))
tok = {q: sum(len(seq_by_h[x]) + 4 for x in hmap[q] + [q]) for q in hmap}
med_q = sorted(tok, key=tok.get)[len(tok) // 2]
lens = torch.tensor([len(seq_by_h[x]) + 4 for x in hmap[med_q] + [med_q]])
N, S = int(lens.sum()), len(lens)
ends = torch.cumsum(lens, 0)
starts = ends - lens
print(f"median query: {S} sequences, {N} tokens")

q = torch.randn(1, N, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, N, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, N, H, D, device=dev, dtype=torch.bfloat16)
seq_ids = torch.repeat_interleave(torch.arange(S), lens)[None].to(dev)


def timeit(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


# fp32 reference
ref = torch.empty(1, N, H, D, device=dev, dtype=torch.float32)
for s0, e0 in zip(starts.tolist(), ends.tolist()):
    qi = q[0, s0:e0].float().transpose(0, 1)
    ki = k[0, :e0].float().transpose(0, 1)
    vi = v[0, :e0].float().transpose(0, 1)
    ref[0, s0:e0] = F.scaled_dot_product_attention(qi, ki, vi).transpose(0, 1)


def flex_e1():
    bm = e1flex.create_block_causal_mask_optimized(seq_ids)
    return e1flex.flex_attention_func(q, k, v, block_mask=bm)


bm_pre = e1flex.create_block_causal_mask_optimized(seq_ids)


def flex_only():
    return e1flex.flex_attention_func(q, k, v, block_mask=bm_pre)


def mask_only():
    return e1flex.create_block_causal_mask_optimized(seq_ids)


bounds = list(zip(starts.tolist(), ends.tolist()))


def flash_loop():
    out = torch.empty_like(q)
    for s0, e0 in bounds:
        out[:, s0:e0] = flash_attn_func(q[:, s0:e0], k[:, :e0], v[:, :e0], causal=False)
    return out


PAGE = 256
npad = math.ceil(N / PAGE) * PAGE
Lmax = int(lens.max())
pos = torch.arange(N, device=dev) - torch.repeat_interleave(starts.to(dev), lens.to(dev))
sid = seq_ids[0]
block_table = torch.arange(npad // PAGE, device=dev, dtype=torch.int32).expand(S, -1).contiguous()
cache_seqlens = ends.to(dev, torch.int32)


def flash_paged():
    kc = F.pad(k[0], (0, 0, 0, 0, 0, npad - N)).view(npad // PAGE, PAGE, H, D)
    vc = F.pad(v[0], (0, 0, 0, 0, 0, npad - N)).view(npad // PAGE, PAGE, H, D)
    qp = torch.zeros(S, Lmax, H, D, device=dev, dtype=q.dtype)
    qp[sid, pos] = q[0]
    o = flash_attn_with_kvcache(qp, kc, vc, cache_seqlens=cache_seqlens, block_table=block_table, causal=False)
    return o[sid, pos][None]


pairs = float(sum(ln * e for ln, e in zip(lens.tolist(), ends.tolist())))
flop = 4 * pairs * H * D
print(f"block-causal attention FLOP per global layer: {flop / 1e12:.3f} TFLOP")
for name, fn in [("mask_only", mask_only), ("flex_only", flex_only), ("flex_e1", flex_e1),
                 ("flash_loop", flash_loop), ("flash_paged", flash_paged)]:
    try:
        ms = timeit(fn)
        out = fn()
        if isinstance(out, torch.Tensor):
            err = (out.float() - ref).abs()
            rel = err.norm() / ref.norm()
            print(f"{name:12s} {ms:8.2f} ms  {flop / ms / 1e9:6.1f} TFLOPS  max|err| {err.max():.3e}  rel {rel:.3e}")
        else:
            print(f"{name:12s} {ms:8.2f} ms")
    except Exception as exc:  # report and continue
        print(f"{name:12s} FAILED: {type(exc).__name__}: {str(exc)[:200]}")

# GEMM roofline: the gated-MLP up-projection and the q/k/v projection at this token count
for (m, kk, n) in [(N, 1280, 3840), (N, 1280, 1280), (N, 3840, 1280), (N, 1280, 7680)]:
    a = torch.randn(m, kk, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, kk, device=dev, dtype=torch.bfloat16)
    ms = timeit(lambda: F.linear(a, b))
    print(f"GEMM {m}x{kk}x{n}: {ms:6.2f} ms  {2 * m * kk * n / ms / 1e9:6.1f} TFLOPS")
print("peak alloc GiB", torch.cuda.max_memory_allocated() / 2**30)
