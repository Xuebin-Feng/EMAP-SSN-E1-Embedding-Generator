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

"""Phase 1 (bit-for-bit) verification of the speed-up fork on the GPU.

Runs the fast driver's real generate_embeddings() on the same random sample as
profile_baseline.py (seed 42) for a ladder of variants, each into its own HDF5 file in
benchmarks/verify/, and compares every embedding with the published file byte for byte:

  fork_noprecast_seq   fork model code (A2, A3, A6), fp32 weights + autocast, sequential loop
  fork_seq             + Linear weights cast to bf16 once (A4)
  fork_pipe            + CPU/GPU pipelined loop (A5)
  fork_pipe_rope       + fused clamp/RoPE kernel (A7)
  fork_pipe_rope_silu  + fused SiLU*gate kernel (A8)   = the phase-1 configuration

Also: full-length BlockMask comparison (original vs fork) for every sampled query, and
stress tests of the two fused kernels against eager PyTorch on random bf16 data.
The published embeddings file is opened read-only.

Usage: python verify_exact.py [--n 64] [--warmup 3]
"""
import argparse
import json
import os
import random
import resource
import sys
import time

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--warmup", type=int, default=3)
args = ap.parse_args()

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
OUT = os.path.join(PROJECT, "benchmarks", "verify")
os.makedirs(OUT, exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1_fast as drv  # noqa: E402  (puts third_party/E1/src first on sys.path)
import E1.model.flex_attention as fx  # noqa: E402
import E1.model.fused_ops as fo  # noqa: E402
from E1.modeling import E1ForMaskedLM  # noqa: E402

assert fx.__file__.startswith(os.path.join(PROJECT, "third_party", "E1")), fx.__file__
summary = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

# Same sample as profile_baseline.py
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    homolog_map = json.load(fh)["homolog_map"]
targets = list(homolog_map)
sample = random.Random(42).sample(targets, args.n + args.warmup)
warm, sample = sample[: args.warmup], sample[args.warmup:]

# ---------------------------------------------------------------- fused-op stress tests
torch.manual_seed(42)
g = torch.Generator(device="cuda").manual_seed(42)
a = (torch.randn(4_000_000, device="cuda", generator=g) * torch.exp(torch.randn(4_000_000, device="cuda", generator=g) * 2)).bfloat16()
b = torch.randn(4_000_000, device="cuda", generator=g).bfloat16()
special = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 80.0, 88.0, 89.0, -80.0, -88.0, -89.0, 1e30, -1e30], device="cuda").bfloat16()
a[: special.numel()] = special
eager = torch.nn.functional.silu(a) * b
fused = fo.silu_mul(a.contiguous(), b.contiguous())
summary["silu_mul_stress_mismatches"] = int((eager.view(torch.int16) != fused.view(torch.int16)).sum())
B, N, H, D = 1, 4099, 20, 64
q, k, v = [(torch.randn(B, N, H * D, device="cuda", generator=g) * 4).bfloat16() for _ in range(3)]
ang = torch.rand(B, N, 1, D // 2, device="cuda", generator=g) * 1000
cos = torch.cat([ang.cos(), ang.cos()], -1).bfloat16()
sin = torch.cat([ang.sin(), ang.sin()], -1).bfloat16()
rot = lambda x: torch.cat((-x[..., D // 2:], x[..., : D // 2]), dim=-1)  # noqa: E731
mism = 0
for heads_first in (False, True):
    qo, ko, vo = fo.rope_qkv(q, k, v, cos, sin, 8.0, D, heads_first)
    qc, kc, vc = (x.view(B, N, H, D).clamp(-8.0, 8.0) for x in (q, k, v))
    qe, ke = qc * cos + rot(qc) * sin, kc * cos + rot(kc) * sin
    for x, y in ((qo, qe), (ko, ke), (vo, vc)):
        mism += int((x.contiguous().view(torch.int16) != y.view(torch.int16)).sum())
summary["rope_qkv_stress_mismatches"] = mism
print("fused-op stress tests (mismatching elements vs eager):",
      summary["silu_mul_stress_mismatches"], summary["rope_qkv_stress_mismatches"])
del a, b, eager, fused, q, k, v, qo, ko, vo, qc, kc, vc, qe, ke

# ---------------------------------------------------------------- model + BlockMask check
t = time.perf_counter()
model = E1ForMaskedLM.from_pretrained(drv.E1_MODEL_MAPPINGS["E1_RA"]).to("cuda").eval()
bp = drv.E1BatchPreparer()
summary["model_load_s"] = round(time.perf_counter() - t, 1)
print("act_fn:", type(model.model.layers[0].ffn.mlp.act_fn).__name__, "clip_qkv:", model.config.clip_qkv)

fasta_headers, fasta_sequences, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
seq_by_h = dict(zip(fasta_headers, fasta_sequences))
_, _, inputs, _ = drv.prepare_e1_inputs(fasta_headers, fasta_sequences, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                        target_headers=warm + sample, homolog_map=homolog_map)
FIELDS = ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
          "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"]
same_masks = 0
for s in inputs:
    sid = bp.get_batch_kwargs([s], device="cuda")["sequence_ids"]
    m0, m1 = fx.create_block_causal_mask_optimized(sid), fx.create_block_causal_mask_from_block_bounds(sid)
    same_masks += all(torch.equal(getattr(m0, f), getattr(m1, f)) for f in FIELDS) and m0.seq_lengths == m1.seq_lengths
summary["blockmask_identical"] = f"{same_masks}/{len(inputs)}"
print("full-length BlockMasks identical:", summary["blockmask_identical"])
torch.cuda.empty_cache()

# ---------------------------------------------------------------- variant ladder
with h5py.File(REF, "r") as ref:
    ref_emb = {h: ref["embeddings"][h][:] for h in sample}


def sequential_batcher(*a, **kw):  # a different function object -> the driver's sequential loop
    return drv.get_e1_embeddings_batch(*a, **kw)


def run(name, headers, pipelined):
    path = os.path.join(OUT, f"{name}.h5")
    if os.path.exists(path):
        os.remove(path)
    marks = {}

    def loader(checkpoint, device):
        torch.cuda.synchronize()
        marks["start"] = time.perf_counter()
        return model, bp

    kw = {} if pipelined else {"embedding_batcher": sequential_batcher}
    drv.generate_embeddings(drv.INPUT_FASTA, path, "RA", "float16", device="cuda",
                            homolog_selector=lambda: (headers, homolog_map), model_loader=loader, **kw)
    torch.cuda.synchronize()
    return path, time.perf_counter() - marks["start"]


def cast_linear_bf16():
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.bfloat16)


ladder = [
    ("fork_noprecast_seq", False, False, False, None),
    ("fork_seq", False, False, False, cast_linear_bf16),
    ("fork_pipe", True, False, False, None),
    ("fork_pipe_rope", True, True, False, None),
    ("fork_pipe_rope_silu", True, True, True, None),
]
summary["variants"] = {}
for name, pipelined, rope, silu, prep in ladder:
    if prep:
        prep()
    fo.ENABLE_ROPE, fo.ENABLE_SILU_MUL = rope, silu
    run(name + "_warmup", warm, pipelined)  # compiles flex / Triton kernels for this setting
    torch.cuda.reset_peak_memory_stats()
    path, loop_s = run(name, sample, pipelined)
    exact, maxabs = 0, 0.0
    with h5py.File(path, "r") as hf:
        for h in sample:
            e = hf["embeddings"][h][:]
            exact += int(np.array_equal(e, ref_emb[h]))
            maxabs = max(maxabs, float(np.abs(e.astype(np.float32) - ref_emb[h].astype(np.float32)).max()))
    res = {"s_per_query": round(loop_s / len(sample), 4), "bitwise_identical": f"{exact}/{len(sample)}",
           "max_abs_diff": maxabs, "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    summary["variants"][name] = res
    print(f"{name:22s} {res}")
summary["peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
with open(os.path.join(OUT, "verify_exact_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
