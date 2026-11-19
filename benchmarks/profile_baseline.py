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

"""Profile the UNMODIFIED E1 RA embedding loop on a fixed random sample of queries.

Nothing in E1/ or Generate_Embeddings_E1.py is changed: both are imported as-is.
Outputs go to benchmarks/ only. The reference embedding file is opened read-only.

Stages timed per query (with explicit synchronisation between stages):
  prep    : E1BatchPreparer.get_batch_kwargs (CPU tokenisation + H2D copy)
  forward : model forward under bf16 autocast (as in the driver)
  extract : query-row selection, D2H copy, float16 conversion (as in the driver)
  store   : validate_embedding_array + HDF5 create_dataset + flush (as in the driver)

Then a torch.profiler pass over a few queries gives the GPU kernel breakdown and
the GPU idle fraction, and every sampled embedding is compared with the
reference file (bitwise equality, per-residue cosine).

Usage: python profile_baseline.py [--n 64] [--warmup 3] [--prof 4]
"""
import argparse
import json
import os
import random
import resource
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
WORK = os.path.join(PROJECT, "benchmarks")
CACHE = os.path.join(WORK, "homolog_map.json")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--prof", type=int, default=4)
ap.add_argument("--tag", default="baseline")
args = ap.parse_args()

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402
import E1.modeling as e1m  # noqa: E402

print("triton layer-norm kernel loaded:", e1m.layer_norm is not None)
from E1.model.flash_attention import is_flash_attention_available  # noqa: E402

print("flash-attn available:", is_flash_attention_available())

fasta_headers, fasta_sequences, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
with open(CACHE) as fh:
    homolog_map = json.load(fh)["homolog_map"]
targets = list(homolog_map)
_, _, input_strings, input_hashes = drv.prepare_e1_inputs(
    fasta_headers, fasta_sequences, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
    target_headers=targets, homolog_map=homolog_map,
)
inp_by_h = dict(zip(targets, input_strings))
seq_by_h = dict(zip(fasta_headers, fasta_sequences))

rng = random.Random(42)
sample = rng.sample(targets, args.n + args.warmup)
warm, sample = sample[: args.warmup], sample[args.warmup:]

t = time.perf_counter()
model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")
t_load = time.perf_counter() - t
print(f"model load: {t_load:.1f} s; param dtype {next(model.parameters()).dtype}")

out_path = os.path.join(WORK, f"test_{args.tag}.h5")
if os.path.exists(out_path):
    os.remove(out_path)


def forward(batch):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            return model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                use_cache=False, output_attentions=False, output_hidden_states=False,
            )


def extract(batch, outputs):
    emb = outputs.embeddings
    last = batch["sequence_ids"].max(dim=1)[0][:, None]
    mask = (batch["sequence_ids"] == last) & ~bp.get_boundary_token_mask(batch["input_ids"])
    return emb[0, mask[0]].cpu().float().numpy().astype(np.float16)


# Warm-up (includes torch.compile of flex_attention); timed separately.
t = time.perf_counter()
for h in warm:
    b = bp.get_batch_kwargs([inp_by_h[h]], device="cuda")
    extract(b, forward(b))
torch.cuda.synchronize()
t_warm = time.perf_counter() - t
print(f"warm-up ({len(warm)} queries, includes compile): {t_warm:.1f} s")

torch.cuda.reset_peak_memory_stats()
times = defaultdict(list)
tokens = []
results = {}
t_all = time.perf_counter()
with h5py.File(out_path, "w") as hf:
    grp = hf.create_group("embeddings")
    for h in sample:
        t0 = time.perf_counter()
        b = bp.get_batch_kwargs([inp_by_h[h]], device="cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        o = forward(b)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        e = extract(b, o)
        t3 = time.perf_counter()
        drv.validate_embedding_array(e, seq_by_h[h], "float16", require_finite=True, header=h)
        grp.create_dataset(h, data=e)
        hf.flush()
        t4 = time.perf_counter()
        del o, b
        times["prep"].append(t1 - t0)
        times["forward"].append(t2 - t1)
        times["extract"].append(t3 - t2)
        times["store"].append(t4 - t3)
        tokens.append(sum(len(s) + 4 for s in inp_by_h[h].split(",")))
        results[h] = e
wall = time.perf_counter() - t_all
peak_alloc = torch.cuda.max_memory_allocated() / 2**30
peak_res = torch.cuda.max_memory_reserved() / 2**30

print(f"\n== {args.n} queries, wall {wall:.1f} s = {wall / args.n:.3f} s/query ==")
for k, v in times.items():
    v = np.array(v)
    print(f"  {k:8s} mean {v.mean() * 1e3:8.1f} ms  median {np.median(v) * 1e3:8.1f} ms  share {v.sum() / wall * 100:5.1f} %")
tok = np.array(tokens)
fw = np.array(times["forward"])
print(f"  tokens/query mean {tok.mean():.0f}; forward ms per 1k tokens {np.mean(fw / tok * 1e6):.1f}")
coef = np.polyfit(tok, fw, 2)
print(f"  forward(s) ~ {coef[0]:.3e}*T^2 + {coef[1]:.3e}*T + {coef[2]:.3e}")
print(f"  peak GPU allocated {peak_alloc:.2f} GiB, reserved {peak_res:.2f} GiB; "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB")

# Equivalence with the published file
print("\n== Comparison with the published embeddings ==")
exact = 0
cos_min, cos_mean, maxabs = [], [], []
with h5py.File(REF, "r") as ref:
    for h, e in results.items():
        r = ref["embeddings"][h][:]
        exact += int(np.array_equal(r, e))
        a, bb = r.astype(np.float32), e.astype(np.float32)
        c = (a * bb).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(bb, axis=1))
        cos_min.append(c.min())
        cos_mean.append(c.mean())
        maxabs.append(np.abs(a - bb).max())
print(f"  bitwise identical: {exact}/{len(results)}")
print(f"  per-residue cosine: min {min(cos_min):.6f}, mean {np.mean(cos_mean):.6f}; max |diff| {max(maxabs):.4g}")

# Kernel-level breakdown
if args.prof:
    from torch.profiler import ProfilerActivity, profile

    prof_inputs = [bp.get_batch_kwargs([inp_by_h[h]], device="cuda") for h in sample[: args.prof]]
    torch.cuda.synchronize()
    t = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in prof_inputs:
            o = forward(b)
            extract(b, o)
        torch.cuda.synchronize()
    pwall = time.perf_counter() - t
    cats = defaultdict(float)
    total = 0.0
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = ev.name.lower()
        dur = ev.time_range.elapsed_us()  # device-side event: kernel duration in us
        total += dur
        if "flash" in name:
            cats["flash-attn (within-seq layers)"] += dur
        elif "flex" in name or "triton_tem" in name:
            cats["flex attention (global layers)"] += dur
        elif any(s in name for s in ("gemm", "cutlass", "xmma", "sm80", "sm89", "ampere", "cublas")):
            cats["matmul (cuBLAS)"] += dur
        elif "triton" in name or "layer_norm" in name or "rms" in name:
            cats["other triton (RMSNorm etc.)"] += dur
        elif "memcpy" in name or "memset" in name:
            cats["memcpy/memset"] += dur
        else:
            cats["elementwise / indexing / other"] += dur
    print(f"\n== Kernel breakdown over {args.prof} queries (profiler wall {pwall:.2f} s) ==")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k:34s} {v / 1e3:9.1f} ms  {v / total * 100:5.1f} %")
    print(f"  GPU busy {total / 1e3:.1f} ms of {pwall * 1e3:.1f} ms wall -> idle {100 - total / 1e4 / pwall:.1f} %")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
