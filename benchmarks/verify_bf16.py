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

"""Phase 2 (bf16-level) measurements (third_party/E1 with the switches in E1/model/bf16_options.py).

Same random sample as phase 1 (seed 42), through the fast driver's real
generate_embeddings (pipelined, all phase-1 changes on), one HDF5 per variant in
benchmarks/verify_bf16/. Variants (bf16_options switches):

  control            all phase-2 switches off      -> must still be bit-for-bit
  merged_gemms       MERGED_GEMMS only
  fused_add_norm     FUSED_ADD_NORM only
  query_only_last    QUERY_ONLY_LAST_LAYER only
  all_phase2         all three

For each: s/query, peak GPU memory and how far the embeddings are from the published
ones (per-residue cosine, per-protein cosine of mean-pooled embeddings, absolute and
relative differences, share of fp16 values that are exactly equal).

Usage: python verify_bf16.py [--n 64] [--warmup 3]
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
os.environ["E1_SOURCE_DIR"] = os.path.join(PROJECT, "third_party", "E1", "src")
# All switches on when the model loads, as on the original speedup-bf16 branch, so
# MERGED_GEMMS merges the weights; each variant below then sets the switches it needs.
os.environ["E1_BF16_SPEEDUPS"] = "1"
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
OUT = os.path.join(PROJECT, "benchmarks", "verify_bf16")
os.makedirs(OUT, exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1_fast as drv  # noqa: E402
import E1.model.bf16_options as opt  # noqa: E402

assert opt.__file__.startswith(os.path.join(PROJECT, "third_party", "E1")), opt.__file__

with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    homolog_map = json.load(fh)["homolog_map"]
sample = random.Random(42).sample(list(homolog_map), args.n + args.warmup)
warm, sample = sample[: args.warmup], sample[args.warmup:]
with h5py.File(REF, "r") as ref:
    ref_emb = {h: ref["embeddings"][h][:] for h in sample}

t = time.perf_counter()
model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")  # bf16 cast + merged weights
print(f"model load {time.perf_counter() - t:.1f} s")


def run(name, headers):
    path = os.path.join(OUT, f"{name}.h5")
    if os.path.exists(path):
        os.remove(path)
    marks = {}

    def loader(checkpoint, device):
        torch.cuda.synchronize()
        marks["start"] = time.perf_counter()
        return model, bp

    drv.generate_embeddings(drv.INPUT_FASTA, path, "RA", "float16", device="cuda",
                            homolog_selector=lambda: (headers, homolog_map), model_loader=loader)
    torch.cuda.synchronize()
    return path, time.perf_counter() - marks["start"]


def differences(path):
    exact = 0
    cos_res, cos_pool, absd, eq, sq_d, sq_v, n_el = [], [], [], 0, 0.0, 0.0, 0
    with h5py.File(path, "r") as hf:
        for h in sample:
            e = hf["embeddings"][h][:]
            r = ref_emb[h]
            exact += int(np.array_equal(e, r))
            eq += int((e == r).sum())
            n_el += r.size
            a, b = e.astype(np.float64), r.astype(np.float64)
            cos_res.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
            pa, pb = a.mean(0), b.mean(0)
            cos_pool.append(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb)))
            d = np.abs(a - b)
            absd.append(d.ravel())
            sq_d += float((d ** 2).sum())
            sq_v += float((b ** 2).sum())
    cos_res, absd = np.concatenate(cos_res), np.concatenate(absd)
    return {
        "bitwise_identical_proteins": f"{exact}/{len(sample)}",
        "fp16_values_exactly_equal": round(eq / n_el, 6),
        "residue_cosine_min": float(cos_res.min()),
        "residue_cosine_p0.1": float(np.percentile(cos_res, 0.1)),
        "residue_cosine_mean": float(cos_res.mean()),
        "protein_meanpool_cosine_min": float(np.min(cos_pool)),
        "abs_diff_max": float(absd.max()),
        "abs_diff_mean": float(absd.mean()),
        "rms_diff_over_rms_value": float(np.sqrt(sq_d / sq_v)),
    }


variants = [
    ("control", False, False, False),
    ("merged_gemms", True, False, False),
    ("fused_add_norm", False, True, False),
    ("query_only_last", False, False, True),
    ("all_phase2", True, True, True),
]
summary = {"variants": {}}
for name, merged, fused, query_only in variants:
    opt.MERGED_GEMMS, opt.FUSED_ADD_NORM, opt.QUERY_ONLY_LAST_LAYER = merged, fused, query_only
    run(name + "_warmup", warm)
    torch.cuda.reset_peak_memory_stats()
    path, loop_s = run(name, sample)
    res = {"s_per_query": round(loop_s / len(sample), 4),
           "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    res.update(differences(path))
    summary["variants"][name] = res
    print(f"{name:18s} {res}")
summary["peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
with open(os.path.join(OUT, "verify_bf16_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
