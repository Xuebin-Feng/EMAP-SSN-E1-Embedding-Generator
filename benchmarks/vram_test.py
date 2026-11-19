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

"""Does E1 RA generation overflow the 12 GB of VRAM, and what happens when it does?

Runs the 24 longest RA inputs (the largest memory need) with one driver:
  phase 'clear'    : nothing else allocated -> per-query time and the extra memory each
                     forward needs on top of the loaded model
  phase 'fit_1GiB' / 'short_1GiB' / 'short_3GiB' (original driver): a filler tensor
                     occupies VRAM like other applications would, sized so the original
                     has 1 GiB to spare, or is 1 / 3 GiB short. Windows (and WSL2) by
                     default spill CUDA memory to system RAM instead of failing ("sysmem
                     fallback"): that shows up as a slowdown, or as out-of-memory errors
                     if fallback is off.
  --fillers=G1,G2,... : fixed filler sizes in GiB, the same for both drivers (the same
                     "other applications"); phases are then named filler_<G>GiB.
                     Write "--fillers=..." with "=": argparse takes "-4.6,..." for an option.
18 Nov 20:41: the original needed 11.39 GiB on top of the model on a clear GPU (14.4 GiB in
all, 7.76 GiB free), i.e. it already spilled ~3.6 GiB to system RAM with no filler, so the
automatic sizes were negative. Rerun with fixed fillers: --fillers=2,4,6 for both drivers.
Output: benchmarks/vram/<driver>.json, or <driver>_fillers.json with --fillers. Usage:
  python vram_test.py --driver original
  python vram_test.py --driver fast --fillers=2,4,6
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--driver", choices=["original", "fast"], required=True)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--fillers", default="")
args = ap.parse_args()

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
drv = importlib.import_module("Generate_Embeddings_E1" if args.driver == "original" else "Generate_Embeddings_E1_fast")
GiB = 2**30

with open("benchmarks/homolog_map.json") as fh:
    hmap = json.load(fh)["homolog_map"]
fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
L = {h: len(s) + 4 for h, s in zip(fh_, fs_)}
tok = {q: sum(L[x] for x in hmap[q] + [q]) for q in hmap}
longest = sorted(hmap, key=tok.get)[-args.n:]
_, _, inputs, _ = drv.prepare_e1_inputs(fh_, fs_, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                        target_headers=longest, homolog_map=hmap)
model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")
drv.get_e1_embeddings_batch([inputs[0]], model, bp, "cuda", np.float16)  # warm-up / compile
torch.cuda.synchronize()
torch.cuda.empty_cache()


def gpu_used_mib():
    out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def phase(label, filler_gib):
    torch.cuda.empty_cache()
    filler = None
    res = {"filler_GiB": round(filler_gib, 2)}
    if filler_gib > 0:
        try:
            filler = torch.empty(int(filler_gib * GiB), dtype=torch.uint8, device="cuda")
            filler.fill_(1)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            res["filler_error"] = str(exc)[:200]
            return res
    res["free_after_filler_GiB"] = round(torch.cuda.mem_get_info()[0] / GiB, 2)
    base_reserved = torch.cuda.memory_reserved()
    times, extra, ooms = [], [], 0
    for s in inputs:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            drv.get_e1_embeddings_batch([s], model, bp, "cuda", np.float16)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            ooms += 1
            torch.cuda.empty_cache()
            continue
        times.append(time.perf_counter() - t)
        extra.append((torch.cuda.max_memory_reserved() - base_reserved) / GiB)
    res.update({"queries_ok": len(times), "oom_errors": ooms,
                "s_per_query_mean": round(float(np.mean(times)), 3) if times else None,
                "s_per_query_max": round(float(np.max(times)), 3) if times else None,
                "extra_reserved_GiB_max": round(max(extra), 2) if extra else None,
                # more than was free after the filler: backed by system RAM (sysmem fallback)
                "spill_GiB_est": round(max(0.0, max(extra) - res["free_after_filler_GiB"]), 2) if extra else None,
                "whole_gpu_used_MiB_after": gpu_used_mib()})
    del filler
    torch.cuda.empty_cache()
    print(f"{label:12s} {res}", flush=True)
    return res


out = {"driver": args.driver, "tokens": [tok[q] for q in longest],
       "model_reserved_GiB": round(torch.cuda.memory_reserved() / GiB, 2),
       "free_before_GiB": round(torch.cuda.mem_get_info()[0] / GiB, 2),
       "total_GiB": round(torch.cuda.mem_get_info()[1] / GiB, 2), "phases": {}}
print({k: v for k, v in out.items() if k != "tokens"}, f"tokens {min(out['tokens'])}-{max(out['tokens'])}", flush=True)
out["phases"]["clear"] = clear = phase("clear", 0)
out["peak_total_reserved_GiB_clear"] = round(out["model_reserved_GiB"] + clear["extra_reserved_GiB_max"], 2)
if args.fillers:
    sizes = [float(x) for x in args.fillers.split(",")]
    names = [f"filler_{s:g}GiB" for s in sizes]
else:
    need = clear["extra_reserved_GiB_max"]
    free = out["free_before_GiB"]
    sizes = [free - need - 1, free - need + 1, free - need + 3]
    names = ["fit_1GiB", "short_1GiB", "short_3GiB"]
    if free - need - 1 <= 0:  # already short of VRAM with nothing else on the GPU
        out["overflows_without_filler_GiB"] = round(need - free, 2)
        sizes, names = [], []
for name, size in zip(names, sizes):
    out["phases"][name] = phase(name, size)
os.makedirs("benchmarks/vram", exist_ok=True)
with open(f"benchmarks/vram/{args.driver}{'_fillers' if args.fillers else ''}.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("fillers GiB:", ",".join(str(round(s, 2)) for s in sizes))
