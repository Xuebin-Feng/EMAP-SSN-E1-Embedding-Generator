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

"""Generate NORM-mode (single-sequence) E1 embeddings for all sequences with either the
original driver (+ the installed E1: unmodified Profluent-E1 for the original numbers)
or the fast driver (+ third_party/E1), into
benchmarks/norm/<tag>.h5, and record wall time, per-query time and peak memory.

Usage: python norm_run.py --driver original|fast --tag NAME
"""
import argparse
import importlib
import json
import os
import resource
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--driver", choices=["original", "fast"], required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
drv = importlib.import_module("Generate_Embeddings_E1" if args.driver == "original" else "Generate_Embeddings_E1_fast")
import E1.modeling as e1m  # noqa: E402

os.makedirs("benchmarks/norm", exist_ok=True)
out = f"benchmarks/norm/{args.tag}.h5"
if os.path.exists(out):
    os.remove(out)
marks = {}


def loader(checkpoint, device):
    t = time.perf_counter()
    r = drv.load_e1_model(checkpoint, device)
    torch.cuda.synchronize()
    marks["load_s"] = time.perf_counter() - t
    marks["loop_start"] = time.perf_counter()
    return r


t0 = time.perf_counter()
n = drv.generate_embeddings(drv.INPUT_FASTA, out, "NORM", "float16", model_loader=loader, device="cuda")
torch.cuda.synchronize()
t1 = time.perf_counter()
res = {"driver": args.driver, "e1_source": e1m.__file__, "n": n, "wall_s": round(t1 - t0, 1),
       "model_load_s": round(marks["load_s"], 1), "loop_s": round(t1 - marks["loop_start"], 1),
       "s_per_query": round((t1 - marks["loop_start"]) / n, 4),
       "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
       "peak_rss_GiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
with open(f"benchmarks/norm/{args.tag}.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(res)
