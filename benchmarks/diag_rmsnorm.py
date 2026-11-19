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

"""Is the RMSNorm kernel's autotuning why the full exact run differs from the published file?

E1 (the unmodified clone and the fork alike) loads its RMSNorm with
get_kernel("kernels-community/triton-layer-norm"), unpinned. That Triton kernel is
autotuned: on first use in each process it times num_warps 1, 2, 4, 8, 16 and 32 on the
live GPU and keeps the fastest, per autotune key (N, flags and tensor dtypes). Different
warp counts sum the squares in a different order, so a near-tie can flip between
processes and change every embedding slightly. That would explain why the full exact run
(18 Nov 17:20) matches the published file on no protein while the 64-query ladder
(16:52), the unmodified profile (15:02) and the NORM runs match bit for bit.

Steps (GPU, ~3-5 min, the published file is opened read-only):
  0. Fresh autotuning on the first query in file order, as any new process does: which
     keys exist and what is chosen.
  1. Each warp count forced: 8 RA queries (the first 4 in file order, which the full run
     also started with, and 4 of the ladder sample) compared with the published file and
     with the 17:20 full run's output. The warp count that reproduces the published file
     is the one the 2 Aug run's autotuner picked; the one that reproduces the full run is
     the one the 17:20 run's autotuner picked.
     If the run has several autotune keys and no single warp count reproduces a file,
     every per-key combination is screened on 2 queries and confirmed on all 8.
  2. Ten fresh autotunings on the real first-layer input: timings and choices.
Output: benchmarks/diag/rmsnorm_autotune.json. If one warp count reproduces all 8 published
embeddings, the fastest such count is written to benchmarks/rmsnorm_pin.env as
E1_RMSNORM_NUM_WARPS=<w> (the fork reads that variable and skips the autotuning).
"""
import importlib
import itertools
import json
import os
import random
import sys
import time

import h5py
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
os.environ.pop("E1_RMSNORM_NUM_WARPS", None)  # this script sets the warp count itself
import Generate_Embeddings_E1_fast as drv  # noqa: E402
import E1.modeling as e1m  # noqa: E402

REF = f"{PROJECT}/Embeddings/Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5"
FULL = f"{PROJECT}/benchmarks/full/exact_[E1_RA]_embeddings.h5"
assert e1m.layer_norm is not None, "kernels-hub RMSNorm did not load"
ln_mod = importlib.import_module(e1m.layer_norm.__name__ + ".layer_norm")
kern = ln_mod._layer_norm_fwd_1pass_kernel
ALL_CONFIGS = list(getattr(kern, "_e1_all_configs", kern.configs))
BY_W = {c.num_warps: c for c in ALL_CONFIGS}
WARPS = list(BY_W)
print("kernel file:", ln_mod.__file__)
print("autotune configs (num_warps):", WARPS, "key:", kern.keys, flush=True)


def autotune_on():
    kern.configs = list(ALL_CONFIGS)
    kern.cache.clear()


def force(w):  # one warp count for every key
    kern.configs = [BY_W[w]]
    kern.cache.clear()


def force_per_key(keys, combo):  # a warp count per key, via the autotuner's own cache
    kern.configs = list(ALL_CONFIGS)
    kern.cache.clear()
    for k, w in zip(keys, combo):
        kern.cache[k] = BY_W[w]


hmap = json.load(open(f"{PROJECT}/benchmarks/homolog_map.json"))["homolog_map"]
fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
ladder_sample = random.Random(42).sample(list(hmap), 67)[3:]
heads = [h for h in fh_ if h in hmap][:4] + ladder_sample[:4]
tgt, _, inputs, _ = drv.prepare_e1_inputs(fh_, fs_, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                          target_headers=heads, homolog_map=hmap)
assert list(tgt) == heads
with h5py.File(REF, "r") as fr, h5py.File(FULL, "r") as ff:
    ref = {h: fr["embeddings"][h][:] for h in heads}
    full = {h: ff["embeddings"][h][:] for h in heads}


def run(idx):
    """Embed queries idx; return (n equal to published, n equal to the 17:20 run, s/query)."""
    n_ref = n_full = 0
    t = time.perf_counter()
    for i in idx:
        e = drv.get_e1_embeddings_batch([inputs[i]], model, bp, "cuda", np.float16)[0]
        n_ref += int(np.array_equal(e, ref[heads[i]]))
        n_full += int(np.array_equal(e, full[heads[i]]))
    torch.cuda.synchronize()
    return n_ref, n_full, round((time.perf_counter() - t) / len(idx), 3)


out = {"kernel_file": ln_mod.__file__, "num_warps_tried": WARPS, "queries": heads}
model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")
first_norm = next(m for m in model.modules() if isinstance(m, e1m.RMSNorm))

# Step 0: fresh autotuning during the first forward (first query in file order)
autotune_on()
captured = {}
hook = first_norm.register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0].detach().clone()))
n_ref0, n_full0, _ = run([0])
hook.remove()
keys = list(kern.cache)
x0 = captured["x"]
out["fresh_autotune_first_forward"] = {
    "keys": [str(k) for k in keys], "chosen": [kern.cache[k].num_warps for k in keys],
    "first_query_equal_to_published": bool(n_ref0), "first_query_equal_to_full_run": bool(n_full0)}
print("step 0:", out["fresh_autotune_first_forward"], flush=True)
print("first-layer RMSNorm input:", tuple(x0.shape), x0.dtype, flush=True)

# Step 1: each warp count forced (also compiles each variant before it is compared)
ALL8 = list(range(len(heads)))
out["forced"] = {}
for w in WARPS:
    force(w)
    n_ref, n_full, spq = run(ALL8)
    out["forced"][w] = {"equal_to_published": f"{n_ref}/8", "equal_to_full_run_17_20": f"{n_full}/8",
                        "s_per_query": spq}
    print(f"num_warps={w:2d}", out["forced"][w], flush=True)
pub_w = [w for w in WARPS if out["forced"][w]["equal_to_published"] == "8/8"]
full_w = [w for w in WARPS if out["forced"][w]["equal_to_full_run_17_20"] == "8/8"]

if len(keys) > 1 and (not pub_w or not full_w):
    combos_pub, combos_full = [], []
    for combo in itertools.product(WARPS, repeat=len(keys)):
        force_per_key(keys, combo)
        n_ref, n_full, _ = run([0, 4])
        if set(kern.cache) != set(keys):
            print("warning: new autotune key during the run:", set(kern.cache) - set(keys))
        if n_ref == 2:
            combos_pub.append(combo)
        if n_full == 2:
            combos_full.append(combo)
    out["per_key_combos"] = {}
    for label, combos in (("published", combos_pub), ("full_run_17_20", combos_full)):
        confirmed = []
        for combo in combos:
            force_per_key(keys, combo)
            n_ref, n_full, _ = run(ALL8)
            if (n_ref if label == "published" else n_full) == 8:
                confirmed.append(list(combo))
        out["per_key_combos"][label] = confirmed
        print(f"per-key warp counts reproducing {label}:", confirmed, flush=True)

# Step 2: what the autotuner picks on the real first-layer input, 10 fresh tunings
trials = []
for _ in range(10):
    autotune_on()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first_norm(x0)
    torch.cuda.synchronize()
    timings = {c.num_warps: round(float(v[0] if isinstance(v, (list, tuple)) else v) * 1000, 1)
               for c, v in kern.configs_timings.items()}
    trials.append({"chosen": kern.best_config.num_warps, "timings_us": timings})
    print("autotune trial:", trials[-1], flush=True)
out["autotune_trials"] = trials
out["autotune_choices"] = {w: sum(t["chosen"] == w for t in trials) for w in WARPS}

out["published_file_num_warps"] = pub_w  # = the 2 Aug run's autotuner choice
out["full_run_17_20_num_warps"] = full_w  # = the 18 Nov 17:20 run's autotuner choice
os.makedirs("benchmarks/diag", exist_ok=True)
with open("benchmarks/diag/rmsnorm_autotune.json", "w") as fh:
    json.dump(out, fh, indent=1)
if os.path.exists("benchmarks/rmsnorm_pin.env"):
    os.remove("benchmarks/rmsnorm_pin.env")
if pub_w:
    mean_t = {w: np.mean([t["timings_us"][w] for t in trials]) for w in pub_w}
    pin = min(pub_w, key=mean_t.get)
    with open("benchmarks/rmsnorm_pin.env", "w") as fh:
        fh.write(f"E1_RMSNORM_NUM_WARPS={pin}\n")
    print("pin:", pin)
print("CONCLUSION: published file (2 Aug) reproduced by num_warps", pub_w or "none (single count)",
      "| 17:20 full run reproduced by num_warps", full_w or "none (single count)",
      "| autotune choices in 10 fresh tunings:", out["autotune_choices"])
