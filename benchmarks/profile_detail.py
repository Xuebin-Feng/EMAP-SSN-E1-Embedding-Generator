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

"""Follow-up to profile_baseline.py: split the 'elementwise / other' bucket.

Measures, on the median-token RA query and with the UNMODIFIED model:
  1. full forward (inference_mode + bf16 autocast, as in the driver)
  2. block-mask construction as it happens inside that forward (runtime wrapper
     around E1.modeling.create_block_causal_mask_optimized; no file is changed)
  3. mask construction and compiled flex attention alone, with and without
     inference_mode (the earlier microbenchmark ran without it)
  4. a profiler table of one forward, sorted by self CUDA time
Outputs: stdout only.
"""
import json
import os
import sys

import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402
import E1.modeling as e1m  # noqa: E402
import E1.model.flex_attention as e1flex  # noqa: E402

torch.manual_seed(42)
fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    hmap = json.load(fh)["homolog_map"]
targets = list(hmap)
_, _, inputs, _ = drv.prepare_e1_inputs(fh_, fs_, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                        target_headers=targets, homolog_map=hmap)
tok = [sum(len(s) + 4 for s in x.split(",")) for x in inputs]
order = sorted(range(len(inputs)), key=tok.__getitem__)
med = order[len(order) // 2]
print(f"median query: {tok[med]} tokens")

model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")
b = bp.get_batch_kwargs([inputs[med]], device="cuda")


def fwd():
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids=b["input_ids"], within_seq_position_ids=b["within_seq_position_ids"],
                     global_position_ids=b["global_position_ids"], sequence_ids=b["sequence_ids"],
                     use_cache=False, output_attentions=False, output_hidden_states=False)


def cuda_ms(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


for i in order[len(order) // 2 - 2: len(order) // 2]:  # warm-up incl. compile
    bb = bp.get_batch_kwargs([inputs[i]], device="cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(**{k: bb[k] for k in ("input_ids", "within_seq_position_ids", "global_position_ids", "sequence_ids")},
              use_cache=False)
print(f"1. forward: {cuda_ms(fwd):.1f} ms")

# 2. mask construction inside the real forward
orig = e1m.create_block_causal_mask_optimized
mask_ms = []


def timed_mask(seq_ids):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = orig(seq_ids)
    e.record()
    torch.cuda.synchronize()
    mask_ms.append(s.elapsed_time(e))
    return out


e1m.create_block_causal_mask_optimized = timed_mask
for _ in range(3):
    fwd()
e1m.create_block_causal_mask_optimized = orig
print(f"2. mask construction inside forward: {[round(x, 1) for x in mask_ms]} ms")

# 3. isolated mask + flex, with / without inference_mode
seq = b["sequence_ids"]
N = seq.shape[1]
q, k, v = (torch.randn(1, N, 20, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
with torch.inference_mode():
    bm_inf = orig(seq)
bm_grad = orig(seq)
for label, ctx in [("inference_mode", torch.inference_mode), ("grad enabled  ", torch.enable_grad)]:
    def m():
        with ctx():
            return orig(seq)

    def f():
        with ctx():
            return e1flex.flex_attention_func(q, k, v, block_mask=bm_inf)
    print(f"3. {label}: mask {cuda_ms(m):7.1f} ms   flex (prebuilt mask) {cuda_ms(f):7.1f} ms")

# 4. profiler table for one forward
from torch.profiler import ProfilerActivity, profile  # noqa: E402

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    fwd()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70))
print("peak alloc GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
