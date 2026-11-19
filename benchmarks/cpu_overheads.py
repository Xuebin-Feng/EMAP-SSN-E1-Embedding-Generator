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

"""CPU-side per-query costs of the driver loop, measured without the GPU.

1. E1BatchPreparer.get_batch_kwargs on CPU for all 12,927 published RA inputs
   (tokenisation, position ids, context decode).
2. validate_embedding_array(require_finite) + h5py create_dataset + flush for
   12,927 float16 arrays of the real query shapes, into a scratch file in benchmarks/
   (same ext4 filesystem as the real run), to see whether the per-query store
   cost grows as the file fills up. The scratch file is deleted afterwards.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402
from E1.batch_preparer import E1BatchPreparer  # noqa: E402

fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    hmap = json.load(fh)["homolog_map"]
targets = list(hmap)
_, seqs, inputs, _ = drv.prepare_e1_inputs(fh_, fs_, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                           target_headers=targets, homolog_map=hmap)

bp = E1BatchPreparer(device=torch.device("cpu"))
t = time.perf_counter()
for x in inputs[:2000]:
    bp.get_batch_kwargs([x], device=torch.device("cpu"))
dt = (time.perf_counter() - t) / 2000
print(f"get_batch_kwargs (CPU part): {dt * 1e3:.2f} ms/query -> {dt * len(inputs):.0f} s for all queries")

path = os.path.join(PROJECT, "benchmarks", "scratch_store_test.h5")
if os.path.exists(path):
    os.remove(path)
rng = np.random.default_rng(42)
store = []
with h5py.File(path, "w") as hf:
    g = hf.create_group("embeddings")
    for h, s in zip(targets, seqs):
        e = rng.standard_normal((len(s), 1280), dtype=np.float32).astype(np.float16)
        t0 = time.perf_counter()
        drv.validate_embedding_array(e, s, "float16", require_finite=True, header=h)
        g.create_dataset(h, data=e)
        hf.flush()
        store.append(time.perf_counter() - t0)
store = np.array(store)
print(f"store (validate + create_dataset + flush): mean {store.mean() * 1e3:.2f} ms, "
      f"first 1000 {store[:1000].mean() * 1e3:.2f} ms, last 1000 {store[-1000:].mean() * 1e3:.2f} ms, "
      f"total {store.sum():.0f} s")
print(f"scratch file size {os.path.getsize(path) / 1e9:.2f} GB")
os.remove(path)
