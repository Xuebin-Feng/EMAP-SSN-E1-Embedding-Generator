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

"""NORM (single-sequence) vs RA (retrieval-augmented) E1 embeddings.

1. Original vs fast driver in NORM mode: bit-for-bit, for all sequences.
2. RA (published file) vs NORM (original driver): how much the 50-homolog context
   changes each embedding (per-residue cosine, per-protein cosine of mean-pooled
   embeddings), and the time per query of each mode.
Writes benchmarks/norm/compare_norm.json.
"""
import json
import os

import h5py
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
REF = "Embeddings/Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5"
res = {}
with h5py.File("benchmarks/norm/norm_original.h5", "r") as fo, h5py.File("benchmarks/norm/norm_fast.h5", "r") as ff, \
        h5py.File(REF, "r") as fr:
    heads = list(fo["embeddings"].keys())
    assert sorted(heads) == sorted(ff["embeddings"].keys())
    exact = sum(np.array_equal(fo["embeddings"][h][:], ff["embeddings"][h][:]) for h in heads)
    res["norm_original_vs_fast_bitwise"] = f"{exact}/{len(heads)}"
    cos_res, cos_pool = [], []
    for h in heads:
        a = fr["embeddings"][h][:].astype(np.float64)  # RA
        b = fo["embeddings"][h][:].astype(np.float64)  # NORM
        cos_res.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
        pa, pb = a.mean(0), b.mean(0)
        cos_pool.append(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb)))
    cos_res, cos_pool = np.concatenate(cos_res), np.array(cos_pool)
    res["ra_vs_norm_residue_cosine"] = {q: float(np.percentile(cos_res, p)) for q, p in
                                        [("p1", 1), ("p10", 10), ("median", 50), ("p90", 90)]}
    res["ra_vs_norm_residue_cosine"]["mean"] = float(cos_res.mean())
    res["ra_vs_norm_protein_meanpool_cosine"] = {q: float(np.percentile(cos_pool, p)) for q, p in
                                                 [("min", 0), ("p10", 10), ("median", 50), ("max", 100)]}
for tag in ("norm_original", "norm_fast"):
    with open(f"benchmarks/norm/{tag}.json") as fh:
        res[tag] = json.load(fh)
with open("benchmarks/norm/compare_norm.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
