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

"""Why is the full exact run not bit-for-bit equal to the published file when the
64-query ladder (verify_exact.py) was? CPU only; reads HDF5 files, writes nothing.

Compares, for the ladder's 64 sample proteins (seed 42, same draw as verify_exact.py):
  full exact run vs published, ladder final variant vs published, full run vs ladder
and, for the first proteins in file order, full run vs published (does it differ from
the very first query?). Also NORM fast vs original on the first proteins.
"""
import json
import os
import random

import h5py
import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = f"{P}/Embeddings/Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5"
FULL = f"{P}/benchmarks/full/exact_[E1_RA]_embeddings.h5"
LAD = f"{P}/benchmarks/verify/fork_pipe_rope_silu.h5"
LAD0 = f"{P}/benchmarks/verify/fork_noprecast_seq.h5"

hmap = json.load(open(f"{P}/benchmarks/homolog_map.json"))["homolog_map"]
sample = random.Random(42).sample(list(hmap), 67)[3:]


def stats(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return {"equal": bool(np.array_equal(a, b)), "frac_eq": round(float((a == b).mean()), 4),
            "maxabs": float(np.abs(a64 - b64).max()),
            "rel_rms": round(float(np.sqrt(((a64 - b64) ** 2).sum() / (b64 ** 2).sum())), 5)}


with h5py.File(REF, "r") as fr, h5py.File(FULL, "r") as ff, h5py.File(LAD, "r") as fl, h5py.File(LAD0, "r") as fl0:
    heads = [h.decode() if isinstance(h, bytes) else h for h in fr["headers"][:]]
    n = {"full=ref": 0, "lad=ref": 0, "full=lad": 0, "lad0=ref": 0}
    fr_eq = []
    for h in sample:
        r, f, l, l0 = (x["embeddings"][h][:] for x in (fr, ff, fl, fl0))
        n["full=ref"] += np.array_equal(f, r)
        n["lad=ref"] += np.array_equal(l, r)
        n["full=lad"] += np.array_equal(f, l)
        n["lad0=ref"] += np.array_equal(l0, r)
        fr_eq.append(stats(f, r)["frac_eq"])
    print("64 ladder proteins, bitwise equal counts:", n)
    print("full vs ref, fraction of equal values over the 64: min %.4f median %.4f max %.4f"
          % (min(fr_eq), float(np.median(fr_eq)), max(fr_eq)))
    print("dtypes/shapes (first sample):", fr["embeddings"][sample[0]].dtype, ff["embeddings"][sample[0]].dtype,
          fr["embeddings"][sample[0]].shape, ff["embeddings"][sample[0]].shape)
    print("\nfirst 8 proteins in file order, full vs ref:")
    for h in heads[:8]:
        print(" ", h[:40], stats(ff["embeddings"][h][:], fr["embeddings"][h][:]))
    # attributes / dataset creation details
    print("\nref attrs:", dict(fr.attrs))
    print("full attrs:", dict(ff.attrs))

NO, NF = f"{P}/benchmarks/norm/norm_original.h5", f"{P}/benchmarks/norm/norm_fast.h5"
with h5py.File(NO, "r") as fo, h5py.File(NF, "r") as fn:
    heads = [h.decode() if isinstance(h, bytes) else h for h in fo["headers"][:]]
    eq = sum(np.array_equal(fo["embeddings"][h][:], fn["embeddings"][h][:]) for h in heads[:300])
    print(f"\nNORM fast vs original, first 300 proteins bitwise equal: {eq}/300")
