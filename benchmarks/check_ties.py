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

"""Why do 221 CPU-selected homolog contexts differ from the published ones?

Hypothesis: equal similarity scores (ties) at the selected ranks (20, 40, ..., 1000),
which torch.topk orders differently on CPU and GPU.
Rebuilds the driver's similarity matrix on CPU (same arithmetic), then
  1. counts rows whose selected ranks touch a tie, for matching vs mismatching rows;
  2. tries two deterministic tie rules (ascending / descending column index) and
     reports how many published input hashes each reproduces.
Writes the homolog map that reproduces the most hashes to benchmarks/homolog_map.json.
Read-only with respect to the network and reference files.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
torch.set_num_threads(4)
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402

with h5py.File(REF, "r") as hf:
    ref_headers = drv.decode_string_dataset(hf["headers"])
    ref_hashes = drv.decode_string_dataset(hf["input_hashes"])
fasta_headers, fasta_sequences, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
seq_by_h = dict(zip(fasta_headers, fasta_sequences))
ckpt = drv.E1_MODEL_MAPPINGS["E1_RA"]

t = time.perf_counter()
with h5py.File(drv.INPUT_NETWORK_H5, "r") as hf:
    headers = [h.decode() if isinstance(h, bytes) else h for h in hf["headers"][:]]
    i = torch.from_numpy(hf["i"][:].astype(np.int64))
    j = torch.from_numpy(hf["j"][:].astype(np.int64))
    s = torch.from_numpy(hf["g_score"][:].astype(np.float32))
    ln = torch.from_numpy(hf["g_len"][:].astype(np.float32))
n = len(headers)
assert headers == ref_headers
norm = s / (ln + 1e-6)
S = torch.full((n, n), float("-inf"))
S[i, j] = norm
S[j, i] = norm
S.fill_diagonal_(float("-inf"))
del i, j, s, ln, norm
print(f"matrix built in {time.perf_counter() - t:.1f} s")

ranks = torch.arange(19, 1000, 20)


def hashes_for(sel):
    out = []
    for q in range(n):
        hom = [seq_by_h[headers[k]] for k in sel[q].tolist()]
        out.append(drv.build_input_hash(ckpt, "RA", hom, seq_by_h[headers[q]]))
    return out


results = {}
# torch.topk on CPU (what the driver did when run on CPU)
_, top = torch.topk(S, k=1000, dim=1)
results["topk_cpu"] = top[:, ranks]
del top
# Stable sort: ties by ascending column index
vals, idx = torch.sort(S, dim=1, descending=True, stable=True)
vals, idx = vals[:, :1001].contiguous(), idx[:, :1000].contiguous()
results["ties_ascending_index"] = idx[:, ranks]
# tie diagnostics: a selected rank r is ambiguous if vals[r] equals vals[r-1] or vals[r+1]
amb = ((vals[:, ranks] == vals[:, ranks - 1]) | (vals[:, ranks] == vals[:, ranks + 1])).any(1).numpy()
del idx
# Stable sort on reversed columns: ties by descending column index
_, idx = torch.sort(S.flip(1), dim=1, descending=True, stable=True)
results["ties_descending_index"] = (n - 1 - idx[:, :1000])[:, ranks]
del idx, S

ref_by_h = dict(zip(ref_headers, ref_hashes))
best = None
for name, sel in results.items():
    hs = hashes_for(sel)
    ok = np.array([ref_by_h[headers[q]] == hs[q] for q in range(n)])
    print(f"{name:24s}: {ok.sum()} / {n} hashes match")
    if name == "topk_cpu":
        print(f"   rows with a tie at a selected rank: {amb.sum()} "
              f"(among mismatching rows: {amb[~ok].sum()} / {(~ok).sum()}; "
              f"among matching rows: {amb[ok].sum()} / {ok.sum()})")
    if best is None or ok.sum() > best[1]:
        best = (name, ok.sum(), sel)

name, nok, sel = best
homolog_map = {headers[q]: [headers[k] for k in sel[q].tolist()] for q in range(n)}
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json"), "w") as fh:
    json.dump({"rule": name, "hash_matches": int(nok), "homolog_map": homolog_map}, fh)
print(f"saved homolog_map.json from rule '{name}' ({nok} matches)")
