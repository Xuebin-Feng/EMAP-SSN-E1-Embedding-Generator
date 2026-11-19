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

"""Compare every homolog selection in benchmarks/selection/ (torch runs: json maps;
numpy runs: npz index arrays) with each other and with the published contexts."""
import glob
import json
import os
import sys

import h5py
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402  (original driver: hashing only)

with h5py.File(drv.INPUT_NETWORK_H5, "r") as hf:
    net_headers = [h.decode() if isinstance(h, bytes) else h for h in hf["headers"][:]]
with h5py.File(os.path.join("Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5"), "r") as hf:
    ref = dict(zip(drv.decode_string_dataset(hf["headers"]), drv.decode_string_dataset(hf["input_hashes"])))
fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
seq = dict(zip(fh_, fs_))
ckpt = drv.E1_MODEL_MAPPINGS["E1_RA"]

runs = {}
for p in sorted(glob.glob("benchmarks/selection/torch_*.json")):
    d = json.load(open(p))
    runs[os.path.basename(p)[:-5]] = (d["homolog_map"], d["score_sha256"], f"torch {d['torch']} {d['device']} threads={d['threads']}")
for p in sorted(glob.glob("benchmarks/selection/numpy_*.npz")):
    d = np.load(p, allow_pickle=True)
    hs = list(d["headers"])
    assert hs == net_headers
    m = {hs[q]: [hs[k] for k in row] for q, row in enumerate(d["selected"].tolist())}
    runs[os.path.basename(p)[:-4]] = (m, str(d["score_sha256"]), f"{d['platform']} numpy {d['numpy']} python {d['python']}")

base_name, (base_map, _, _) = next(iter(runs.items()))
for name, (m, sha, desc) in runs.items():
    same = sum(m[h] == base_map[h] for h in net_headers)
    ok = sum(ref[h] == drv.build_input_hash(ckpt, "RA", [seq[x] for x in m[h]], seq[h]) for h in net_headers)
    print(f"{name:22s} {desc}\n    score sha256 {sha[:16]}  identical to {base_name}: {same}/{len(net_headers)}  "
          f"published contexts reproduced: {ok}/{len(net_headers)}")
