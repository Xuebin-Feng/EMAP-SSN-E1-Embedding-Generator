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

"""Run the new deterministic homolog selection of Generate_Embeddings_E1_fast.py
on one device / thread count and record what it selected.

Writes benchmarks/selection/torch_<device>_t<threads>.json with
  - sha256 of the normalised score vector (float32 bytes), i.e. the arithmetic
  - the homolog map (query header -> 50 homolog headers)
  - how many published input hashes it reproduces
Read-only with respect to the network and the published embeddings.

Usage: python selection_determinism.py --device cpu --threads 4
"""
import argparse
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1_fast as drv  # noqa: E402

drv.DEVICE = args.device

# 1. The arithmetic: the normalised scores, exactly as extract_homologs computes them
with h5py.File(drv.INPUT_NETWORK_H5, "r") as hf:
    s = torch.tensor(hf["g_score"][:].astype(np.float32), device=args.device)
    ln = torch.tensor(hf["g_len"][:].astype(np.float32), device=args.device)
norm = (s / (ln + 1e-6)).cpu().numpy()
score_sha = hashlib.sha256(norm.tobytes()).hexdigest()
n_zero, n_negzero = int((norm == 0).sum()), int(((norm == 0) & np.signbit(norm)).sum())
del s, ln, norm

# 2. The selection, through the driver function itself
t = time.perf_counter()
headers, homolog_map = drv.extract_homologs()
t_sel = time.perf_counter() - t

# 3. Compare with the published contexts (input hashes)
fasta_headers, fasta_sequences, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
_, _, _, hashes = drv.prepare_e1_inputs(fasta_headers, fasta_sequences, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
                                        target_headers=headers, homolog_map=homolog_map)
with h5py.File(REF, "r") as hf:
    ref = dict(zip(drv.decode_string_dataset(hf["headers"]), drv.decode_string_dataset(hf["input_hashes"])))
matches = sum(ref[h] == x for h, x in zip(headers, hashes))
map_sha = hashlib.sha256(json.dumps(homolog_map, sort_keys=True).encode()).hexdigest()

out = {"impl": "torch", "torch": torch.__version__, "device": args.device, "threads": args.threads,
       "score_sha256": score_sha, "zero_scores": n_zero, "negative_zero_scores": n_negzero,
       "selection_seconds": round(t_sel, 1), "published_hash_matches": matches, "n": len(headers),
       "homolog_map_sha256": map_sha, "homolog_map": homolog_map}
os.makedirs(os.path.join(PROJECT, "benchmarks", "selection"), exist_ok=True)
path = os.path.join(PROJECT, "benchmarks", "selection", f"torch_{args.device}_t{args.threads}.json")
with open(path, "w") as fh:
    json.dump(out, fh)
print({k: v for k, v in out.items() if k != "homolog_map"})
