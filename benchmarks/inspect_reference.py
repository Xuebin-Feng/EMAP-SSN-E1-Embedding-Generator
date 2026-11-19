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

"""Read-only checks of the published E1_RA embedding file and of the driver's
homolog selection, run on CPU (4 threads) so the busy GPU is not touched.

- Prints the reference file's attributes, dataset layout and storage options.
- Re-runs the driver's extract_homologs() on CPU, rebuilds the input hashes
  and compares them with the hashes stored in the reference file.
- Reports the token count per query (homologs + query), which sets the
  forward-pass cost.
"""
import os
import resource
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

drv.DEVICE = "cpu"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


print("== Reference file (opened read-only) ==")
with h5py.File(REF, "r") as hf:
    for k, v in hf.attrs.items():
        print(f"  attr {k} = {v!r}")
    print("  top-level keys:", list(hf.keys()))
    ref_headers = drv.decode_string_dataset(hf["headers"])
    ref_sequences = drv.decode_string_dataset(hf["sequences"])
    ref_hashes = drv.decode_string_dataset(hf["input_hashes"])
    emb = hf["embeddings"]
    print("  n embeddings:", len(emb))
    d0 = emb[ref_headers[0]]
    print(f"  first dataset: shape={d0.shape} dtype={d0.dtype} chunks={d0.chunks} "
          f"compression={d0.compression} layout={'contiguous' if d0.chunks is None else 'chunked'}")
    print("  file size GB:", os.path.getsize(REF) / 1e9)

lens = np.array([len(s) for s in ref_sequences])
print(f"  query lengths: n={len(lens)} mean={lens.mean():.1f} median={np.median(lens):.0f} "
      f"p95={np.percentile(lens, 95):.0f} max={lens.max()} min={lens.min()}")

print("\n== Sanitized FASTA ==")
t = time.perf_counter()
fasta_headers, fasta_sequences, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
print(f"  {len(fasta_headers)} records in {time.perf_counter() - t:.2f} s")
same_manifest = fasta_headers == ref_headers and fasta_sequences == ref_sequences
print("  FASTA manifest identical to reference (order + sequences):", same_manifest)

print("\n== Homolog selection on CPU (driver's extract_homologs) ==")
t = time.perf_counter()
with h5py.File(drv.INPUT_NETWORK_H5, "r") as hf:
    n_edges = hf["i"].shape[0]
    _ = hf["i"][:], hf["j"][:], hf["g_score"][:], hf["g_len"][:]
    print("  network datasets:", {k: (hf[k].dtype.str, hf[k].shape, hf[k].compression) for k in hf.keys()})
t_read = time.perf_counter() - t
del _
print(f"  raw read of i/j/g_score/g_len ({n_edges:,} edges): {t_read:.1f} s")

t = time.perf_counter()
target_headers, homolog_map = drv.extract_homologs()
t_sel = time.perf_counter() - t
print(f"  extract_homologs() total on CPU: {t_sel:.1f} s, peak RSS so far {rss_gb():.2f} GB")

t = time.perf_counter()
out_headers, out_sequences, input_strings, input_hashes = drv.prepare_e1_inputs(
    fasta_headers, fasta_sequences, "RA", drv.E1_MODEL_MAPPINGS["E1_RA"],
    target_headers=target_headers, homolog_map=homolog_map,
)
print(f"  prepare_e1_inputs: {time.perf_counter() - t:.2f} s")
print("  target order identical to reference:", out_headers == ref_headers)
ref_hash_by_header = dict(zip(ref_headers, ref_hashes))
matches = sum(ref_hash_by_header.get(h) == x for h, x in zip(out_headers, input_hashes))
print(f"  input hashes matching reference: {matches} / {len(out_headers)}")
if matches != len(out_headers):
    bad = [h for h, x in zip(out_headers, input_hashes) if ref_hash_by_header.get(h) != x]
    print("  first mismatching headers:", bad[:5])

tokens = np.array([sum(len(s) + 4 for s in x.split(",")) for x in input_strings])
nseq = np.array([x.count(",") + 1 for x in input_strings])
print(f"\n== Tokens per forward pass ==\n  sequences per input: min={nseq.min()} max={nseq.max()}")
print(f"  tokens: mean={tokens.mean():.0f} median={np.median(tokens):.0f} p95={np.percentile(tokens, 95):.0f} "
      f"max={tokens.max()} total={tokens.sum():,}")
np.save(os.path.join(PROJECT, "benchmarks", "tokens_per_query.npy"), tokens)
print(f"\npeak RSS {rss_gb():.2f} GB")
