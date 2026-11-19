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

"""Independent re-implementation of the deterministic homolog selection with numpy
only (no torch, no driver import), so it can run on another OS / Python build.

Rule: score = g_score / (g_len + 1e-6) in float32; rank each row by descending
score with ties by ascending column index (stable sort); keep ranks 20, 40, ..., 1000.
Writes <out_dir>/numpy_<tag>.npz with the selected column indices (12927 x 50),
the network header order and the sha256 of the score vector.

Usage: python select_numpy.py <network.h5> <out_dir> <tag>
"""
import hashlib
import platform
import sys
import time

import h5py
import numpy as np

net, out_dir, tag = sys.argv[1:4]
t = time.perf_counter()
with h5py.File(net, "r") as hf:
    headers = [h.decode() if isinstance(h, bytes) else h for h in hf["headers"][:]]
    i = hf["i"][:].astype(np.int64)
    j = hf["j"][:].astype(np.int64)
    s = hf["g_score"][:].astype(np.float32)
    ln = hf["g_len"][:].astype(np.float32)
norm = s / (ln + np.float32(1e-6))
assert norm.dtype == np.float32
score_sha = hashlib.sha256(norm.tobytes()).hexdigest()
n = len(headers)
S = np.full((n, n), -np.inf, dtype=np.float32)
S[i, j] = norm
S[j, i] = norm
np.fill_diagonal(S, -np.inf)
del i, j, s, ln, norm

ranks = np.arange(19, 1000, 20)
sel = np.empty((n, len(ranks)), dtype=np.int32)
for r0 in range(0, n, 512):
    order = np.argsort(-S[r0:r0 + 512], axis=1, kind="stable")
    sel[r0:r0 + 512] = order[:, ranks]
np.savez(f"{out_dir}/numpy_{tag}.npz", selected=sel, headers=np.array(headers, dtype=object),
         score_sha256=score_sha, platform=platform.platform(), numpy=np.__version__,
         python=sys.version.split()[0])
print(f"{tag}: {platform.platform()} numpy {np.__version__} python {sys.version.split()[0]}; "
      f"score sha256 {score_sha[:16]}...; {time.perf_counter() - t:.0f} s")
