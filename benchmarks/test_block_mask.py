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

"""CPU test: the fork's create_block_causal_mask_from_block_bounds must give a BlockMask
whose tensors are identical to the original create_block_causal_mask_optimized."""
import json
import os
import sys

import torch

torch.set_num_threads(4)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "third_party", "E1", "src"))
os.chdir(PROJECT)
import E1.model.flex_attention as fx  # noqa: E402  (the fork)

assert fx.__file__.startswith(os.path.join(PROJECT, "third_party", "E1")), fx.__file__
sys.path.insert(1, PROJECT)
import Generate_Embeddings_E1 as drv  # noqa: E402

FIELDS = ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
          "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"]


def ids_from_lengths(rows):
    return torch.stack([torch.repeat_interleave(torch.arange(len(r)), torch.tensor(r)) for r in rows])


def same(rows):
    sid = ids_from_lengths(rows)
    a = fx.create_block_causal_mask_optimized(sid)
    b = fx.create_block_causal_mask_from_block_bounds(sid)
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if (x is None) != (y is None) or (x is not None and (x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y))):
            return f"{f} differs"
    if a.BLOCK_SIZE != b.BLOCK_SIZE or a.seq_lengths != b.seq_lengths:
        return "BLOCK_SIZE/seq_lengths differ"
    return "ok"


fh_, fs_, _ = drv.load_sanitized_fasta(drv.INPUT_FASTA)
L = {h: len(s) + 4 for h, s in zip(fh_, fs_)}
hmap = json.load(open("benchmarks/homolog_map.json"))["homolog_map"]
tok = {q: sum(L[x] for x in hmap[q] + [q]) for q in hmap}
qs = sorted(hmap, key=tok.get)
print(f"RA inputs have {tok[qs[0]]}-{tok[qs[-1]]} tokens; the dense N x N original needs ~8 N^2 bytes per "
      "temporary on CPU, so real layouts are tested as prefixes of <= 9000 tokens (full lengths on GPU)")
pick = [qs[0], qs[-1]] + [qs[i] for i in range(0, len(qs), len(qs) // 22)][:22]


def prefix(lengths, cap=9000):
    out = []
    for x in lengths:
        if sum(out) + x > cap:
            break
        out.append(x)
    return out


cases = {f"RA prefix {q[:24]} ({sum(prefix([L[x] for x in hmap[q] + [q]]))} tok)": [prefix([L[x] for x in hmap[q] + [q]])]
         for q in pick}
cases.update({
    "NORM single seq 300": [[300]], "NORM single seq 128": [[128]], "NORM single seq 77": [[77]],
    "NORM single seq 554": [[554]], "exact multiples 128+256": [[128, 256]], "tiny seqs": [[3, 5, 1, 7, 2, 130]],
    "batch of 2, no padding": [[100, 156, 40], [60, 236]],
})
bad = 0
for name, rows in cases.items():
    r = same(rows)
    bad += r != "ok"
    print(f"{r:24s} {name}")
print(f"\n{len(cases) - bad}/{len(cases)} cases identical")
