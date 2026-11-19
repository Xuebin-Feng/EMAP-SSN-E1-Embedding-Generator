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

"""Bit-for-bit check of the project's .venv (made by start.sh) against the published embeddings.

Run with .venv/bin/python (e.g. ./start.sh python benchmarks/verify_venv.py) on an idle GPU.
Same 64 RA queries as the phase-1 ladder (random.Random(42).sample of the homolog map, after
3 warm-up queries), through the fast driver's real generate_embeddings() with the code the
repository ships: third_party/E1, the bundled RMSNorm kernel, num_warps 4, bf16 switches off.
Writes benchmarks/verify/venv_check.json (the published file is opened read-only).
"""
import json
import os
import random
import sys
import time

import h5py
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
OUT = os.path.join(PROJECT, "benchmarks", "verify")
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
assert os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(PROJECT, ".venv")), sys.prefix
import Generate_Embeddings_E1_fast as drv  # noqa: E402
import E1.modeling as e1m  # noqa: E402
import flash_attn  # noqa: E402
import triton  # noqa: E402
from E1.model import bf16_options  # noqa: E402

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(PROJECT, "benchmarks", "homolog_map.json")) as fh:
    homolog_map = json.load(fh)["homolog_map"]
sample = random.Random(42).sample(list(homolog_map), 67)
warm, sample = sample[:3], sample[3:]
with h5py.File(REF, "r") as ref:
    ref_emb = {h: ref["embeddings"][h][:] for h in sample}

t = time.perf_counter()
model, bp = drv.load_e1_model(drv.E1_MODEL_MAPPINGS["E1_RA"], "cuda")
load_s = time.perf_counter() - t


def run(name, headers):
    path = os.path.join(OUT, f"{name}.h5")
    if os.path.exists(path):
        os.remove(path)
    marks = {}

    def loader(checkpoint, device):
        torch.cuda.synchronize()
        marks["start"] = time.perf_counter()
        return model, bp

    drv.generate_embeddings(drv.INPUT_FASTA, path, "RA", "float16", device="cuda",
                            homolog_selector=lambda: (headers, homolog_map), model_loader=loader)
    torch.cuda.synchronize()
    return path, time.perf_counter() - marks["start"]


run("venv_warmup", warm)
path, secs = run("venv_check", sample)
with h5py.File(path, "r") as fo:
    same = sum(np.array_equal(fo["embeddings"][h][:], ref_emb[h]) for h in sample)
res = {
    "python": sys.version.split()[0], "prefix": sys.prefix,
    "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
    "flash_attn": flash_attn.__version__, "gpu": torch.cuda.get_device_name(0),
    "e1_source": e1m.__file__,
    "rmsnorm_kernel": getattr(e1m.layer_norm, "__name__", None),
    "rmsnorm_kernel_file": getattr(e1m.layer_norm, "__file__", None),
    "rmsnorm_num_warps": e1m.RMSNORM_NUM_WARPS,
    "bf16_switches": [bf16_options.MERGED_GEMMS, bf16_options.FUSED_ADD_NORM, bf16_options.QUERY_ONLY_LAST_LAYER],
    "kernels_package_imported": "kernels" in sys.modules,
    "model_load_s": round(load_s, 1),
    "queries": len(sample), "s_per_query": round(secs / len(sample), 4),
    "bitwise_identical_to_published": f"{same}/{len(sample)}",
}
with open(os.path.join(OUT, "venv_check.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if same == len(sample) else 1)
