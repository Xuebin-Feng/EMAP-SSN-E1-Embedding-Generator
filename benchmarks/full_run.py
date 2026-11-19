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

"""Full 12,927-query RA timing run of the fast driver (the timing reported for the EMAP-SSN paper).

Runs the real Generate_Embeddings_E1_fast.generate_embeddings() end to end, including
homolog selection on the GPU, and records:
  - hardware and software (CPU, RAM, GPU, driver, library versions, fork commit)
  - wall time per stage: startup (imports), FASTA + homolog selection, input preparation
    + HDF5 manifest, model load, embedding generation, total
  - peak RSS, peak GPU memory (PyTorch allocated / reserved; whole-GPU in the smi log)
  - comparison of all embeddings with the published file (bit-for-bit count; if not all
    identical, the size of the differences)
Output: benchmarks/full/<tag>_[E1_RA]_embeddings.h5 and benchmarks/full/<tag>_summary.json.
The E1 source is chosen by E1_SOURCE_DIR (default: third_party/E1/src).

Usage: python full_run.py --tag exact
"""
import time

T_START = time.perf_counter()
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
args = ap.parse_args()

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "Embeddings", "Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5")
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
import Generate_Embeddings_E1_fast as drv  # noqa: E402
import E1.modeling as e1m  # noqa: E402

os.makedirs("benchmarks/full", exist_ok=True)
out = os.path.abspath(f"benchmarks/full/{args.tag}_[E1_RA]_embeddings.h5")
assert out != os.path.abspath(REF)
if os.path.exists(out):
    os.remove(out)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({exc})"


def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return "n/a"


src_dir = os.path.dirname(os.path.dirname(os.path.dirname(e1m.__file__)))  # .../<worktree>
info = {
    "tag": args.tag,
    "started": datetime.now().isoformat(timespec="seconds"),
    "cpu": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
    "cpu_threads_visible": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "ram_total_GiB": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    "os": platform.platform(),
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader"),
    "python": platform.python_version(),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "flash_attn": version("flash_attn"), "triton": version("triton"),
    "transformers": version("transformers"), "h5py": h5py.__version__, "numpy": np.__version__,
    "e1_source": e1m.__file__,
    "e1_fork_commit": sh(f"git -C {src_dir} log -1 --format='%h %s'"),
    "e1_fork_branch": sh(f"git -C {src_dir} branch --show-current"),
    "driver_sha256": hashlib.sha256(open(drv.__file__, "rb").read()).hexdigest(),
    "homolog_selection_device": drv.DEVICE,
    "nice": os.nice(0),
}
print(json.dumps(info, indent=1), flush=True)

marks = {"t0": time.perf_counter()}


def selector():
    marks["sel_start"] = time.perf_counter()
    r = drv.extract_homologs()
    torch.cuda.synchronize()
    marks["sel_end"] = time.perf_counter()
    return r


def loader(checkpoint, device):
    marks["load_start"] = time.perf_counter()
    r = drv.load_e1_model(checkpoint, device)
    torch.cuda.synchronize()
    marks["load_end"] = time.perf_counter()
    return r


n = drv.generate_embeddings(drv.INPUT_FASTA, out, "RA", "float16", homolog_selector=selector,
                            model_loader=loader, device="cuda")
torch.cuda.synchronize()
marks["end"] = time.perf_counter()

stages = {
    "startup_imports_s": marks["t0"] - T_START,
    "fasta_and_homolog_selection_s": marks["sel_end"] - marks["t0"],
    "homolog_selection_only_s": marks["sel_end"] - marks["sel_start"],
    "input_preparation_and_manifest_s": marks["load_start"] - marks["sel_end"],
    "model_load_s": marks["load_end"] - marks["load_start"],
    "embedding_generation_s": marks["end"] - marks["load_end"],
    "total_s": marks["end"] - T_START,
}
stages = {k: round(v, 1) for k, v in stages.items()}
res = {"info": info, "generated": n, "stages": stages,
       "generation_s_per_query": round(stages["embedding_generation_s"] / n, 4),
       "peak_gpu_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "peak_gpu_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
       "peak_rss_GiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
       "finished": datetime.now().isoformat(timespec="seconds")}
try:  # RMSNorm warp count: fixed by E1_RMSNORM_NUM_WARPS, or what the autotuner chose
    import importlib  # noqa: E402
    _k = importlib.import_module(e1m.layer_norm.__name__ + ".layer_norm")._layer_norm_fwd_1pass_kernel
    res["rmsnorm_num_warps"] = {"fixed": getattr(e1m, "RMSNORM_NUM_WARPS", None),
                                "configs": [c.num_warps for c in _k.configs],
                                "autotuned": {str(k): c.num_warps for k, c in _k.cache.items()}}
except Exception as exc:  # noqa: BLE001
    res["rmsnorm_num_warps"] = f"n/a ({exc})"
print(json.dumps(res, indent=1), flush=True)

# Compare with the published embeddings (read-only)
exact, eq, n_el, sq_d, sq_v, maxabs, cos_min, cos_sum, n_res = 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0
pool_min = 1.0
with h5py.File(out, "r") as fo, h5py.File(REF, "r") as fr:
    heads = drv.decode_string_dataset(fr["headers"])
    res["published_hashes_equal"] = drv.decode_string_dataset(fo["input_hashes"]) == drv.decode_string_dataset(fr["input_hashes"])
    for h in heads:
        e, r = fo["embeddings"][h][:], fr["embeddings"][h][:]
        if np.array_equal(e, r):
            exact += 1
            eq += r.size
            n_el += r.size
            cos_sum += r.shape[0]
            n_res += r.shape[0]
            sq_v += float((r.astype(np.float64) ** 2).sum())
            continue
        a, b = e.astype(np.float64), r.astype(np.float64)
        eq += int((e == r).sum())
        n_el += r.size
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        cos_min, cos_sum, n_res = min(cos_min, float(c.min())), cos_sum + float(c.sum()), n_res + c.size
        pa, pb = a.mean(0), b.mean(0)
        pool_min = min(pool_min, float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb))))
        d = np.abs(a - b)
        maxabs = max(maxabs, float(d.max()))
        sq_d += float((d ** 2).sum())
        sq_v += float((b ** 2).sum())
res["vs_published"] = {
    "bitwise_identical_proteins": f"{exact}/{len(heads)}",
    "fp16_values_exactly_equal": eq / n_el,
    "residue_cosine_min": cos_min, "residue_cosine_mean": cos_sum / n_res,
    "protein_meanpool_cosine_min": pool_min,
    "abs_diff_max": maxabs, "rms_diff_over_rms_value": float(np.sqrt(sq_d / sq_v)),
}
with open(f"benchmarks/full/{args.tag}_summary.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res["vs_published"], indent=1))
