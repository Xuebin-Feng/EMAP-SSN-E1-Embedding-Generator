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

"""Was a timing run contaminated by other machine load? Reads the 30 s logs written by
full_run.sh (benchmarks/full/smi_<tag>.csv, winload_<tag>.csv) and reports:
  - GPU: utilisation, memory range, samples with memory well above the run's own level,
    Windows GPU processes that appeared after the run started, and changes of the GPU hold
    file's content if the run logged one (GPU_HOLD_FILE in full_run.sh)
  - CPU (WSL only, from winload_<tag>.csv): Windows total CPU and the busiest non-WSL
    processes; WSL load average
Flags 'sustained heavy CPU load' when, for >= 3 consecutive samples (1.5 min), a non-WSL
process uses >= 25 % of all cores or total CPU is >= 60 %.
Usage: python check_load.py <tag>
"""
import os
import json
import sys
from collections import Counter

tag = sys.argv[1]
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = f"{PROJECT}/benchmarks/full/{tag}"
smi = [ln.rstrip("\n").split(";") for ln in open(f"{PROJECT}/benchmarks/full/smi_{tag}.csv")][1:]
win_path = f"{PROJECT}/benchmarks/full/winload_{tag}.csv"  # written only under WSL
win = ([ln.lstrip("﻿").rstrip("\n").split(";") for ln in open(win_path, encoding="utf-8-sig")][1:]
       if os.path.exists(win_path) else [])

rows = []
for t, g, load, hold, procs in (r + [""] * (5 - len(r)) for r in smi):
    util, mem = (float(x.split()[0]) for x in g.split(",")[:2])
    rows.append({"t": t, "util": util, "mem": mem, "load1": float(load.split()[0]) if load else None,
                 "hold": hold, "procs": set(p for p in procs.split("|") if p)})
run = rows[2:-1] if len(rows) > 4 else rows  # skip start-up (model load) and the last partial sample
mems = sorted(r["mem"] for r in run)
steady = mems[len(mems) // 2]
first_procs = rows[0]["procs"] if rows else set()
new_procs = Counter(p for r in rows for p in r["procs"] - first_procs)
gpu = {
    "samples": len(rows),
    "util_mean": round(sum(r["util"] for r in run) / len(run), 1),
    "util_min": min(r["util"] for r in run),
    "mem_MiB_min_median_max": [mems[0], steady, mems[-1]],
    "samples_mem_above_median_plus_1GiB": sum(r["mem"] > steady + 1024 for r in run),
    "gpu_processes_new_during_run": dict(new_procs),
    # The hold file logged at the start belongs to this run; a different content means
    # someone else took the GPU (always 0 when no hold file was logged)
    "samples_hold_changed": sum(r["hold"] != rows[0]["hold"] for r in rows) if rows else 0,
    "wsl_load1_max": max((r["load1"] for r in rows if r["load1"] is not None), default=None),
}

ours = ("vmmemwsl", "vmmem", "idle", "_total")
heavy_streak, worst, total_streak = Counter(), Counter(), 0
max_total_streak, busiest, totals = 0, Counter(), []
for r in win:
    if len(r) < 3:
        continue
    total = float(r[1].replace(",", "."))
    totals.append(total)
    total_streak = total_streak + 1 if total >= 60 else 0
    max_total_streak = max(max_total_streak, total_streak)
    seen = set()
    for item in filter(None, r[2].split("|")):
        name, pct = item.rsplit("=", 1)
        pct = float(pct.replace(",", "."))
        if name.lower().startswith(ours):
            continue
        busiest[name] = max(busiest[name], pct)
        if pct >= 25:
            heavy_streak[name] += 1
            worst[name] = max(worst[name], heavy_streak[name])
            seen.add(name)
    for name in list(heavy_streak):
        if name not in seen:
            heavy_streak[name] = 0
cpu = {
    "samples": len(totals),
    "windows_total_cpu_mean_max": [round(sum(totals) / len(totals), 1), max(totals)] if totals else None,
    "busiest_non_wsl_processes_max_pct_of_cores": dict(busiest.most_common(6)),
    "longest_run_total_cpu_ge_60pct_samples": max_total_streak,
    "longest_run_process_ge_25pct_samples": dict(worst),
}
contaminated = (bool(new_procs) or gpu["samples_mem_above_median_plus_1GiB"] > 2 or gpu["samples_hold_changed"] > 0
                or max_total_streak >= 3 or any(v >= 3 for v in worst.values()))
res = {"tag": tag, "gpu": gpu, "cpu": cpu, "flag_contaminated": contaminated}
json.dump(res, open(f"{base}_load_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
