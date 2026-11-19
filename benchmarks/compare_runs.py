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

"""Where did one full run lose time against another? Both process the 12,927 queries in the
same order, so the time spent on the same range of queries is directly comparable. Reads the
tqdm progress lines (count, elapsed) of both logs in benchmarks/full/ and, for the second run,
its summary, to print the wall-clock time of each range.

Used on 18 Nov 2026 for the repeat run "exact2" against the clean run "exact": +32 s in all
(0.45 %), +9 s in one 5-minute window with other CPU load on the machine.
Usage: python compare_runs.py [reference tag] [compared tag]   (default: exact exact2)
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full")
REF_TAG, TAG = (sys.argv[1:3] + ["exact", "exact2"][len(sys.argv[1:3]):])[:2]


def progress(tag):
    text = open(os.path.join(D, f"{tag}.log"), errors="replace").read()
    pts = {}
    for m in re.finditer(r"Batches:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<", text):
        n = int(m.group(1))
        s = int(m.group(3) or 0) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        pts.setdefault(n, s)  # first time each count was reached
    return sorted(pts.items())


def at(pts, n):  # elapsed seconds when count n was reached (linear interpolation)
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if n0 <= n <= n1:
            return s0 + (s1 - s0) * (n - n0) / max(n1 - n0, 1)
    return pts[-1][1]


a, b = progress(REF_TAG), progress(TAG)
total = min(a[-1][0], b[-1][0])
summary = json.load(open(os.path.join(D, f"{TAG}_summary.json")))
st = summary["stages"]
gen_start = datetime.fromisoformat(summary["info"]["started"]) + timedelta(
    seconds=st["startup_imports_s"] + st["fasta_and_homolog_selection_s"]
    + st["input_preparation_and_manifest_s"] + st["model_load_s"])
print(f"{REF_TAG} vs {TAG}: {total} queries; {TAG} generation started ~{gen_start:%H:%M:%S}")
print(f"{'queries':>13} {TAG + ' clock':>17} {REF_TAG + ' s':>9} {TAG + ' s':>9} {'ratio':>6}")
extra = 0.0
for lo in range(0, total, 500):
    hi = min(lo + 500, total)
    da, db = at(a, hi) - at(a, lo), at(b, hi) - at(b, lo)
    extra += db - da
    t0, t1 = (gen_start + timedelta(seconds=at(b, x)) for x in (lo, hi))
    print(f"{lo:>6}-{hi:<6} {t0:%H:%M:%S}-{t1:%H:%M:%S} {da:9.0f} {db:9.0f} {db / da:6.3f}")
print(f"{TAG} minus {REF_TAG} over all ranges: {extra:+.0f} s (tqdm, 1 s resolution)")
