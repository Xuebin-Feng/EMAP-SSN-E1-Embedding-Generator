#!/usr/bin/env bash
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

# One full timing run (12,927 RA queries) of benchmarks/full_run.py, with a GPU log every 30 s
# in benchmarks/full/smi_<tag>.csv: utilisation, memory, power, SM clock, temperature and the
# load average. Under WSL it also logs the Windows GPU process list (nvidia-smi.exe, since
# WSL's nvidia-smi cannot list processes) and, with benchmarks/winload.ps1, the Windows CPU
# load (benchmarks/full/winload_<tag>.csv). benchmarks/check_load.py reads these logs to tell
# whether other machine load could have affected the timing. On a shared GPU, set
# GPU_HOLD_FILE to your hold file to log who holds it.
#
# Run in the project environment on an otherwise idle machine:
#   ./start.sh bash benchmarks/full_run.sh <tag> [E1 src dir, default third_party/E1/src]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
tag=$1
src=${2:-$PWD/third_party/E1/src}
mkdir -p benchmarks/full
WIN_SMI=/mnt/c/Windows/System32/nvidia-smi.exe
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe

(
  echo "time;util_mem_power_clock_temp;loadavg_1_5_15;hold;windows_gpu_processes"
  while true; do
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader)
    p=""
    if [ -x "$WIN_SMI" ]; then
      p=$("$WIN_SMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
          | sed 's/.*\\//' | tr -d '\r' | sort | tr '\n' '|')
    fi
    h=""
    if [ -n "${GPU_HOLD_FILE:-}" ]; then
      h=$(head -c 60 "$GPU_HOLD_FILE" 2>/dev/null | tr -d '\n')
    fi
    echo "$(date '+%F %T');$g;$(cut -d' ' -f1-3 /proc/loadavg);$h;$p"
    sleep 30
  done
) > "benchmarks/full/smi_${tag}.csv" &
SMI=$!

if [ -x "$PS" ]; then  # WSL: Windows CPU load (total + top processes), one windowless process
  rm -f "benchmarks/full/winload_${tag}.stop"
  "$PS" -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w benchmarks/winload.ps1)" \
    -Out "$(wslpath -w "$PWD/benchmarks/full")\\winload_${tag}.csv" \
    -StopFile "$(wslpath -w "$PWD/benchmarks/full")\\winload_${tag}.stop" -Every 30 > /dev/null 2>&1 &
fi

echo "$(date '+%F %T') full run $tag start (E1 source $src, RMSNorm num_warps ${E1_RMSNORM_NUM_WARPS:-4, the driver default})"
# nice 10 and 4 CPU threads, as in the reported runs
E1_SOURCE_DIR="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /usr/bin/time -v nice -n 10 \
  python benchmarks/full_run.py --tag "$tag" > "benchmarks/full/${tag}.log" 2>&1
rc=$?
kill $SMI
if [ -x "$PS" ]; then
  touch "benchmarks/full/winload_${tag}.stop"
fi
echo "$(date '+%F %T') full run $tag end (exit $rc)"
exit $rc
