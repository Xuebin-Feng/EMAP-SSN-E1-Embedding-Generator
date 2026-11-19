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

# Windows-side machine load during a timing run: every $Every seconds, total CPU % and the
# top processes by current CPU (% of all logical cores). WSL shows up as VmmemWSL.
# Stops when $StopFile exists. One long-lived process, so no PowerShell start-up per sample.
param([string]$Out, [string]$StopFile, [int]$Every = 30)
$cores = [Environment]::ProcessorCount
"time;total_cpu_pct;top_processes_pct_of_all_cores" | Out-File -FilePath $Out -Encoding utf8
while (-not (Test-Path $StopFile)) {
    try {
        $c = Get-Counter -Counter '\Processor(_Total)\% Processor Time', '\Process(*)\% Processor Time' -ErrorAction SilentlyContinue
        $total = ($c.CounterSamples | Where-Object { $_.Path -like '*\processor(_total)\*' }).CookedValue
        $top = $c.CounterSamples |
            Where-Object { $_.Path -like '*\process(*' -and $_.InstanceName -notin @('_total', 'idle') } |
            Sort-Object CookedValue -Descending | Select-Object -First 6 |
            ForEach-Object { '{0}={1:N1}' -f $_.InstanceName, ($_.CookedValue / $cores) }
        ('{0};{1:N1};{2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $total, ($top -join '|')) |
            Out-File -FilePath $Out -Append -Encoding utf8
    } catch { }
    Start-Sleep -Seconds $Every
}
