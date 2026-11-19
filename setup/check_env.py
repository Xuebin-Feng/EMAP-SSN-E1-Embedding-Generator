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

"""Read-only readiness check of the project's .venv, run by start.sh with .venv/bin/python.

Checks, without importing torch or touching the GPU:
  - Python is 3.12 and this interpreter is the project's .venv;
  - every package pinned in setup/requirements.txt and setup/flash_attn.txt is installed at
    exactly that version;
  - the E1 package resolves to third_party/E1/src (the bit-for-bit fork in this repository),
    not to another E1 install.
With --import it also imports torch, flash-attn and E1 (this initialises CUDA and fetches the
RMSNorm kernel from the Hugging Face Hub) and reports the GPU.

Exit status: 0 ready, 10 setup needed (start.sh then builds or repairs .venv).
"""
import argparse
import importlib.metadata as md
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN_FILES = [ROOT / "setup" / "requirements.txt", ROOT / "setup" / "flash_attn.txt"]
E1_SRC = ROOT / "third_party" / "E1" / "src"


def pins():
    for path in PIN_FILES:
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                name, _, version = line.partition("==")
                yield name.strip(), version.strip()


def installed_version(name):
    for candidate in {name, re.sub(r"[-_.]+", "-", name), re.sub(r"[-_.]+", "_", name)}:
        try:
            return md.version(candidate)
        except md.PackageNotFoundError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="also import torch, flash-attn and E1 (initialises CUDA)")
    args = ap.parse_args()
    problems = []

    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python is {sys.version.split()[0]}, not 3.12")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        problems.append(f"this interpreter is not the project's .venv ({sys.prefix})")

    for name, want in pins():
        have = installed_version(name)
        if have is None:
            problems.append(f"{name} is not installed (need {want})")
        elif have != want:
            problems.append(f"{name} is {have}, need {want}")

    # flash-attn's setup can produce a package without its compiled GPU module (when the
    # prebuilt wheel could not be downloaded and there was nothing to compile with).
    if installed_version("flash-attn") is not None and importlib.util.find_spec("flash_attn_2_cuda") is None:
        problems.append("flash-attn is installed without its CUDA module (flash_attn_2_cuda)")

    spec = importlib.util.find_spec("E1")
    if spec is None or spec.origin is None:
        problems.append("E1 is not installed (third_party/E1)")
    elif not Path(spec.origin).resolve().is_relative_to(E1_SRC.resolve()):
        problems.append(f"E1 resolves to {Path(spec.origin).parent}, not third_party/E1/src")

    if problems:
        print("Setup is needed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 10

    print(f"[OK] .venv ready: Python {sys.version.split()[0]}, {sum(1 for _ in pins())} pinned packages, "
          f"E1 from third_party/E1")
    if args.do_import:
        import os

        # The same RMSNorm warp count Generate_Embeddings_E1_fast.py uses, so the report
        # matches what the driver runs
        os.environ.setdefault("E1_RMSNORM_NUM_WARPS", "4")
        import torch
        import flash_attn  # noqa: F401
        import E1.modeling as e1m
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU visible"
        print(f"[OK] torch {torch.__version__}, flash-attn {flash_attn.__version__}, GPU: {gpu}")
        if e1m.layer_norm is None:
            print("[WARNING] The Triton RMSNorm kernel did not load: E1 uses PyTorch's RMSNorm, and "
                  "embeddings will differ from the published ones at bf16 rounding level.", file=sys.stderr)
        elif e1m.layer_norm.__name__ != "E1.model.triton_layer_norm":
            print(f"[WARNING] RMSNorm kernel loaded from {e1m.layer_norm.__name__}, not the bundled "
                  "E1.model.triton_layer_norm.", file=sys.stderr)
        else:
            print(f"[OK] RMSNorm kernel: bundled copy (E1.model.triton_layer_norm), num_warps "
                  f"{e1m.RMSNORM_NUM_WARPS or 'autotuned'} (E1_RMSNORM_NUM_WARPS; the driver's default is 4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
