# EMAP-SSN-E1-Embedding-Generator

**This project builds upon Profluent-E1 under the terms of the Profluent-E1 Clickthrough License Agreement.**

This project generates per-residue Profluent-E1 (E1-600m) embeddings of a protein sequence set for
EMAP-SSN, in single-sequence (NORM) or retrieval-augmented (RA) mode. In RA mode, homologs are
selected from an ESMC-based sequence similarity network. The repository also contains the scripts
used to profile, speed up and verify that generation.

It is not affiliated with or endorsed by Profluent Bio.

## Quick start

```bash
./start.sh
```

The first run creates `.venv` in this folder with [uv](https://docs.astral.sh/uv/) and installs
the pinned packages: about 4 GB of downloads. If uv is missing, it is installed into
`~/.local/bin`; your shell start-up files are not changed. `start.sh` then opens a shell in this
folder with the environment active, so you can run the drivers as usual:

```bash
python Generate_Embeddings_E1_fast.py --data-dir ~/my_data    # the configuration at the top of the file
python Generate_Embeddings_E1_fast.py --data-dir ~/my_data --sequence-set MySet --mode NORM
python Generate_Embeddings_E1_fast.py --help                  # all options and their defaults
```

**Where the data goes.** The data is not part of this repository. The driver reads and writes
it in a data folder:
- `<data dir>/Sequence_Sets/<set>.fasta`;
- `<data dir>/Networks_EValues/<set>_[<network model>]_network.h5` (RA mode);
- `<data dir>/Embeddings/<set>_[E1_<mode>]_embeddings_speedup.h5`, the output.

The data folder is `--data-dir`, else the `E1_DATA_DIR` environment variable, else the current
folder. After `export E1_DATA_DIR=~/my_data`, plain `python Generate_Embeddings_E1_fast.py` works.
`--fasta`, `--network` and `--out` name single files directly.

Every other option defaults to the `CONFIGURATION` block at the top of the driver:
- input and output: `--sequence-set`, `--network-model`;
- generation: `--mode`, `--precision`, `--bf16-speedups`;
- homolog selection: `--max-homologs`, `--top-k`, `--stride`, …

An unfinished output file is resumed. Later runs of `start.sh` only check the environment,
which takes about a second, and open the shell.

| Command | What it does |
| --- | --- |
| `./start.sh` | Checks `.venv`, builds or repairs it if needed, then opens a shell in it, in this folder. |
| `./start.sh <command> ...` | The same, but runs the command in the environment instead of opening a shell. The command runs in the folder you called `start.sh` from, so relative paths in it are relative to that folder. |
| `./start.sh --check-only` | Only reports: exit 0 if ready, 10 if setup is needed. With `--import`, it also imports torch, flash-attn and E1, which uses the GPU. |
| `./start.sh --setup-only` | Builds or repairs `.venv`, then exits. |

**Requirements and limitations:**
- **Operating system: Linux on x86_64, or WSL2 on Windows.** Tested on WSL2 (Ubuntu) on
  Windows 11. `start.sh` is a bash script and the pinned packages are Linux x86_64 CUDA
  builds, so it does not install on macOS or on Windows outside WSL2. ARM Linux is untested.
- **GPU: an NVIDIA GPU of the Ampere generation or newer** (compute capability 8.0 or higher,
  for example RTX 30 and 40 series, A100, L40, H100), with a driver for CUDA 12.8.
  flash-attn 2 supports no older GPUs.
  - Tested on an RTX 4070 (12 GB); a full RA run used at most 3.8 GiB of GPU memory.
  - Upstream E1 recommends such a GPU, for bf16 and flash attention, and also runs on the
    CPU. This repository needs the GPU: `start.sh` always installs flash-attn, and E1 then
    uses it and the Triton RMSNorm kernel without checking where the tensors are.
  - **`--device cpu` fails on a machine with an NVIDIA GPU** (the Triton RMSNorm kernel gets
    CPU tensors).
  - **Without an NVIDIA GPU,** setup and `./start.sh --check-only --import` succeed, but the
    first embedding fails in flash-attn.
    - With `USE_FLASH_ATTN=0`, single-sequence (NORM) mode runs on the CPU, at about 2.5 s
      per sequence of about 300 residues on 4 threads.
    - RA mode does not run on the CPU in practice: there, the first query needed more than
      14 GB of RAM.
    - CPU results are not bit-for-bit identical to the published GPU embeddings.
- **Memory and disk:**
  - peak RAM 5.3 GiB for a full RA run;
  - disk: about 8 GB for `.venv`, 1.2 GB for the E1 weights (in the Hugging Face cache), and
    about 10 GB for the RA embeddings of the 12,927 sequences.
- **Network:** PyPI, GitHub (the prebuilt flash-attn wheel) and Hugging Face (the E1 weights,
  fetched on first use; no login needed).

**Pinned versions:**
- `setup/requirements.txt` pins the top-level packages, and `setup/constraints.txt` pins
  everything else. They match the environment the published embeddings were checked with
  (Python 3.12, torch 2.8.0 with CUDA 12.8, triton 3.4.0, flash-attn 2.8.3.post1,
  transformers 4.56.2).
- Bit-for-bit results depend on these versions.

## Results and reproducibility

- **Default: identical to upstream.** `Generate_Embeddings_E1_fast.py` with `third_party/E1`
  computes exactly what the unmodified E1 computes, in fewer GPU steps. Checks on an RTX 4070
  (18 Nov 2026):
  - 64 of 64 RA queries were bitwise identical to the published embeddings;
  - in NORM mode, all 12,927 sequences were identical to the unmodified code, 1.9× faster;
  - a full RA run of the speed-up code with 4 warps reproduced all 12,927 published
    embeddings bit for bit;
  - the `.venv` built by `start.sh`, with `third_party/E1` and the bundled RMSNorm kernel,
    reproduced the 64-query check sample bit for bit (`benchmarks/verify_venv.py`).
- **Full RA run of 12,927 queries:**
  - about 2.0 h in total (0.55 s per query), including 8 s of homolog selection on the GPU;
  - peak GPU memory 3.8 GiB reserved, peak RAM 5.3 GiB.
  - For comparison, on its longest inputs the original driver needs about 14 GiB of GPU memory,
    more than a 12 GB card has.
- **RMSNorm warp count.**
  - E1's RMSNorm is a Triton kernel that chooses its warp count by timing it on first use in
    each process. When warp counts are nearly tied, separate runs of the same code can differ
    slightly.
  - 4 warps reproduce the published embeddings. The fast driver therefore fixes the count at 4
    (`E1_RMSNORM_NUM_WARPS`; set it to an empty string to let the kernel autotune).
- **Faster, not bit-for-bit (opt-in).** With `E1_BF16_SPEEDUPS=1`, three further changes run
  about 5 % faster, and results differ at bf16 rounding level (per-residue cosine ≥ 0.9997).
  See `third_party/E1/MODIFICATIONS.md`.
- **The RMSNorm kernel is bundled.**
  - Upstream E1 downloads its RMSNorm kernel from the Hugging Face repository
    `kernels-community/triton-layer-norm` on every run, without fixing a version.
  - That repository says that Hugging Face removes kernel repositories of its type from
    13 Sep 2026. Without the kernel, E1 silently falls back to PyTorch's RMSNorm, and the
    embeddings change at bf16 rounding level.
  - `third_party/E1/src/E1/model/triton_layer_norm/` therefore contains the same files, byte
    for byte (the Hub revision used for the published embeddings), and E1 loads that copy.
  - `./start.sh --check-only --import` reports which RMSNorm E1 loaded.

## Layout

| Path | Contents |
| --- | --- |
| `start.sh`, `setup/` | Environment setup and readiness check (`setup/check_env.py`), pinned versions |
| `Generate_Embeddings_E1_fast.py` | The driver to use. Deterministic homolog selection (ties broken by sequence index); bf16 weights cast once; MLM head skipped; CPU work overlapped with the GPU. Uses `third_party/E1` (override with `E1_SOURCE_DIR`). |
| `Generate_Embeddings_E1.py` | The driver that produced the published embeddings, kept for the record. It imports whichever E1 is installed: in `.venv`, that is `third_party/E1`, whose default results are identical. Its original speed needs the unmodified E1 (`pip install "git+https://github.com/Profluent-AI/E1@bfd2620a602248499f3d2583d85a7ecddf0b6e02"`). |
| `third_party/E1/` | Profluent-E1 at `bfd2620` with this project's changes, listed in `MODIFICATIONS.md` |
| `benchmarks/` | The profiling, equivalence and timing scripts behind the numbers above; `benchmarks/README.md` says which script produced which number and how to rerun it. |

Not in the repository:
- the data: `Sequence_Sets/`, `Networks_EValues/`, `Embeddings/`;
- everything the `benchmarks` scripts produce (logs, summaries, test embedding files);
- `.venv`;
- the local git clone of Profluent-AI/E1 the changes were developed in (`E1/`).

## Profluent-E1

- Code: <https://github.com/Profluent-AI/E1> (Profluent Bio Inc.), commit
  `bfd2620a602248499f3d2583d85a7ecddf0b6e02`.
- Model weights: <https://huggingface.co/Profluent-Bio/E1-600m>. They are not included, and
  are downloaded on first use.
- Paper: Profluent E1, bioRxiv, <https://doi.org/10.1101/2025.11.12.688125>
- Licences:
  - Profluent-E1 is under the Profluent-E1 Clickthrough License Agreement and its Attribution
    Guidelines. The model code on its own is under the Apache License 2.0 (Agreement § 4).
  - `third_party/E1` keeps upstream's `LICENSE`, `NOTICE`, `ATTRIBUTION` and `README.md`.
  - Every file changed from upstream starts with a notice, and `third_party/E1/MODIFICATIONS.md`
    lists the changes (Agreement § 5).
  - `src/E1/model/flash_attention_utils.py` is adapted from
    [flash-attention](https://github.com/Dao-AILab/flash-attention) (BSD-3-Clause); its header
    is kept.
  - The bundled kernel `src/E1/model/triton_layer_norm/` is also from flash-attention
    (BSD-3-Clause), with its `LICENSE`, `AUTHORS` and `PROVENANCE.md`.
- In this repository's history, the commit "Import Profluent-E1 at bfd2620, unmodified" is the
  untouched upstream copy, so `git diff` against it shows every change.

## Licence

- The files outside `third_party/` are licensed under the Apache License 2.0 (`LICENSE`).
  Copyright 2026 Xuebin Feng, University of Toronto; see `NOTICE`.
- `third_party/E1` stays under Profluent-E1's own terms (above).
