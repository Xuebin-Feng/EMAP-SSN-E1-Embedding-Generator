# Benchmarks and equivalence checks

The scripts behind the numbers in the main README. They were run on 18 Nov 2026 on one
machine:
- Intel i5-13600K, 23 GiB RAM visible to WSL2 (Windows 11);
- NVIDIA RTX 4070 (12 GB), driver 621.24;
- torch 2.8.0+cu128, triton 3.4.0, flash-attn 2.8.3.post1.

## How to run them

- **Environment:** run them from the project folder in the project environment, for example
  `./start.sh python benchmarks/verify_exact.py`. Each script finds the project folder from
  its own location.
- **Outputs:** they go next to the scripts, in `benchmarks/<topic>/` and `benchmarks/logs/`.
  These folders are not tracked by git.
- **Inputs**, which these scripts expect in the project folder itself:
  - `Sequence_Sets/Foldtype_IV_ATAs_13000.fasta`;
  - the ESMC 6B network `Networks_EValues/Foldtype_IV_ATAs_13000_[esmc_6b]_network.h5`;
  - for comparisons, the published embeddings
    `Embeddings/Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5`, which is only ever opened
    read-only.

  If the data lives elsewhere, link the three folders into the project folder (git ignores
  them): `ln -s /path/to/data/{Sequence_Sets,Networks_EValues,Embeddings} .`
  (`--data-dir` and `E1_DATA_DIR` apply to the driver's command line only.)
- **Homolog map:** several scripts also use `benchmarks/homolog_map.json`, the homolog
  selection of every query. `check_ties.py` writes it; run that first.
- **GPU scripts:** they need an otherwise idle GPU for meaningful timings.
- **Scripts about the original code** (the "Original code" rows below) import
  `Generate_Embeddings_E1.py` and whichever `E1` package is installed. To reproduce the
  original numbers, use an environment with the unmodified Profluent-E1:
  `pip install "git+https://github.com/Profluent-AI/E1@bfd2620a602248499f3d2583d85a7ecddf0b6e02"`.
  In this project's `.venv`, `E1` is `third_party/E1`.

## Which script gave which result

| Result | Script(s) | Output |
| --- | --- | --- |
| **Homolog selection is deterministic and machine-independent.** The same 12,927 contexts as the published run on CPU (1 and 4 threads), CUDA, numpy on Linux and on Windows, once score ties are broken by sequence index. | `selection_determinism.py --device cpu\|cuda --threads N`, `select_numpy.py <network.h5> <out_dir> <tag>`, `compare_selections.py`, `check_ties.py` | `selection/` |
| The published embedding file and its input hashes | `inspect_reference.py` | printed |
| **Original code:** 0.716 s/query on 64 random RA queries (seed 42), 64/64 bitwise identical to the published file; time breakdown per operation; attention variants; CPU overheads | `profile_baseline.py`, `profile_detail.py`, `attn_microbench.py`, `cpu_overheads.py` | `logs/` |
| The block-causal `BlockMask` built from tile corners equals upstream's (CPU) | `test_block_mask.py` | printed |
| **Bit-for-bit speed-ups, step by step.** Same 64 queries; every step 64/64 bitwise identical to the published file: 0.700 → 0.647 → 0.644 → 0.629 → 0.567 → 0.551 s/query. Also stress tests of the fused kernels against eager PyTorch. | `verify_exact.py` | `verify/` |
| **Opt-in bf16 speed-ups** (`E1_BF16_SPEEDUPS`), each switch alone and all three: all three give 0.522 s/query, per-residue cosine ≥ 0.99972, relative RMS difference 1.0 %. With all switches off, 64/64 bitwise. | `verify_bf16.py` | `verify_bf16/` |
| **Single-sequence (NORM) mode, all 12,927 sequences.** Fast and original drivers bitwise identical; 0.0585 → 0.0304 s/query. Also RA vs NORM similarity. | `norm_run.py --driver original\|fast --tag NAME`, then `compare_norm.py` | `norm/` |
| **Full RA run, 12,927 queries.** 7,175 s in all (0.554 s/query for generation, 8 s homolog selection); peak GPU 3.8 GiB reserved, peak RSS 5.3 GiB; 12,927/12,927 bitwise identical to the published file. | `full_run.sh <tag>` (runs `full_run.py`, logs GPU and machine load), then `check_load.py <tag>` | `full/` |
| Two full runs compared range by range (+0.45 %) | `compare_runs.py [ref tag] [tag]` | printed |
| **Why an unpinned run can differ from the published file.** The RMSNorm kernel's warp-count autotuning: 4 warps reproduce the published file, 2 warps another run; near-tie timings. | `diag_rmsnorm.py`, `diag_exact.py` | `diag/` |
| **GPU memory on the 24 longest RA inputs.** The original needs about 14.4 GiB and spills to system RAM on a 12 GB card; the fast driver needs 3.2 GiB and does not spill, even with 6 GiB of the GPU taken by other programs. | `vram_test.py --driver original\|fast [--fillers=2,4,6]` | `vram/` |
| **The `.venv` made by `start.sh` reproduces the published embeddings:** 64/64 bitwise identical, 0.551 s/query | `verify_venv.py` | `verify/venv_check.json` |

## Notes

- **`full_run.sh` under WSL** also records the Windows GPU process list and, with
  `winload.ps1`, the Windows CPU load, so that `check_load.py` can flag runs other programs
  may have slowed down.
- **The fast driver fixes the RMSNorm warp count at 4** (`E1_RMSNORM_NUM_WARPS`). Without it,
  the kernel autotunes on first use in each process, and two runs of the same code can differ
  at bf16 rounding level (`diag_rmsnorm.py`).
- **Some scripts need the large full-run embedding files**, which were deleted after the
  checks: `diag_exact.py`, and the comparison part of `diag_rmsnorm.py`. Running
  `full_run.sh` recreates them.
