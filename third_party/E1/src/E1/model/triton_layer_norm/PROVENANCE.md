# Bundled Triton layer-norm kernel (unmodified copy)

This folder is an unmodified copy of the RMSNorm / layer-norm Triton kernel that upstream
Profluent-E1 loads from the Hugging Face Hub every time it runs
(`get_kernel("kernels-community/triton-layer-norm")`, `src/E1/modeling.py` at `bfd2620`).
It was added for EMAP-SSN-E1-Embedding-Generator (2026); it is not part of upstream Profluent-E1.

- **Source:** <https://huggingface.co/kernels-community/triton-layer-norm>, folder
  `build/torch-universal/triton_layer_norm/`.
  - Hub revision `5ebc83aa387c282ff3f233bc3022c3a8be33a013` was `main` from 27 Oct 2025 to
    1 Sep 2026. It produced the published E1 embeddings (2 Aug 2026).
  - Revision `e1ac02e79e08c20f42ff12db8360c53f566326b5` (1 Sep 2026) has the same files.
- **Why it is bundled:**
  - The Hub repository's README (revision `e1ac02e`) says that Hugging Face removes "model"-type
    kernel repositories such as this one from 13 Sep 2026.
  - Without the kernel, E1 falls back to PyTorch's RMSNorm, and the embeddings change at bf16
    rounding level.
  - The upstream call is also not pinned to a revision.
- **Identical to the Hub files.** The git blob ids of the bundled files equal the Hub's (the
  same in both revisions):

  | File | git blob id |
  | --- | --- |
  | `__init__.py` | `9fc8d5c3585310bde70f7a6cda6bfdb8b58870c8` |
  | `_ops.py` | `9fdb014cb58ae4a9eb6bd9da235506f4969922b7` |
  | `layer_norm.py` | `0e333f39b04c82d009d34efebee8cf8a5078b231` |
  | `layers.py` | `c732ca21cba5a45a346c67fb6370db5b9e3788bb` |
  | `utils/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
  | `utils/library.py` | `05324bb01a48daf051b0fee6b8000bd16d92da9d` |
  | `utils/torch.py` | `98cbf9a274cb1513c710fb749845ae241afa758a` |

  To check, run `git hash-object <file>`.
- **Licences:**
  - The kernel is from [flash-attention](https://github.com/Dao-AILab/flash-attention)
    (`layer_norm.py`: "Copyright (c) 2024, Tri Dao."). The Hub repository declares
    `license: bsd-3-clause`.
  - `LICENSE` and `AUTHORS` here are flash-attention's BSD 3-Clause licence and authors file,
    copied from the flash-attn 2.8.3.post1 distribution.
  - `utils/library.py` is adapted from PyTorch (its header says so). PyTorch is also under a
    BSD-style licence, which ships with torch (`torch-*.dist-info/LICENSE`).
- **How E1 uses it:**
  - `src/E1/modeling.py` imports this copy as `E1.model.triton_layer_norm`.
  - It tries the Hub only if this copy cannot be imported, and uses PyTorch's RMSNorm as the
    last resort.
  - Only one of the two can load in a process: both register the same custom-op namespace
    (`_triton_layer_norm_9b61b27_dirty`, `_ops.py`).
