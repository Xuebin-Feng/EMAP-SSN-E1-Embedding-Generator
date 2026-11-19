# Changes from upstream Profluent-E1

This folder is **Profluent-E1** (<https://github.com/Profluent-AI/E1>, Profluent Bio Inc.) at
commit `bfd2620a602248499f3d2583d85a7ecddf0b6e02`, with the changes listed below, made in 2026
for EMAP-SSN-E1-Embedding-Generator. This is not an official Profluent release.

- Upstream's `LICENSE` (Profluent-E1 Clickthrough License Agreement), `NOTICE`, `ATTRIBUTION`
  and `README.md` are kept unchanged and apply to this folder, including the changes.
- In this repository, the commit "Import Profluent-E1 at bfd2620, unmodified" is the untouched
  copy, so `git diff <that commit> -- third_party/E1` shows every change from upstream.
- Every changed file starts with a comment that says so, and the added files say they are not
  upstream. The exception is the bundled kernel in `src/E1/model/triton_layer_norm/`: it is
  kept byte-identical to its source, and its `PROVENANCE.md` says where it comes from.
- Only inference code changed. The model weights are not included: E1 downloads them from
  Hugging Face (`Profluent-Bio/E1-600m`), under the same Agreement.

## Default: results identical to upstream, bit for bit

Same arithmetic as upstream, in fewer GPU steps. Checked on an RTX 4070 (18 Nov 2026):
- 64 of 64 retrieval-augmented (RA) queries were bitwise identical to embeddings made with
  the unmodified code;
- in single-sequence mode, 12,927 of 12,927 sequences were identical to the unmodified code.

| File | Change |
| --- | --- |
| `src/E1/modeling.py` | One host sync per forward pass for position checks, RoPE table size and padding (upstream synced again in RoPE in every layer and in unpadding in every within-sequence layer). Builds the per-forward values shared by all layers (unpadding data, RoPE cache). Chooses the faster block-mask builder when there is no padding. Optional fixed warp count for the kernels-hub RMSNorm: `E1_RMSNORM_NUM_WARPS` / `pin_rmsnorm_num_warps()`; unset, the kernel autotunes as upstream does. |
| `src/E1/model/attention.py` | RoPE cos/sin gathered once per forward pass per layer type and shared by the layers of that type. Optional fused clamp + RoPE kernel. |
| `src/E1/model/ffn.py` | Optional fused SiLU × gate kernel. |
| `src/E1/model/flash_attention.py` | Uses the unpadding data computed once per forward pass. Without padding, the gather/scatter copies around flash attention become reshapes. |
| `src/E1/model/flash_attention_utils.py` | `get_varlen_metadata()`: unpadding data computed once (upstream recomputed it, with three host syncs, in each within-sequence layer). This file is adapted from flash-attention (BSD-3-Clause); its header is kept. |
| `src/E1/model/flex_attention.py` | `create_block_causal_mask_from_block_bounds()`: the same block-causal `BlockMask`, built from tile corner ids instead of the dense seqlen × seqlen mask. |
| `src/E1/model/fused_ops.py` (added) | Triton kernels for clamp + RoPE and SiLU × gate. They round to bf16 at the same points as eager PyTorch and are compiled with fp-fusion off. Each can be switched off. |
| `src/E1/model/triton_layer_norm/` (added, third-party, unmodified) | The RMSNorm / layer-norm Triton kernel that upstream downloads from the Hugging Face Hub on every run (`kernels-community/triton-layer-norm`, unpinned; that repository is due for removal). It is byte-identical to Hub revisions `5ebc83a` (used for the published embeddings) and `e1ac02e`. It is BSD-3-Clause, from flash-attention; `LICENSE`, `AUTHORS` and `PROVENANCE.md` are in the folder. |
| `src/E1/modeling.py` (kernel loading) | Imports the bundled kernel. It falls back to the Hub (`kernels.get_kernel`) only if that fails, and then to PyTorch's RMSNorm, as upstream does. The unconditional `from kernels import get_kernel` at the top is gone, so the `kernels` package is optional. |

## Opt-in: bf16-level speed-ups (`E1_BF16_SPEEDUPS=1`)

These are off by default. With `E1_BF16_SPEEDUPS=1` set before E1 is imported, all three are
on, and results then differ from upstream at the level of bf16 rounding.

Measured on 64 RA queries (RTX 4070, 18 Nov 2026), all three on:
- 0.522 instead of 0.551 s per query;
- per-residue cosine to the reference ≥ 0.99972;
- relative RMS difference 1.0 %.

| File | Change |
| --- | --- |
| `src/E1/model/bf16_options.py` (added) | The three switches: `MERGED_GEMMS`, `FUSED_ADD_NORM`, `QUERY_ONLY_LAST_LAYER`. |
| `src/E1/modeling.py` | `merge_projection_weights()` (MERGED_GEMMS), `RMSNorm.add_and_norm()` (FUSED_ADD_NORM), `E1Model.forward(keep_rows=...)` and `_forward_phase2()` (QUERY_ONLY_LAST_LAYER). With every switch off, `_forward_phase2()` performs upstream's operations in upstream's order. |
| `src/E1/model/attention.py` | Merged q/k/v projection; `Attention.forward_rows()` for the last layer (query rows only, flash attention). |
| `src/E1/model/ffn.py` | Merged w1/w3 projection. |
| `src/E1/model/fused_ops.py` | SiLU × gate kernel for the merged w1/w3 output; the RoPE kernel accepts column slices of the merged q/k/v output. |
