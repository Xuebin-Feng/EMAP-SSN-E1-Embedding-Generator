# Added for EMAP-SSN-E1-Embedding-Generator (2026); not part of upstream Profluent-E1.
# Distributed with third_party/E1 under the same terms (see LICENSE). Details: MODIFICATIONS.md.
"""Phase-2 speed-ups: NOT bit-for-bit with upstream E1; results differ at bf16 rounding level.

All three are OFF by default, so E1 computes exactly what upstream computes. Set the
environment variable E1_BF16_SPEEDUPS=1 before E1 is imported to turn all three on. The
code reads these attributes on every forward pass, so they can also be set one at a time
at run time to measure each (see benchmarks/verify_bf16.py); MERGED_GEMMS also needs
modeling.merge_projection_weights(model) after the model is loaded and cast.

MERGED_GEMMS           q/k/v projections as one matrix multiply, w1/w3 as one (the weights
                       become views of one merged tensor, so no extra memory). cuBLAS may
                       pick different kernels for the wider product.
FUSED_ADD_NORM         each residual add is fused into the following RMSNorm. The norm then
                       sees the fp32 sum instead of the bf16-rounded sum.
QUERY_ONLY_LAST_LAYER  in the last (global) layer, attention output, MLP and final norm are
                       computed only for the rows the caller keeps (the query residues);
                       keys and values still use every token. Different GEMM shapes and
                       flash attention instead of flex attention for those rows.

Measured on 64 RA queries (RTX 4070, 18 Nov 2026), all three on: 0.522 vs 0.551 s/query,
per-residue cosine to the published embeddings >= 0.99972, relative RMS difference 1.0 %.
"""
import os

_ON = os.environ.get("E1_BF16_SPEEDUPS", "") == "1"

MERGED_GEMMS = _ON
FUSED_ADD_NORM = _ON
QUERY_ONLY_LAST_LAYER = _ON
