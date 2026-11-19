# Added for EMAP-SSN-E1-Embedding-Generator (2026); not part of upstream Profluent-E1.
# Distributed with third_party/E1 under the same terms (see LICENSE). Details: MODIFICATIONS.md.
"""Fused Triton kernels that reproduce E1's eager bf16 arithmetic bit for bit.

Eager PyTorch evaluates every bf16 elementwise op in fp32 and rounds the result to bf16
(round to nearest even). These kernels round at exactly the same points:

* rope_qkv: clamp(q, k, v) to +-clip_qkv, then RoPE on q and k:
      out = bf16( bf16(x * cos) + bf16(rotate_half(x) * sin) )
  The product of two bf16 numbers is exact in fp32, so each bf16(product) equals the
  eager result; the add is one fp32 add rounded to bf16, as in eager. The q/k/v of
  global layers are written straight into the (batch, head, token, dim) layout that
  flex attention copies them into, which removes that copy.
* silu_mul: bf16( bf16(silu(a)) * b ), with silu(a) = a / (1 + expf(-a)) computed with
  the same libdevice expf and IEEE division as PyTorch's CUDA silu kernel.

FMA contraction is disabled for both kernels so that no multiply and add are merged.
Whether these are bit-exact on a given GPU / PyTorch build is checked by
the kernel stress tests in benchmarks/verify_exact.py and by comparing whole embeddings with the published file.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

# Switches (the driver / tests can turn either off to measure or to fall back to eager).
ENABLE_ROPE = True
ENABLE_SILU_MUL = True


if HAS_TRITON:

    @triton.jit
    def _clamp(x, clip, HAS_CLIP: tl.constexpr):
        if HAS_CLIP:
            x = tl.minimum(tl.maximum(x, -clip), clip)
        return x

    @triton.jit
    def _rope_one(X, x_stride_t, XO, xo_sb, xo_st, xo_sh, t, b, n, c, s, clip,
                  H: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr, HAS_CLIP: tl.constexpr):
        hs = tl.arange(0, BLOCK_H)[:, None]
        ds = tl.arange(0, D)[None, :]
        ds_rot = (ds + D // 2) % D
        m = hs < H
        base = X + t * x_stride_t + hs * D
        x = _clamp(tl.load(base + ds, mask=m, other=0.0).to(tl.float32), clip, HAS_CLIP)
        xr = _clamp(tl.load(base + ds_rot, mask=m, other=0.0).to(tl.float32), clip, HAS_CLIP)
        xr = tl.where(ds < D // 2, -xr, xr)  # rotate_half: cat(-x2, x1)
        t1 = (x * c).to(tl.bfloat16).to(tl.float32)
        t2 = (xr * s).to(tl.bfloat16).to(tl.float32)
        tl.store(XO + b * xo_sb + n * xo_st + hs * xo_sh + ds, (t1 + t2).to(tl.bfloat16), mask=m)

    @triton.jit
    def _rope_qkv_kernel(
        Q, K, V, COS, SIN, QO, KO, VO,
        seq_len, q_stride_t, kv_stride_t, cs_stride_t,
        qo_sb, qo_st, qo_sh, ko_sb, ko_st, ko_sh, vo_sb, vo_st, vo_sh,
        clip,
        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr, HAS_CLIP: tl.constexpr,
    ):
        t = tl.program_id(0)  # token index over batch * seq_len
        b = t // seq_len
        n = t - b * seq_len
        ds = tl.arange(0, D)[None, :]
        c = tl.load(COS + t * cs_stride_t + ds).to(tl.float32)
        s = tl.load(SIN + t * cs_stride_t + ds).to(tl.float32)
        _rope_one(Q, q_stride_t, QO, qo_sb, qo_st, qo_sh, t, b, n, c, s, clip, HQ, D, BLOCK_H, HAS_CLIP)
        _rope_one(K, kv_stride_t, KO, ko_sb, ko_st, ko_sh, t, b, n, c, s, clip, HKV, D, BLOCK_H, HAS_CLIP)
        hs = tl.arange(0, BLOCK_H)[:, None]
        m = hs < HKV
        v = _clamp(tl.load(V + t * kv_stride_t + hs * D + ds, mask=m, other=0.0).to(tl.float32), clip, HAS_CLIP)
        tl.store(VO + b * vo_sb + n * vo_st + hs * vo_sh + ds, v.to(tl.bfloat16), mask=m)

    @triton.jit
    def _silu_mul_kernel(A, B, O, n_elements, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_elements
        a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=m, other=0.0).to(tl.float32)
        silu = libdevice.div_rn(a, 1.0 + libdevice.exp(-a))
        tl.store(O + offs, (silu.to(tl.bfloat16).to(tl.float32) * b).to(tl.bfloat16), mask=m)


def _rows_ok(x: torch.Tensor) -> bool:
    # (B, N, C) with unit stride along C and token t = b * N + n at offset t * stride(1);
    # true for contiguous tensors and for column slices of a merged q/k/v output.
    return x.dim() == 3 and x.stride(2) == 1 and x.stride(0) == x.shape[1] * x.stride(1)


def rope_qkv_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor) -> bool:
    return (
        ENABLE_ROPE
        and HAS_TRITON
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == cos.dtype == torch.bfloat16
        and _rows_ok(q) and _rows_ok(k) and _rows_ok(v)
        and k.stride(1) == v.stride(1)
        and cos.is_contiguous()
    )


def rope_qkv(q, k, v, cos, sin, clip, head_dim, heads_first):
    """q: (B, N, HQ*D), k, v: (B, N, HKV*D) linear outputs; cos, sin: (B, N, 1, D).

    Returns (B, N, H, D) tensors equal to the eager clamp + RoPE result. With
    heads_first=True they are views of (B, H, N, D) buffers (the layout flex attention needs).
    """
    B, N, _ = q.shape
    D = head_dim
    HQ, HKV = q.shape[-1] // D, k.shape[-1] // D

    def out(h):
        if heads_first:
            return torch.empty(B, h, N, D, device=q.device, dtype=q.dtype).transpose(1, 2)
        return torch.empty(B, N, h, D, device=q.device, dtype=q.dtype)

    qo, ko, vo = out(HQ), out(HKV), out(HKV)
    _rope_qkv_kernel[(B * N,)](
        q, k, v, cos, sin, qo, ko, vo,
        N, q.stride(1), k.stride(1), cos.stride(1),
        qo.stride(0), qo.stride(1), qo.stride(2),
        ko.stride(0), ko.stride(1), ko.stride(2),
        vo.stride(0), vo.stride(1), vo.stride(2),
        float(clip) if clip is not None else 0.0,
        HQ=HQ, HKV=HKV, D=D, BLOCK_H=triton.next_power_of_2(max(HQ, HKV)), HAS_CLIP=clip is not None,
        enable_fp_fusion=False,
    )
    return qo, ko, vo


def silu_mul_supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        ENABLE_SILU_MUL
        and HAS_TRITON
        and a.is_cuda
        and a.dtype == b.dtype == torch.bfloat16
        and a.shape == b.shape
        and a.is_contiguous() and b.is_contiguous()
    )


if HAS_TRITON:

    @triton.jit
    def _silu_mul_merged_kernel(GU, O, n_elements, F_DIM, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_elements
        row = offs // F_DIM
        col = offs - row * F_DIM
        a = tl.load(GU + row * 2 * F_DIM + col, mask=m, other=0.0).to(tl.float32)
        b = tl.load(GU + row * 2 * F_DIM + F_DIM + col, mask=m, other=0.0).to(tl.float32)
        silu = libdevice.div_rn(a, 1.0 + libdevice.exp(-a))
        tl.store(O + offs, (silu.to(tl.bfloat16).to(tl.float32) * b).to(tl.bfloat16), mask=m)


def silu_mul_merged_supported(gate_up: torch.Tensor) -> bool:
    return (
        ENABLE_SILU_MUL and HAS_TRITON and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous() and gate_up.shape[-1] % 2 == 0
    )


def silu_mul_merged(gate_up: torch.Tensor) -> torch.Tensor:
    """silu_mul for the output of the merged w1/w3 GEMM: (..., 2F) -> (..., F)."""
    f = gate_up.shape[-1] // 2
    o = torch.empty(*gate_up.shape[:-1], f, device=gate_up.device, dtype=gate_up.dtype)
    n = o.numel()
    BLOCK = 2048
    _silu_mul_merged_kernel[(triton.cdiv(n, BLOCK),)](gate_up, o, n, f, BLOCK=BLOCK, enable_fp_fusion=False)
    return o


def silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16(bf16(silu(a)) * b), equal to F.silu(a) * b in eager bf16."""
    o = torch.empty_like(a)
    n = a.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](a, b, o, n, BLOCK=BLOCK, enable_fp_fusion=False)
    return o
