# Changed for EMAP-SSN-E1-Embedding-Generator (2026); this is not the upstream Profluent-E1 file.
# Upstream: https://github.com/Profluent-AI/E1 at bfd2620. What changed: third_party/E1/MODIFICATIONS.md.
from collections.abc import Callable
from typing import TypedDict

try:
    import torch
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention, dynamic=True)
except ImportError:
    flex_attention = None

try:  # private helpers that create_block_mask itself uses (torch 2.8); optional
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, _create_sparse_block_from_block_mask

    HAS_BLOCK_BOUNDS_MASK = True
except ImportError:
    HAS_BLOCK_BOUNDS_MASK = False


class FlexAttentionArgs(TypedDict, total=False):
    block_mask: BlockMask | None
    score_mod: Callable | None


def create_block_causal_mask_optimized(sequence_ids: torch.Tensor) -> BlockMask:
    # Assumes sequence_ids is sorted in increasing order for each batch item, except for
    # the -1 values, which are used to indicate the padding tokens.
    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    batch_size, seqlen = sequence_ids.shape
    return create_block_mask(document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device)


def create_block_causal_mask_from_block_bounds(sequence_ids: torch.Tensor) -> BlockMask:
    """Same BlockMask as create_block_causal_mask_optimized, for inputs WITHOUT padding.

    create_block_mask evaluates the mask on all seqlen x seqlen pairs and then sums
    128 x 128 tiles. Sequence ids never decrease along a row, so a tile (query block a,
    key block c) has at least one allowed pair iff id[last query] >= id[first key], and
    is fully allowed iff id[first query] >= id[last key] and the tile lies wholly inside
    the input (create_block_mask pads with False, so tiles running past the end are
    never full). Those tile flags go through the same torch routine as before, so every
    tensor of the resulting BlockMask, and the mask_mod applied inside partial tiles, is
    the same; only the O(seqlen^2) mask evaluation is gone. Callers must fall back to
    create_block_causal_mask_optimized when padding (-1) is present.
    """

    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    block = _DEFAULT_SPARSE_BLOCK_SIZE
    batch_size, seqlen = sequence_ids.shape
    starts = torch.arange(0, seqlen, block, device=sequence_ids.device)
    last_idx = torch.clamp(starts + block, max=seqlen) - 1
    first_id = sequence_ids[:, starts]  # (B, n_blocks)
    last_id = sequence_ids[:, last_idx]  # (B, n_blocks)
    complete = (starts + block) <= seqlen  # (n_blocks,)
    any_allowed = last_id[:, :, None] >= first_id[:, None, :]  # (B, q block, kv block)
    full = (first_id[:, :, None] >= last_id[:, None, :]) & complete[:, None] & complete[None, :]
    partial = any_allowed & ~full
    return _create_sparse_block_from_block_mask(
        (partial[:, None].to(torch.int8), full[:, None].to(torch.int8)),
        document_mask,
        (seqlen, seqlen),
        block,
        block,
    )


def flex_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
) -> torch.Tensor:
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    assert score_mod is None, "Score mod is not supported yet"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    outputs = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=query_states.shape[1] != key_states.shape[1],  # if nkv != nh
    )

    outputs = outputs.transpose(1, 2)  # (bs, seqlen, nh, hs)
    return outputs
