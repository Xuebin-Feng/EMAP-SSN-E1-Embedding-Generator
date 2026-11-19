# Changed for EMAP-SSN-E1-Embedding-Generator (2026); this is not the upstream Profluent-E1 file.
# Upstream: https://github.com/Profluent-AI/E1 at bfd2620. What changed: third_party/E1/MODIFICATIONS.md.
"""PyTorch E1 model."""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .config import E1Config
from .dynamic_cache import DynamicCache
from .model import bf16_options
from .model.attention import Attention, AttentionArgs, AttentionLayerType
from .model.ffn import FFN, GLUMLP
from .model.flash_attention_utils import get_varlen_metadata
from .model.flex_attention import (
    HAS_BLOCK_BOUNDS_MASK,
    FlexAttentionArgs,
    create_block_causal_mask_from_block_bounds,
    create_block_causal_mask_optimized,
)

logger = logging.get_logger(__name__)

# RMSNorm kernel. Upstream fetches it from the Hugging Face Hub on every run, unpinned
# (get_kernel("kernels-community/triton-layer-norm")), and that repository is due to be
# removed. This fork bundles the same files, byte for byte, in model/triton_layer_norm
# (BSD-3-Clause, from flash-attention; see PROVENANCE.md there) and loads that copy. The Hub
# is tried only when the bundled copy cannot be imported, and PyTorch's RMSNorm is the last
# resort, as upstream. Only one copy can load per process: both register the same op names.
try:
    from .model import triton_layer_norm as layer_norm
except Exception as e:  # noqa: BLE001
    logger.warning(f"Failed to import the bundled triton layer norm kernel: {e}; trying the Hugging Face Hub")
    try:
        from kernels import get_kernel

        layer_norm = get_kernel("kernels-community/triton-layer-norm")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load triton layer norm kernel: {e}; Will be using PyTorch RMSNorm instead")
        layer_norm = None


def pin_rmsnorm_num_warps(num_warps=None):
    """Fix the warp count of the Triton RMSNorm kernel instead of autotuning it.

    The Triton kernel behind `layer_norm.rms_norm_fn` is autotuned: on first use in each
    process it times num_warps 1-32 on the live GPU and keeps the fastest. Each warp count
    sums the squares in a different order, so when two counts are nearly tied, separate
    runs of the same code can produce slightly different embeddings. Setting
    E1_RMSNORM_NUM_WARPS (or passing num_warps) makes every run use one warp count.
    Unset, the kernel autotunes exactly as before. Returns the warp count in force, or None.
    """
    if num_warps is None:
        num_warps = os.environ.get("E1_RMSNORM_NUM_WARPS") or None
    if num_warps is None or layer_norm is None:
        return None
    import importlib

    kernel = importlib.import_module(layer_norm.__name__ + ".layer_norm")._layer_norm_fwd_1pass_kernel
    if not hasattr(kernel, "_e1_all_configs"):
        kernel._e1_all_configs = list(kernel.configs)
    chosen = [c for c in kernel._e1_all_configs if c.num_warps == int(num_warps)]
    if not chosen:
        raise ValueError(f"E1_RMSNORM_NUM_WARPS={num_warps}: not one of "
                         f"{[c.num_warps for c in kernel._e1_all_configs]}")
    kernel.configs = chosen  # a single config: Triton skips the benchmark
    kernel.cache.clear()
    return int(num_warps)


RMSNORM_NUM_WARPS = pin_rmsnorm_num_warps()


@dataclass
class E1ModelOutputWithPast(ModelOutput):
    """Base class for model's outputs, with potential hidden states and attentions.

    Attributes:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class E1MaskedLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    mlm_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if layer_norm is None:
            return torch.nn.functional.rms_norm(
                hidden_states, (self.hidden_size,), self.weight, self.variance_epsilon
            ).to(input_dtype)
        else:
            return layer_norm.rms_norm_fn(
                x=hidden_states,
                weight=self.weight,
                bias=None,  # no bias
                residual=None,
                eps=self.variance_epsilon,
                dropout_p=0.0,  # no dropout by default
                prenorm=False,
                residual_in_fp32=False,
            ).to(input_dtype)

    def add_and_norm(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FUSED_ADD_NORM: returns (rms_norm(x + residual), x + residual) from one kernel.

        The sum is normalised in fp32 before rounding (eager rounds it to bf16 first), so
        this differs from norm(residual + x) at bf16 rounding level.
        """
        out, residual_out = layer_norm.rms_norm_fn(
            x=x,
            weight=self.weight,
            bias=None,
            residual=residual,
            eps=self.variance_epsilon,
            dropout_p=0.0,
            prenorm=True,
            residual_in_fp32=False,
        )
        return out.to(x.dtype), residual_out


class NormAttentionNorm(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, DynamicCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.norm_attn_norm = NormAttentionNorm(config, layer_idx)
        self.ffn = FFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, DynamicCache | None]:
        hidden_states, residual, self_attn_weights, present_key_value = self.norm_attn_norm(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, present_key_value


class E1PreTrainedModel(PreTrainedModel):
    config_class = E1Config
    config: E1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _transformer_layer_cls = [DecoderLayer]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        super().post_init()

    def _backward_compatibility_gradient_checkpointing(self) -> None:
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable(dict(use_reentrant=False))

    @classmethod
    def from_pretrained(  # type: ignore[no-untyped-def]
        cls, pretrained_model_name_or_path: str | os.PathLike | None, *args, **kwargs
    ) -> "E1PreTrainedModel":
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


class E1Model(E1PreTrainedModel):
    config: E1Config

    def __init__(self, config: E1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        keep_rows: torch.LongTensor | None = None,
    ) -> E1ModelOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool
            keep_rows: (n,) token indices of the LAST sequence to return (batch size 1 only).
                With bf16_options.QUERY_ONLY_LAST_LAYER the last layer is computed only for
                these rows and last_hidden_state has shape (1, n, hidden_size).

        Returns:
            E1ModelOutputWithPast: Model Outputs
        """
        batch_size, seq_length = input_ids.shape

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        elif not use_cache:
            # To avoid weirdness with gradient checkpointing: https://github.com/huggingface/transformers/issues/28499
            past_key_values = None

        global_position_ids = global_position_ids.view(-1, seq_length).long()
        within_seq_position_ids = within_seq_position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()

        # One host sync for every data-dependent value this forward needs. The original synced
        # here, again in RoPE in every layer and in unpadding in every within-sequence layer,
        # each time leaving the GPU idle while the CPU caught up.
        max_position_id, min_position_id, max_global_position_id, num_padding = torch.stack(
            [
                within_seq_position_ids.max(),
                within_seq_position_ids.min(),
                global_position_ids.max(),
                (sequence_ids == -1).sum(),
            ]
        ).tolist()
        assert max_position_id < self.config.max_num_positions_within_seq and min_position_id >= -1, (
            f"Position ids must be in the range [-1, {self.config.max_num_positions_within_seq}); got max {max_position_id} and min {min_position_id}"
        )

        inputs_embeds = self.embed_tokens(input_ids)
        # -1 is used to indicate padding tokens, so we need to clamp the sequence ids to 0
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids.clamp(min=0))

        # In case we need to do any manual typecasting
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)

        # (batch_size, query_length, keyval_length)
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        # Create block mask for flex attention
        attention_args: AttentionArgs | None = None
        if past_key_values_length == 0:
            if num_padding == 0 and HAS_BLOCK_BOUNDS_MASK:
                block_mask = create_block_causal_mask_from_block_bounds(sequence_ids)
            else:
                block_mask = create_block_causal_mask_optimized(sequence_ids)
            flex_attention_args = FlexAttentionArgs(block_mask=block_mask)
            attention_args = AttentionArgs(
                flex_attention_args=flex_attention_args,
                varlen=get_varlen_metadata(sequence_ids, has_padding=num_padding > 0),
                rope_seq_len={
                    AttentionLayerType.WITHIN_SEQ: max_position_id + 1,
                    AttentionLayerType.GLOBAL: max_global_position_id + 1,
                },
                rope_cache={},
            )

        phase2 = (
            attention_args is not None
            and not use_cache
            and not output_hidden_states
            and not output_attentions
            and not (self.gradient_checkpointing and self.training)
            and (bf16_options.FUSED_ADD_NORM and layer_norm is not None or keep_rows is not None)
        )
        if phase2:
            rows_ok = (
                keep_rows is not None
                and bf16_options.QUERY_ONLY_LAST_LAYER
                and batch_size == 1
                and num_padding == 0
                and self.layers[-1].norm_attn_norm.self_attn.layer_type == AttentionLayerType.GLOBAL
            )
            hidden_states = self._forward_phase2(
                hidden_states,
                within_seq_position_ids,
                global_position_ids,
                sequence_ids,
                attention_args,
                keep_rows if rows_ok else None,
            )
            return E1ModelOutputWithPast(last_hidden_state=hidden_states)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    within_seq_position_ids,
                    global_position_ids,
                    sequence_ids,
                    attention_args,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    attention_args=attention_args,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states, self_attn_weights, present_key_value = layer_outputs

            if use_cache:
                # NOTE: it's necessary to re-assign past_key_values because FSDP2
                # passes certain arguments by value, not by reference.
                # See https://github.com/huggingface/transformers/issues/38190#issuecomment-2914016168
                next_decoder_cache = past_key_values = present_key_value

            if output_attentions:
                all_self_attns += (self_attn_weights,)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        next_cache = next_decoder_cache if use_cache else None

        return E1ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _forward_phase2(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs,
        keep_rows: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Decoder stack for inference with the phase-2 options (bf16_options).

        The true hidden state is residual + delta; the add is deferred so it can be fused
        into the next RMSNorm (FUSED_ADD_NORM). With FUSED_ADD_NORM off and keep_rows None
        this performs exactly the upstream operations in the upstream order.
        """
        fuse = bf16_options.FUSED_ADD_NORM and layer_norm is not None
        residual, delta = hidden_states, None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            block = layer.norm_attn_norm
            if delta is None:
                h = block.input_layernorm(residual)
            elif fuse:
                h, residual = block.input_layernorm.add_and_norm(delta, residual)
            else:
                residual = residual + delta
                h = block.input_layernorm(residual)
            if i == last and keep_rows is not None:
                attn_out = block.self_attn.forward_rows(h, keep_rows, global_position_ids, attention_args)
                residual = residual[:, keep_rows]
            else:
                attn_out, _, _ = block.self_attn(
                    hidden_states=h,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    attention_args=attention_args,
                )
            if fuse:
                h, residual = block.post_attention_layernorm.add_and_norm(attn_out, residual)
            else:
                residual = residual + attn_out
                h = block.post_attention_layernorm(residual)
            delta = layer.ffn(h)
        if fuse:
            return self.norm.add_and_norm(delta, residual)[0]
        return self.norm(residual + delta)


def merge_projection_weights(model: nn.Module) -> None:
    """MERGED_GEMMS: store q/k/v and w1/w3 weights as one tensor each per layer; the
    original Linear weights become views of it (no extra memory). Call after any dtype cast."""
    for module in model.modules():
        if isinstance(module, Attention):
            qkv = torch.cat([module.q_proj.weight, module.k_proj.weight, module.v_proj.weight]).contiguous()
            n_q, n_k = module.q_proj.weight.shape[0], module.k_proj.weight.shape[0]
            module.q_proj.weight = nn.Parameter(qkv[:n_q], requires_grad=False)
            module.k_proj.weight = nn.Parameter(qkv[n_q : n_q + n_k], requires_grad=False)
            module.v_proj.weight = nn.Parameter(qkv[n_q + n_k :], requires_grad=False)
            module.qkv_weight = qkv
        elif isinstance(module, GLUMLP):
            w13 = torch.cat([module.w1.weight, module.w3.weight]).contiguous()
            n = module.w1.weight.shape[0]
            module.w1.weight = nn.Parameter(w13[:n], requires_grad=False)
            module.w3.weight = nn.Parameter(w13[n:], requires_grad=False)
            module.w13_weight = w13


class E1ForMaskedLM(E1PreTrainedModel):
    config: E1Config

    def __init__(self, config: E1Config):
        super().__init__(config)
        self.model: E1Model = E1Model(config)
        self.vocab_size = config.vocab_size
        self.mlm_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
            torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True),
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> E1MaskedLMOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            labels: (batch_size, seq_length)
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool

        Returns:
            E1MaskedLMOutputWithPast: Model Outputs
        """
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        loss = None

        # Compute masked language modeling loss
        mlm_logits = self.mlm_head(hidden_states).float()
        mlm_loss = 0.0
        if labels is not None:
            mlm_logits_flat = mlm_logits.contiguous().view(-1, self.config.vocab_size)
            mlm_labels_flat = labels.to(mlm_logits_flat.device).contiguous().view(-1)
            mlm_loss = F.cross_entropy(mlm_logits_flat, mlm_labels_flat, reduction="none")
            mask = mlm_labels_flat != self.model.padding_idx
            n_mlm = mask.sum()
            mlm_loss = (mlm_loss * mask.to(mlm_loss)).sum() / (1 if n_mlm == 0 else n_mlm)
            loss = 0.0
            loss += mlm_loss

        return E1MaskedLMOutputWithPast(
            loss=loss,
            mlm_loss=mlm_loss,
            logits=mlm_logits,
            embeddings=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
