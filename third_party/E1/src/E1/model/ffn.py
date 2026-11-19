# Changed for EMAP-SSN-E1-Embedding-Generator (2026); this is not the upstream Profluent-E1 file.
# Upstream: https://github.com/Profluent-AI/E1 at bfd2620. What changed: third_party/E1/MODIFICATIONS.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ..config import E1Config
from . import bf16_options, fused_ops


class MLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class GLUMLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.act_is_silu = isinstance(self.act_fn, nn.SiLU)
        self.w13_weight: torch.Tensor | None = None  # set by merge_projection_weights (MERGED_GEMMS)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if bf16_options.MERGED_GEMMS and self.w13_weight is not None:
            gate_up = F.linear(hidden_states, self.w13_weight)  # one GEMM for w1 and w3
            if self.act_is_silu and fused_ops.silu_mul_merged_supported(gate_up):
                hidden_states = fused_ops.silu_mul_merged(gate_up)
            else:
                hidden_states = self.act_fn(gate_up[..., : self.ffn_dim]) * gate_up[..., self.ffn_dim :]
            return self.w2(hidden_states)
        gate, up = self.w1(hidden_states), self.w3(hidden_states)
        if self.act_is_silu and fused_ops.silu_mul_supported(gate, up):
            hidden_states = fused_ops.silu_mul(gate, up)  # one kernel, same bf16 rounding as eager
        else:
            hidden_states = self.act_fn(gate) * up
        hidden_states = self.w2(hidden_states)
        return hidden_states


class FFN(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        mlp_cls = GLUMLP if config.gated_mlp else MLP
        self.mlp = mlp_cls(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)
