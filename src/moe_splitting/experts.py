"""GeGLU Expert Bank compatible with Gemma 2 / Gemma 3 MLP weights."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeGLUMoEExperts(nn.Module):
    """
    Stacked bank of E Gemma-style GeGLU experts.

    Each expert j computes:
        h_j(x) = (gelu_tanh(x @ gate_proj[j].T) * (x @ up_proj[j].T)) @ down_proj[j].T
    Storing weights in stacked (E, ...) tensors enables vectorized donor->dead cloning,
    orthogonal noise injection, and clean AdamW optimizer state migration.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        std = 0.02
        self.gate_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * std)
        self.up_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * std)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * std)

    def forward(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to their assigned Top-K_r experts and compute routing energy.

        Returns:
            out: Aggregated tensor of shape (T, hidden_size).
            batch_energy: Mean per-token weighted squared L2 energy (E,) in FP32.
        """
        T, d = x_flat.shape
        out = torch.zeros_like(x_flat)
        batch_energy = torch.zeros(self.num_experts, device=x_flat.device, dtype=torch.float32)

        for j in range(self.num_experts):
            # Mask of shape (T, K_r) where expert j was chosen
            match_mask = topk_idx == j
            if not match_mask.any():
                continue

            tok_pos, slot_pos = torch.nonzero(match_mask, as_tuple=True)
            x_j = x_flat[tok_pos]                             # (T_j, d)
            w_j = topk_weights[tok_pos, slot_pos].unsqueeze(-1)  # (T_j, 1)

            gated = F.gelu(F.linear(x_j, self.gate_proj[j]), approximate="tanh")
            up = F.linear(x_j, self.up_proj[j])
            y_j = F.linear(gated * up, self.down_proj[j])     # (T_j, d)

            out.index_add_(0, tok_pos, (w_j * y_j).to(out.dtype))

            # S_j signal: g_j(x) * ||h_j(x)||_2^2 averaged over T tokens
            sq_norm = (y_j.float() ** 2).sum(dim=-1)
            weighted_e = (w_j.squeeze(-1).float() * sq_norm).sum() / max(T, 1)
            batch_energy[j] = weighted_e.detach()

        return out, batch_energy
