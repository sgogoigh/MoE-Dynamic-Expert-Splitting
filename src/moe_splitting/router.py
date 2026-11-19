"""Auxiliary-loss-free EMA MoE Router plus Switch/DeepSeek-V3 baseline modes."""

from dataclasses import dataclass
import math
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import EMeansMoEConfig


@dataclass
class RouterOutput:
    """Container returned by MoE router forward pass."""

    topk_idx: torch.Tensor       # (T, K_r) long
    topk_weights: torch.Tensor   # (T, K_r) float
    probs: torch.Tensor          # (T, E) softmax probabilities
    batch_load: torch.Tensor     # (E,) token selection fractions summing to ~K_r
    aux_loss: torch.Tensor       # scalar tensor (0.0 in e-means / deepseek_bias modes)
    z_loss: torch.Tensor         # scalar router z-loss


class EMeansMoERouter(nn.Module):
    """
    Sparse Top-K_r MoE router tracking FP32 EMA utilization (N_j) and energy (S_j).

    Supports:
      - router_mode="emeans": Pure LM loss + EMA load/energy tracking (no aux penalty).
      - router_mode="unbalanced": Pure LM loss baseline (demonstrates routing collapse).
      - router_mode="aux_loss": Switch / Mixtral differentiable auxiliary load-balancing loss.
      - router_mode="deepseek_bias": Detached per-expert selection bias updated by sign(error).
    """

    def __init__(self, d_model: int, cfg: EMeansMoEConfig):
        super().__init__()
        self.d_model = d_model
        self.cfg = cfg
        self.E = cfg.num_experts
        self.K_r = cfg.top_k
        self.tau_r = cfg.tau_r

        self.gate = nn.Linear(d_model, self.E, bias=True)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

        fair_share = self.K_r / self.E
        self.register_buffer("ema_load", torch.full((self.E,), fair_share, dtype=torch.float32))
        self.register_buffer("ema_energy", torch.zeros(self.E, dtype=torch.float32))
        self.register_buffer("cooldown", torch.zeros(self.E, dtype=torch.long))
        # Detached dynamic bias used only by "deepseek_bias" baseline comparator
        self.register_buffer("deepseek_bal_bias", torch.zeros(self.E, dtype=torch.float32))

    def cosine_alpha(self, step: int, total_steps: int) -> float:
        """Cosine decay from alpha_0 down to non-stationary tracking floor alpha_min."""
        progress = min(max(step, 0), max(total_steps, 1)) / max(total_steps, 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.cfg.alpha_min + (self.cfg.alpha_0 - self.cfg.alpha_min) * cos_val)

    def forward(self, x_flat: torch.Tensor) -> RouterOutput:
        """
        Compute routing logits, Top-K_r assignments, and baseline losses (if enabled).

        Args:
            x_flat: Flattened token hidden states of shape (T, d_model).
        """
        logits = self.gate(x_flat) / self.tau_r  # (T, E)
        probs = F.softmax(logits.float(), dim=-1)

        if self.cfg.router_mode == "deepseek_bias":
            sel_scores = probs + self.deepseek_bal_bias.unsqueeze(0)
            _, topk_idx = torch.topk(sel_scores, k=self.K_r, dim=-1)
            topk_weights = torch.gather(probs, dim=-1, index=topk_idx)
        else:
            topk_weights, topk_idx = torch.topk(probs, k=self.K_r, dim=-1)

        if self.cfg.renorm_topk_weights and self.K_r > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        topk_weights = topk_weights.to(x_flat.dtype)

        # Token fraction routed to each expert (hard counts / T)
        one_hot = F.one_hot(topk_idx, num_classes=self.E).float()  # (T, K_r, E)
        batch_load = one_hot.sum(dim=1).mean(dim=0)                 # (E,), sum == K_r

        # Optional Switch auxiliary loss baseline
        if self.cfg.router_mode == "aux_loss" and self.cfg.aux_loss_coef > 0.0:
            mean_probs = probs.mean(dim=0)
            # Standard Switch formulation normalized by K_r so uniform gives 1.0
            aux_loss = self.cfg.aux_loss_coef * (self.E / max(self.K_r, 1)) * torch.sum(batch_load.detach() * mean_probs)
        else:
            aux_loss = logits.new_zeros(())

        # Optional router z-loss
        if self.cfg.router_z_loss_coef > 0.0:
            z_loss = self.cfg.router_z_loss_coef * torch.mean(torch.logsumexp(logits.float(), dim=-1) ** 2)
        else:
            z_loss = logits.new_zeros(())

        return RouterOutput(
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            probs=probs,
            batch_load=batch_load.detach(),
            aux_loss=aux_loss,
            z_loss=z_loss,
        )

    @torch.no_grad()
    def update_ema(
        self,
        batch_load: torch.Tensor,
        batch_energy: torch.Tensor,
        step: int,
        total_steps: int,
        sync_distributed: bool = True,
    ) -> float:
        """Update FP32 EMA load/energy buffers and decrement cooldown counters."""
        load_f32 = batch_load.detach().float().clone()
        energy_f32 = batch_energy.detach().float().clone()

        if sync_distributed and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(load_f32, op=dist.ReduceOp.AVG)
            dist.all_reduce(energy_f32, op=dist.ReduceOp.AVG)

        if self.cfg.router_mode == "deepseek_bias":
            fair_share = self.K_r / self.E
            err = load_f32 - fair_share
            self.deepseek_bal_bias.sub_(self.cfg.deepseek_bias_lr * torch.sign(err))

        alpha = self.cosine_alpha(step, total_steps)
        gamma = 1.0 - alpha

        self.ema_load.mul_(gamma).add_(load_f32, alpha=alpha)
        self.ema_energy.mul_(gamma).add_(energy_f32, alpha=alpha)
        self.cooldown.sub_(1).clamp_min_(0)
        return alpha
