"""Auxiliary-loss-free EMA MoE router plus Switch / DeepSeek-V3 baseline modes.

Top-K function preservation after a split
-----------------------------------------
Splitting donor d into siblings (d, c) with both biases at b_d - tau*ln2 keeps the *dense*
softmax mixture unchanged, but not a Top-K router: each sibling carries half the donor's
probability, so a token that picked d either keeps d with half the weight (K=1) or picks both
siblings and drops a different expert (K>=2).

Sibling routing removes that discontinuity. While (d, c) are merged, they are ranked as ONE
group with mass P = p_d + p_c. The token is dispatched to the sibling with the larger
probability and receives the group weight P (straight-through: the value is P, the gradient
flows through the chosen sibling's probability). The splitter places the sibling hyperplane
through the donor's EMA token centroid mu, so z_d - z_c = 2 r.(x - mu) / tau and
P = p_donor_before * cosh(r.(x - mu) / tau): routing and weights are preserved up to O(r^2).

A pair is un-merged once plain Top-K would move less than `unmerge_tol` of routing weight on
the pair's columns. This is measured on every batch (`pair_disagree`) and tracked as an EMA,
so the un-merge is itself a measured, bounded perturbation, not an assumption.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import EMeansMoEConfig


@dataclass
class RouterOutput:
    """Container returned by MoE router forward pass."""

    # (T, S) long: experts the tokens are dispatched to. S = K_r, except while a sibling pair is in
    # its un-merge ramp, when S = min(2 K_r, E) and unused slots hold -1 with weight 0.
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor   # (T, S) combine weights (input dtype)
    probs: torch.Tensor          # (T, E) fp32 softmax probabilities
    batch_load: torch.Tensor     # (E,) token selection fractions summing to K_r
    aux_loss: torch.Tensor       # scalar tensor (0.0 in e-means / deepseek_bias modes)
    z_loss: torch.Tensor         # scalar router z-loss
    batch_sum: Optional[torch.Tensor] = None      # (E, d) fp32: sum of dispatched x / T
    batch_sqnorm: Optional[torch.Tensor] = None   # (E,) fp32: sum of dispatched ||x||^2 / T
    pair_disagree: Optional[torch.Tensor] = None  # (E,) fp32: mean |dW| on a pair if un-merged


def _all_reduce_mean(flat: torch.Tensor) -> torch.Tensor:
    """Mean across ranks. ReduceOp.AVG is not supported by gloo, so SUM then divide."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
    return flat


class EMeansMoERouter(nn.Module):
    """
    Sparse Top-K_r MoE router tracking FP32 EMA utilization (N_j), energy, token centroid
    (S_j / N_j) and token spread per expert.

    Supports:
      - router_mode="emeans": pure LM loss + EMA tracking (no aux penalty) + dynamic splitting.
      - router_mode="unbalanced": pure LM loss baseline (demonstrates routing collapse).
      - router_mode="aux_loss": Switch / Mixtral differentiable auxiliary load-balancing loss.
      - router_mode="deepseek_bias": detached per-expert selection bias updated by sign(error).

    The router parameters and statistics are kept in fp32 even when the surrounding model is
    cast to bf16/fp16 (see `_apply`): the split offsets (r, r.mu) and the EMAs are far below
    bf16 resolution.
    """

    def __init__(self, d_model: int, cfg: EMeansMoEConfig):
        super().__init__()
        self.d_model = d_model
        self.cfg = cfg
        self.E = cfg.num_experts
        self.K_r = cfg.top_k
        self.tau_r = cfg.tau_r
        if not 1 <= self.K_r <= self.E:
            raise ValueError(f"top_k={self.K_r} must be in [1, num_experts={self.E}]")

        self.gate = nn.Linear(d_model, self.E, bias=True)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

        fair_share = self.K_r / self.E
        f32 = torch.float32
        self.register_buffer("ema_load", torch.full((self.E,), fair_share, dtype=f32))
        self.register_buffer("ema_energy", torch.zeros(self.E, dtype=f32))
        # e-means sufficient statistics of the tokens dispatched to each expert
        self.register_buffer("ema_sum", torch.zeros(self.E, d_model, dtype=f32))
        self.register_buffer("ema_sqnorm", torch.zeros(self.E, dtype=f32))
        self.register_buffer("cooldown", torch.zeros(self.E, dtype=torch.long))
        # Sibling groups: partner[j] == j for a singleton; partner[d] == c, partner[c] == d when merged
        self.register_buffer("partner", torch.arange(self.E, dtype=torch.long))
        self.register_buffer("expert_ids", torch.arange(self.E, dtype=torch.long), persistent=False)
        # EMA of pair_disagree (< 0 = not measured yet) and steps since the merge
        self.register_buffer("ema_pair_disagree", torch.zeros(self.E, dtype=f32))
        self.register_buffer("merge_age", torch.zeros(self.E, dtype=torch.long))
        # Un-merge ramp: lambda in (0, 1) while a pair moves from grouped to plain routing
        self.register_buffer("unmerge_lambda", torch.zeros(self.E, dtype=f32))
        self.register_buffer("unmerge_forced", torch.zeros(self.E, dtype=torch.bool))
        # Detached dynamic bias used only by the "deepseek_bias" baseline comparator
        self.register_buffer("deepseek_bal_bias", torch.zeros(self.E, dtype=f32))

    # ------------------------------------------------------------------ dtype guard
    def _apply(self, fn, *args, **kwargs):
        """Keep router parameters and statistics in fp32 across `model.to(dtype)` / `.half()`."""
        out = super()._apply(fn, *args, **kwargs)
        for mod in (self, self.gate):
            for t in list(mod._parameters.values()) + list(mod._buffers.values()):
                if t is not None and t.is_floating_point() and t.dtype != torch.float32:
                    t.data = t.data.float()
        return out

    # ------------------------------------------------------------------ helpers
    def cosine_alpha(self, step: int, total_steps: int) -> float:
        """Cosine decay from alpha_0 down to the non-stationary tracking floor alpha_min."""
        progress = min(max(step, 0), max(total_steps, 1)) / max(total_steps, 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.cfg.alpha_min + (self.cfg.alpha_0 - self.cfg.alpha_min) * cos_val)

    def merged_mask(self) -> torch.Tensor:
        return self.partner != self.expert_ids

    def num_groups(self) -> int:
        return self.E - int(self.merged_mask().sum().item()) // 2

    def centroid(self, j: int) -> torch.Tensor:
        """EMA mean of the tokens dispatched to expert j (S_j / N_j)."""
        n = float(self.ema_load[j].item())
        if n <= 1e-12:
            return torch.zeros(self.d_model, device=self.ema_sum.device)
        return self.ema_sum[j] / n

    def token_spread(self, j: int) -> float:
        """EMA total variance E||x - mu_j||^2 of the tokens dispatched to expert j (0 if unknown)."""
        n = float(self.ema_load[j].item())
        if n <= 1e-12:
            return 0.0
        mu = self.ema_sum[j] / n
        return max(float(self.ema_sqnorm[j].item()) / n - float(mu.dot(mu).item()), 0.0)

    def _renorm(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.renorm_topk_weights and self.K_r > 1:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w

    def _plain_topk(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w, idx = torch.topk(probs, k=self.K_r, dim=-1)
        return idx, self._renorm(w)

    def _sibling_topk(
        self, probs: torch.Tensor, partner: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-K over sibling groups. With no merged pairs this equals plain Top-K exactly."""
        partner = self.partner if partner is None else partner
        rep = torch.minimum(self.expert_ids, partner)                         # group id per expert
        group_mass = torch.zeros_like(probs).index_add_(1, rep, probs)        # mass at rep column
        group_mass = group_mass.masked_fill((rep != self.expert_ids).unsqueeze(0), -1.0)
        g_w, g_idx = torch.topk(group_mass, k=self.K_r, dim=-1)               # g_idx = rep ids
        alt = partner[g_idx]
        p_rep = probs.gather(1, g_idx)
        p_alt = probs.gather(1, alt)
        take_alt = p_alt > p_rep
        idx = torch.where(take_alt, alt, g_idx)
        p_chosen = torch.where(take_alt, p_alt, p_rep)
        paired = alt != g_idx
        # value = group mass P; gradient flows through the chosen sibling only (straight-through)
        w = torch.where(paired, p_chosen + (g_w - p_chosen).detach(), p_chosen)
        return idx, self._renorm(w)

    def _ramp_mix(
        self, probs: torch.Tensor, idx_g: torch.Tensor, w_g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blend the current grouped routing with its un-merge target.

        All pairs of a router that are ramping share one lambda (the splitter starts a ramp
        only when none is running). The target is the grouped routing with exactly those pairs
        un-merged (other merged pairs stay merged), so W = (1 - lambda) W_now + lambda W_target
        reaches the post-un-merge routing exactly at lambda = 1. Tokens that touch no ramping
        pair have W_target = W_now and are unaffected. Returns (T, S), S = min(2 K_r, E);
        empty slots are -1 with weight 0.
        """
        ramping = self.unmerge_lambda > 0
        lam = self.unmerge_lambda.max()
        target_partner = torch.where(ramping, self.expert_ids, self.partner)
        idx_t, w_t = self._sibling_topk(probs, target_partner)
        dense_g = torch.zeros_like(probs).scatter(1, idx_g, w_g)
        dense_t = torch.zeros_like(probs).scatter(1, idx_t, w_t)
        mixed = dense_g + lam * (dense_t - dense_g)
        w, idx = torch.topk(mixed, k=min(2 * self.K_r, self.E), dim=-1)
        idx = torch.where(w > 0, idx, torch.full_like(idx, -1))
        return idx, w

    @torch.no_grad()
    def _pair_disagreement(
        self, probs: torch.Tensor, idx_g: torch.Tensor, w_g: torch.Tensor
    ) -> torch.Tensor:
        """Per merged expert: mean over tokens of 1[pair involved] * sum_e |W_grouped - W_plain|.

        A token that selects no member of the pair under either scheme gets identical routing
        from both, so for one pair this bounds the un-merge's output change:
        mean_t ||y_plain - y_grouped|| <= pair_disagree * max_e ||h_e||. Tokens involving
        several pairs are counted for each pair (conservative).
        """
        idx_p, w_p = self._plain_topk(probs)
        dense_g = torch.zeros_like(probs).scatter_(1, idx_g, w_g.detach().float())
        dense_p = torch.zeros_like(probs).scatter_(1, idx_p, w_p.float())
        row_diff = (dense_g - dense_p).abs().sum(dim=-1, keepdim=True)            # (T, 1)
        member = torch.zeros_like(probs, dtype=torch.bool)
        member.scatter_(1, idx_g, True).scatter_(1, idx_p, True)                  # (T, E)
        involved = member | member[:, self.partner]
        per_pair = (involved.float() * row_diff).mean(dim=0)
        return torch.where(self.merged_mask(), per_pair, torch.zeros_like(per_pair))

    # ------------------------------------------------------------------ forward
    def forward(self, x_flat: torch.Tensor) -> RouterOutput:
        """
        Compute fp32 routing logits, Top-K_r assignments and baseline losses (if enabled).

        Args:
            x_flat: Flattened token hidden states of shape (T, d_model).
        """
        x32 = x_flat.float()
        dev_type = x_flat.device.type if x_flat.device.type in ("cuda", "cpu", "xpu", "mps") else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):   # autocast would run this in bf16
            logits = F.linear(x32, self.gate.weight.float(), self.gate.bias.float()) / self.tau_r
            probs = F.softmax(logits, dim=-1)

        pair_disagree = None
        if self.cfg.router_mode == "deepseek_bias":
            sel_scores = probs + self.deepseek_bal_bias.unsqueeze(0)
            _, topk_idx = torch.topk(sel_scores, k=self.K_r, dim=-1)
            topk_weights = self._renorm(torch.gather(probs, dim=-1, index=topk_idx))
        elif self.cfg.sibling_routing:
            topk_idx, topk_weights = self._sibling_topk(probs)
            pair_disagree = self._pair_disagreement(probs, topk_idx, topk_weights)
            if bool((self.unmerge_lambda > 0).any()):
                topk_idx, topk_weights = self._ramp_mix(probs, topk_idx, topk_weights)
        else:
            topk_idx, topk_weights = self._plain_topk(probs)

        with torch.no_grad():
            T = max(x32.shape[0], 1)
            valid = topk_idx >= 0                                             # -1 marks an empty slot
            dispatch = torch.zeros_like(probs).scatter_add_(1, topk_idx.clamp_min(0), valid.float())
            batch_load = dispatch.sum(dim=0) / T                             # sums to K_r
            batch_sum = batch_sqnorm = None
            if self.cfg.split_through_centroid:
                batch_sum = dispatch.t() @ x32 / T
                batch_sqnorm = dispatch.t() @ (x32 * x32).sum(dim=-1) / T

        # Optional Switch auxiliary loss baseline
        if self.cfg.router_mode == "aux_loss" and self.cfg.aux_loss_coef > 0.0:
            mean_probs = probs.mean(dim=0)
            # Standard Switch formulation normalized by K_r so uniform gives 1.0
            aux_loss = self.cfg.aux_loss_coef * (self.E / max(self.K_r, 1)) * torch.sum(batch_load * mean_probs)
        else:
            aux_loss = logits.new_zeros(())

        # Optional router z-loss
        if self.cfg.router_z_loss_coef > 0.0:
            z_loss = self.cfg.router_z_loss_coef * torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        else:
            z_loss = logits.new_zeros(())

        return RouterOutput(
            topk_idx=topk_idx,
            topk_weights=topk_weights.to(x_flat.dtype),
            probs=probs,
            batch_load=batch_load,
            aux_loss=aux_loss,
            z_loss=z_loss,
            batch_sum=batch_sum,
            batch_sqnorm=batch_sqnorm,
            pair_disagree=pair_disagree,
        )

    # ------------------------------------------------------------------ statistics
    @torch.no_grad()
    def update_ema(
        self,
        batch_load: torch.Tensor,
        batch_energy: torch.Tensor,
        step: int,
        total_steps: int,
        sync_distributed: bool = True,
        batch_sum: Optional[torch.Tensor] = None,
        batch_sqnorm: Optional[torch.Tensor] = None,
        pair_disagree: Optional[torch.Tensor] = None,
    ) -> float:
        """Update the fp32 EMA buffers (one packed all-reduce) and advance cooldown / merge age.

        Every rank must pass the same set of optional tensors (they follow from the shared
        config and the shared `partner` buffer, so they do).
        """
        E, d = self.E, self.d_model
        parts = [batch_load.detach().float().reshape(E), batch_energy.detach().float().reshape(E)]
        has_sum = batch_sum is not None and batch_sqnorm is not None
        has_dis = pair_disagree is not None
        if has_sum:
            parts += [batch_sum.detach().float().reshape(E * d), batch_sqnorm.detach().float().reshape(E)]
        if has_dis:
            parts.append(pair_disagree.detach().float().reshape(E))
        flat = torch.cat([p.to(self.ema_load.device) for p in parts])
        if sync_distributed:
            flat = _all_reduce_mean(flat)

        load_f32, energy_f32 = flat[:E], flat[E:2 * E]
        off = 2 * E
        if has_sum:
            sum_f32 = flat[off:off + E * d].view(E, d)
            sqn_f32 = flat[off + E * d:off + E * d + E]
            off += E * d + E
        if has_dis:
            dis_f32 = flat[off:off + E]

        if self.cfg.router_mode == "deepseek_bias":
            err = load_f32 - self.K_r / self.E
            self.deepseek_bal_bias.sub_(self.cfg.deepseek_bias_lr * torch.sign(err))

        alpha = self.cosine_alpha(step, total_steps)
        gamma = 1.0 - alpha
        self.ema_load.mul_(gamma).add_(load_f32, alpha=alpha)
        self.ema_energy.mul_(gamma).add_(energy_f32, alpha=alpha)
        if has_sum:
            self.ema_sum.mul_(gamma).add_(sum_f32, alpha=alpha)
            self.ema_sqnorm.mul_(gamma).add_(sqn_f32, alpha=alpha)

        merged = self.merged_mask()
        self.merge_age.add_(merged.long())
        if has_dis:
            beta = 2.0 / (max(self.cfg.unmerge_min_steps, 1) + 1.0)
            prev = self.ema_pair_disagree
            fresh = prev < 0
            ema = torch.where(fresh, dis_f32, (1.0 - beta) * prev + beta * dis_f32)
            self.ema_pair_disagree.copy_(torch.where(merged, ema, torch.zeros_like(ema)))

        self.cooldown.sub_(1).clamp_min_(0)
        return alpha
