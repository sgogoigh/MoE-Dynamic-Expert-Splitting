"""Dynamic expert splitting: function-preserving donor -> dead split under Top-K routing.

A split copies the donor FFN into the dead slot, shifts both router biases by -tau_r*ln2,
and opens a hyperplane between the siblings through the donor's EMA token centroid:

    donor:  W + r,  b - tau*ln2 - r.mu        z_d = z - ln2 + r.(x - mu)/tau
    clone:  W - r,  b - tau*ln2 + r.mu        z_c = z - ln2 - r.(x - mu)/tau

so the pair's softmax mass is p_donor * cosh(r.(x - mu)/tau) (second order in r). With
`sibling_routing` the router ranks the pair as one Top-K group until plain Top-K would move
less than `unmerge_tol` routing weight on it (see router.py), so the split preserves the
Top-K MoE output, not only the dense mixture. `|r|` is set from the donor's EMA token spread
so the typical sibling logit gap at split time is `router_noise_eps` (scale-free).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .experts import GeGLUMoEExperts
from .router import EMeansMoERouter


@dataclass
class SplitEvent:
    """Record of a single donor -> dead expert split."""

    step: int
    layer_name: str
    dead_idx: int
    donor_idx: int
    dead_load_before: float
    donor_load_before: float
    donor_energy_before: float
    sibling_merged: bool = False
    router_offset_norm: float = 0.0


@dataclass
class UnmergeEvent:
    """Record of a sibling pair returning to independent Top-K routing."""

    step: int
    layer_name: str
    expert_a: int
    expert_b: int
    ema_disagreement: float   # EMA routing weight that un-merging moves on the pair's columns
    forced: bool              # True when merge_max_steps was hit before reaching unmerge_tol
    merge_age: int


def _orthogonal_perturbation(donor_tensor: torch.Tensor, eps: float, gen: torch.Generator) -> torch.Tensor:
    """
    Gaussian noise orthogonal to donor_tensor (Gram-Schmidt) with norm eps * ||donor_tensor||.
    Preserves the donor weight norm to first order.
    """
    if eps <= 0.0:
        return torch.zeros_like(donor_tensor)
    raw = torch.randn(donor_tensor.shape, device=donor_tensor.device, dtype=torch.float32, generator=gen)
    d_f32 = donor_tensor.float()
    d_norm2 = torch.sum(d_f32 * d_f32).clamp_min(1e-12)
    proj = (torch.sum(raw * d_f32) / d_norm2) * d_f32
    ortho = raw - proj
    ortho_norm = torch.norm(ortho).clamp_min(1e-12)
    target_mag = eps * torch.sqrt(d_norm2)
    return (target_mag * (ortho / ortho_norm)).to(donor_tensor.dtype)


def _unit_direction(reference: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random fp32 unit vector, orthogonal to `reference` when `reference` is non-zero."""
    raw = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=gen)
    ref = reference.float()
    ref_n2 = float(ref.dot(ref).item())
    if ref_n2 > 1e-24:
        raw = raw - (raw.dot(ref) / ref_n2) * ref
    return raw / raw.norm().clamp_min(1e-12)


def _migrate_adam_slice(
    optimizer: Optional[torch.optim.Optimizer],
    param: nn.Parameter,
    donor_idx: int,
    dead_idx: int,
    halve_exp_avg: bool = False,
) -> None:
    """Copy AdamW exp_avg_sq (v) and exp_avg (m) from donor slice to revived dead slice."""
    if optimizer is None or param not in optimizer.state:
        return
    st = optimizer.state[param]
    if "exp_avg" in st and st["exp_avg"] is not None:
        val = st["exp_avg"][donor_idx].clone()
        if halve_exp_avg:
            val.mul_(0.5)
            st["exp_avg"][donor_idx].mul_(0.5)
        st["exp_avg"][dead_idx].copy_(val)
    if "exp_avg_sq" in st and st["exp_avg_sq"] is not None:
        # Preserve denominator preconditioner 1/sqrt(v) directly to avoid LR spike on dead slot
        st["exp_avg_sq"][dead_idx].copy_(st["exp_avg_sq"][donor_idx])


class DynamicExpertSplitter:
    """
    Periodically inspects EMeansMoERouter EMA load buffers, splits overloaded donor experts
    into starving slots (function-preserving under Top-K with sibling routing), un-merges
    sibling pairs once separable, and keeps ranks and AdamW moments synchronized.
    """

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    @torch.no_grad()
    def maybe_split_layer(
        self,
        router: EMeansMoERouter,
        experts: GeGLUMoEExperts,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        layer_name: str = "layer",
    ) -> List[SplitEvent]:
        """Check dead threshold and execute up to max_splits_per_step donor splits."""
        cfg = router.cfg
        if cfg.router_mode != "emeans":
            return []
        if step <= 0 or (step % max(cfg.split_interval, 1) != 0):
            return []

        is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if is_dist else 0

        split_plan: List[Tuple[int, int, int]] = []
        if rank == 0:
            split_plan = self._select_split_pairs(router, step)

        if is_dist:
            obj_box = [split_plan]
            dist.broadcast_object_list(obj_box, src=0)
            split_plan = obj_box[0]

        events: List[SplitEvent] = []
        for dead_idx, donor_idx, pair_seed in split_plan:
            dead_before = float(router.ema_load[dead_idx].item())
            donor_before = float(router.ema_load[donor_idx].item())
            donor_e_before = float(router.ema_energy[donor_idx].item())

            r_norm = self._execute_single_split(router, experts, optimizer, dead_idx, donor_idx, pair_seed)
            events.append(
                SplitEvent(
                    step=step,
                    layer_name=layer_name,
                    dead_idx=dead_idx,
                    donor_idx=donor_idx,
                    dead_load_before=dead_before,
                    donor_load_before=donor_before,
                    donor_energy_before=donor_e_before,
                    sibling_merged=bool(cfg.sibling_routing),
                    router_offset_norm=r_norm,
                )
            )
        return events

    def _select_split_pairs(self, router: EMeansMoERouter, step: int) -> List[Tuple[int, int, int]]:
        cfg = router.cfg
        fair_share = router.K_r / router.E
        free = router.cooldown == 0
        if cfg.sibling_routing:
            free = free & ~router.merged_mask()   # merged siblings are neither dead nor donors
        dead_mask = (router.ema_load < cfg.split_tau * fair_share) & free
        if not dead_mask.any():
            return []

        dead_ids = torch.nonzero(dead_mask, as_tuple=False).flatten().tolist()
        # Sort most starved experts first
        dead_ids.sort(key=lambda idx: float(router.ema_load[idx].item()))
        max_new = cfg.max_splits_per_step
        if cfg.sibling_routing:
            # every merge removes one routing group; Top-K needs at least K_r groups
            max_new = min(max_new, router.num_groups() - router.K_r)
        if max_new <= 0:
            return []
        dead_ids = dead_ids[:max_new]

        donor_mask = (~dead_mask) & (router.ema_load >= cfg.donor_min_mult * fair_share) & free
        if not donor_mask.any():
            # Fallback: allow highest active expert exceeding fair share
            donor_mask = (~dead_mask) & (router.ema_load > fair_share) & free
        if not donor_mask.any():
            return []

        donor_candidates = torch.nonzero(donor_mask, as_tuple=False).flatten()
        if cfg.donor_weight_by == "energy" and float(router.ema_energy[donor_candidates].sum().item()) > 1e-12:
            weights = router.ema_energy[donor_candidates].clamp_min(1e-12)
        else:
            weights = router.ema_load[donor_candidates].clamp_min(1e-12)

        n_pairs = min(len(dead_ids), donor_candidates.numel())
        gen_cpu = torch.Generator(device="cpu").manual_seed(self.base_seed + step * 997 + len(dead_ids))
        chosen_pos = torch.multinomial(weights.float().cpu(), num_samples=n_pairs, replacement=False, generator=gen_cpu).tolist()

        plan: List[Tuple[int, int, int]] = []
        for k in range(n_pairs):
            d_dead = int(dead_ids[k])
            d_donor = int(donor_candidates[chosen_pos[k]].item())
            pair_seed = self.base_seed + step * 1009 + k * 37
            plan.append((d_dead, d_donor, pair_seed))
        return plan

    def _router_offset(self, router: EMeansMoERouter, donor_idx: int, gen: torch.Generator) -> torch.Tensor:
        """Sibling hyperplane normal r, scaled so the typical |r.(x - mu)| / tau is router_noise_eps."""
        cfg = router.cfg
        donor_row = router.gate.weight[donor_idx]
        if cfg.router_noise_eps <= 0.0:
            return torch.zeros_like(donor_row, dtype=torch.float32)
        direction = _unit_direction(donor_row, gen)
        spread = router.token_spread(donor_idx)          # E||x - mu||^2 over the donor's tokens
        if spread > 0.0:
            # projection of (x - mu) on a random unit direction has std ~ sqrt(spread / d)
            proj_std = math.sqrt(spread / router.d_model)
            return direction * (cfg.router_noise_eps * router.tau_r / proj_std)
        # no token statistics yet: fall back to a perturbation relative to the row norm
        return direction * cfg.router_noise_eps * float(donor_row.float().norm().clamp_min(1e-6).item())

    @torch.no_grad()
    def _execute_single_split(
        self,
        router: EMeansMoERouter,
        experts: GeGLUMoEExperts,
        optimizer: Optional[torch.optim.Optimizer],
        dead_idx: int,
        donor_idx: int,
        pair_seed: int,
    ) -> float:
        cfg = router.cfg
        dev = experts.gate_proj.device
        gen = torch.Generator(device=dev).manual_seed(pair_seed)

        # 1. Clone GeGLU FFN weights (+delta / -delta; delta = 0 by default keeps the copy exact)
        for p in (experts.gate_proj, experts.up_proj, experts.down_proj):
            donor_w = p[donor_idx].clone()
            delta = _orthogonal_perturbation(donor_w, cfg.orthogonal_eps, gen)
            p[donor_idx].copy_(donor_w + delta)
            p[dead_idx].copy_(donor_w - delta)
            _migrate_adam_slice(optimizer, p, donor_idx, dead_idx, halve_exp_avg=False)

        # 2. Router: rows W +/- r; biases b - tau*ln2 -/+ r.mu (hyperplane through the centroid)
        r_gen = gen if router.gate.weight.device == dev else torch.Generator(device=router.gate.weight.device).manual_seed(pair_seed)
        r = self._router_offset(router, donor_idx, r_gen)
        mu = router.centroid(donor_idx) if cfg.split_through_centroid else torch.zeros_like(r)
        r_dot_mu = float(r.dot(mu.to(r.device)).item())

        donor_row = router.gate.weight[donor_idx].float().clone()
        router.gate.weight[donor_idx].copy_(donor_row + r)
        router.gate.weight[dead_idx].copy_(donor_row - r)
        _migrate_adam_slice(optimizer, router.gate.weight, donor_idx, dead_idx, halve_exp_avg=cfg.halve_router_momentum)

        if router.gate.bias is not None:
            shifted = float(router.gate.bias[donor_idx].item()) - router.tau_r * math.log(2.0)
            router.gate.bias[donor_idx] = shifted - r_dot_mu
            router.gate.bias[dead_idx] = shifted + r_dot_mu
            _migrate_adam_slice(optimizer, router.gate.bias, donor_idx, dead_idx, halve_exp_avg=cfg.halve_router_momentum)

        # 3. Halve the EMA sufficient statistics across siblings (centroid and spread unchanged)
        for buf in (router.ema_load, router.ema_energy, router.ema_sqnorm):
            half = 0.5 * float(buf[donor_idx].item())
            buf[donor_idx] = half
            buf[dead_idx] = half
        router.ema_sum[donor_idx].mul_(0.5)
        router.ema_sum[dead_idx].copy_(router.ema_sum[donor_idx])

        router.cooldown[donor_idx] = cfg.cooldown_steps
        router.cooldown[dead_idx] = cfg.cooldown_steps

        # 4. Merge the siblings into one routing group until they are separable
        if cfg.sibling_routing:
            router.partner[donor_idx] = dead_idx
            router.partner[dead_idx] = donor_idx
            for j in (donor_idx, dead_idx):
                router.ema_pair_disagree[j] = -1.0
                router.merge_age[j] = 0
                router.unmerge_lambda[j] = 0.0
                router.unmerge_forced[j] = False
        return float(r.norm().item())

    @torch.no_grad()
    def maybe_unmerge_layer(self, router: EMeansMoERouter, step: int, layer_name: str = "layer") -> List[UnmergeEvent]:
        """Advance un-merge ramps and start / finish un-merges. Call once per step after update_ema.

        - separable pair (EMA disagreement < unmerge_tol after unmerge_min_steps): un-merged at
          once; the output change is bounded by the measured disagreement.
        - pair merged for merge_max_steps: forced; routing is ramped from grouped to plain Top-K
          over unmerge_ramp_steps (see router._ramp_mix), then un-merged.
        Events are returned when a pair is actually un-merged. The decision reads only
        all-reduced buffers, so every rank takes the same decision without communication.
        """
        cfg = router.cfg
        merged = router.merged_mask()
        if not bool(merged.any()):
            return []
        ramp = max(int(cfg.unmerge_ramp_steps), 0)
        # one ramp at a time per router: pairs forced while a ramp runs wait for it to finish,
        # so every ramping pair shares one lambda and the blended routing is exact
        ramp_running = bool((router.unmerge_lambda > 0).any())
        events: List[UnmergeEvent] = []
        for a in torch.nonzero(merged, as_tuple=False).flatten().tolist():
            b = int(router.partner[a].item())
            if b < a:
                continue
            age = int(router.merge_age[a].item())
            dis = float(router.ema_pair_disagree[a].item())
            lam = float(router.unmerge_lambda[a].item())
            if lam > 0.0:                                   # ramp in progress
                lam = lam + 1.0 / max(ramp, 1)
                if lam < 1.0 - 1e-6:
                    router.unmerge_lambda[a] = router.unmerge_lambda[b] = lam
                    continue
                forced = bool(router.unmerge_forced[a].item())
            else:
                separable = age >= cfg.unmerge_min_steps and 0.0 <= dis < cfg.unmerge_tol
                forced = (not separable) and cfg.merge_max_steps > 0 and age >= cfg.merge_max_steps
                if not (separable or forced):
                    continue
                if forced and ramp > 1:                     # start the ramp; finish later
                    if ramp_running:
                        continue
                    router.unmerge_lambda[a] = router.unmerge_lambda[b] = 1.0 / ramp
                    router.unmerge_forced[a] = router.unmerge_forced[b] = True
                    continue
            self._finish_unmerge(router, a, b)
            events.append(UnmergeEvent(step, layer_name, a, b, dis, forced, age))
        return events

    @staticmethod
    def _finish_unmerge(router: EMeansMoERouter, a: int, b: int) -> None:
        for j in (a, b):
            router.partner[j] = j
            router.ema_pair_disagree[j] = 0.0
            router.merge_age[j] = 0
            router.unmerge_lambda[j] = 0.0
            router.unmerge_forced[j] = False
