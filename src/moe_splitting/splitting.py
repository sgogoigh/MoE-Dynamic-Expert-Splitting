"""Dynamic Expert Splitting engine with exact -tau_r ln(2) bias invariance & AdamW sync."""

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


def _orthogonal_perturbation(donor_tensor: torch.Tensor, eps: float, gen: torch.Generator) -> torch.Tensor:
    """
    Sample Gaussian noise orthogonal to donor_tensor (Gram-Schmidt projection).
    Preserves donor weight norm to first order while breaking gradient symmetry.
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
    Periodically inspects EMeansMoERouter EMA load buffers, splits overloaded donor
    experts into starving slots with function-preserving bias shift (-tau_r * ln 2),
    and keeps distributed ranks & AdamW moments synchronized.
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

        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if is_dist else 0

        split_plan: List[Tuple[int, int, int]] = []
        if rank == 0:
            split_plan = self._select_split_pairs(router, step)

        if is_dist:
            obj_box = [split_plan]
            dist.broadcast_object_list(obj_box, src=0)
            split_plan = obj_box[0]

        if not split_plan:
            return []

        events: List[SplitEvent] = []
        for dead_idx, donor_idx, pair_seed in split_plan:
            dead_before = float(router.ema_load[dead_idx].item())
            donor_before = float(router.ema_load[donor_idx].item())
            donor_e_before = float(router.ema_energy[donor_idx].item())

            self._execute_single_split(router, experts, optimizer, dead_idx, donor_idx, pair_seed)
            events.append(
                SplitEvent(
                    step=step,
                    layer_name=layer_name,
                    dead_idx=dead_idx,
                    donor_idx=donor_idx,
                    dead_load_before=dead_before,
                    donor_load_before=donor_before,
                    donor_energy_before=donor_e_before,
                )
            )
        return events

    def _select_split_pairs(self, router: EMeansMoERouter, step: int) -> List[Tuple[int, int, int]]:
        cfg = router.cfg
        fair_share = router.K_r / router.E
        dead_mask = (router.ema_load < cfg.split_tau * fair_share) & (router.cooldown == 0)
        if not dead_mask.any():
            return []

        dead_ids = torch.nonzero(dead_mask, as_tuple=False).flatten().tolist()
        # Sort most starved experts first
        dead_ids.sort(key=lambda idx: float(router.ema_load[idx].item()))
        dead_ids = dead_ids[: cfg.max_splits_per_step]

        donor_mask = (~dead_mask) & (router.ema_load >= cfg.donor_min_mult * fair_share) & (router.cooldown == 0)
        if not donor_mask.any():
            # Fallback: allow highest active expert exceeding fair share
            donor_mask = (~dead_mask) & (router.ema_load > fair_share)
        if not donor_mask.any():
            return []

        donor_candidates = torch.nonzero(donor_mask, as_tuple=False).flatten()
        if cfg.donor_weight_by == "energy" and float(router.ema_energy[donor_candidates].sum().item()) > 1e-12:
            weights = router.ema_energy[donor_candidates].clamp_min(1e-12)
        else:
            weights = router.ema_load[donor_candidates].clamp_min(1e-12)

        n_pairs = min(len(dead_ids), donor_candidates.numel())
        gen_cpu = torch.Generator(device="cpu").manual_seed(self.base_seed + step * 997 + len(dead_ids))
        chosen_pos = torch.multinomial(weights.cpu(), num_samples=n_pairs, replacement=False, generator=gen_cpu).tolist()

        plan: List[Tuple[int, int, int]] = []
        for k in range(n_pairs):
            d_dead = int(dead_ids[k])
            d_donor = int(donor_candidates[chosen_pos[k]].item())
            pair_seed = self.base_seed + step * 1009 + k * 37
            plan.append((d_dead, d_donor, pair_seed))
        return plan

    @torch.no_grad()
    def _execute_single_split(
        self,
        router: EMeansMoERouter,
        experts: GeGLUMoEExperts,
        optimizer: Optional[torch.optim.Optimizer],
        dead_idx: int,
        donor_idx: int,
        pair_seed: int,
    ) -> None:
        cfg = router.cfg
        dev = experts.gate_proj.device
        gen = torch.Generator(device=dev).manual_seed(pair_seed)

        # 1. Clone GeGLU FFN weights + symmetric orthogonal perturbation (+delta / -delta)
        for p in (experts.gate_proj, experts.up_proj, experts.down_proj):
            donor_w = p[donor_idx].clone()
            delta = _orthogonal_perturbation(donor_w, cfg.orthogonal_eps, gen)
            p[donor_idx].copy_(donor_w + delta)
            p[dead_idx].copy_(donor_w - delta)
            _migrate_adam_slice(optimizer, p, donor_idx, dead_idx, halve_exp_avg=False)

        # 2. Clone router row + tiny orthogonal break; shift BOTH biases by -tau_r * ln(2)
        donor_row = router.gate.weight[donor_idx].clone()
        r_delta = _orthogonal_perturbation(donor_row, cfg.router_noise_eps, gen)
        router.gate.weight[donor_idx].copy_(donor_row + r_delta)
        router.gate.weight[dead_idx].copy_(donor_row - r_delta)
        _migrate_adam_slice(optimizer, router.gate.weight, donor_idx, dead_idx, halve_exp_avg=cfg.halve_router_momentum)

        if router.gate.bias is not None:
            shifted_bias = router.gate.bias[donor_idx].item() - (router.tau_r * math.log(2.0))
            router.gate.bias[donor_idx] = shifted_bias
            router.gate.bias[dead_idx] = shifted_bias
            _migrate_adam_slice(optimizer, router.gate.bias, donor_idx, dead_idx, halve_exp_avg=cfg.halve_router_momentum)

        # 3. Halve EMA load & energy across siblings and set cooldown window
        half_load = 0.5 * router.ema_load[donor_idx].item()
        half_energy = 0.5 * router.ema_energy[donor_idx].item()
        router.ema_load[donor_idx] = half_load
        router.ema_load[dead_idx] = half_load
        router.ema_energy[donor_idx] = half_energy
        router.ema_energy[dead_idx] = half_energy

        router.cooldown[donor_idx] = cfg.cooldown_steps
        router.cooldown[dead_idx] = cfg.cooldown_steps
