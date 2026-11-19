"""Routing health metrics shared across NumPy simulations and PyTorch runs."""

from typing import Dict
import numpy as np


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().to(dtype=None).numpy().astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def compute_routing_health(load_vec, top_k: int, dead_tau: float = 0.1) -> Dict[str, float]:
    """
    Compute diagnostic load-balancing and utilization metrics from expert selection loads.

    Args:
        load_vec: Length-E vector of per-expert selection fractions (expected sum ~= top_k).
        top_k: Active experts selected per token (K_r).
        dead_tau: Fraction of fair share (K_r / E) below which an expert is flagged dead.
    """
    loads = np.maximum(_to_numpy(load_vec).ravel(), 0.0)
    num_experts = loads.size
    fair_share = top_k / max(num_experts, 1)

    max_to_mean = float(loads.max() / max(fair_share, 1e-12))
    min_to_fair = float(loads.min() / max(fair_share, 1e-12))
    dead_count = int(np.sum(loads < (dead_tau * fair_share)))
    active_frac = float(1.0 - (dead_count / max(num_experts, 1)))

    # Normalized Shannon entropy over normalized routing distribution (0 = collapsed, 1 = uniform)
    total = float(loads.sum())
    if total > 1e-12 and num_experts > 1:
        p = loads / total
        nz = p[p > 1e-15]
        entropy = -float(np.sum(nz * np.log(nz)))
        norm_entropy = entropy / float(np.log(num_experts))
    else:
        norm_entropy = 0.0

    # Gini coefficient (0 = perfectly equal, 1 = maximally unequal)
    if total > 1e-12 and num_experts > 1:
        sorted_l = np.sort(loads)
        idx = np.arange(1, num_experts + 1, dtype=np.float64)
        gini = float((2.0 * np.sum(idx * sorted_l)) / (num_experts * total) - (num_experts + 1.0) / num_experts)
    else:
        gini = 1.0

    return {
        "fair_share": float(fair_share),
        "max_to_mean": max_to_mean,
        "min_to_fair": min_to_fair,
        "dead_count": float(dead_count),
        "active_fraction": active_frac,
        "norm_entropy": norm_entropy,
        "gini": max(0.0, gini),
    }
