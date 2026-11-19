"""Log-space OLS power-law fitter for non-stationary tracking floor alpha_min."""

from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np


@dataclass
class ScalingObservation:
    """Single empirical sweep optimum for alpha_min at given (BL, E, K_r, tau_r)."""

    tokens_per_step: float  # BL
    num_experts: float      # E
    top_k: float            # K_r
    tau_r: float            # router temperature
    alpha_min_opt: float    # empirically best tracking floor


@dataclass
class PowerLawFit:
    """Fitted power law: alpha_min = C * BL^a_B * E^a_E * K_r^a_K * tau_r^a_tau."""

    c_alpha: float
    a_bl: float
    a_e: float
    a_k: float
    a_tau: float
    r2_log: float

    def predict(self, tokens_per_step: float, num_experts: float, top_k: float, tau_r: float) -> float:
        val = (
            self.c_alpha
            * (tokens_per_step ** self.a_bl)
            * (num_experts ** self.a_e)
            * (top_k ** self.a_k)
            * (tau_r ** self.a_tau)
        )
        return float(np.clip(val, 1e-6, 0.99))


def fit_alpha_min_powerlaw(observations: Sequence[ScalingObservation]) -> PowerLawFit:
    """Fit log-linear OLS regression predicting alpha_min from proxy sweeps."""
    if len(observations) < 5:
        raise ValueError("Need at least 5 observations to fit 5 power-law parameters.")

    X = np.array(
        [
            [
                1.0,
                np.log(o.tokens_per_step),
                np.log(o.num_experts),
                np.log(o.top_k),
                np.log(o.tau_r),
            ]
            for o in observations
        ],
        dtype=np.float64,
    )
    y = np.array([np.log(o.alpha_min_opt) for o in observations], dtype=np.float64)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

    return PowerLawFit(
        c_alpha=float(np.exp(beta[0])),
        a_bl=float(beta[1]),
        a_e=float(beta[2]),
        a_k=float(beta[3]),
        a_tau=float(beta[4]),
        r2_log=float(r2),
    )


def evaluate_loso(observations: Sequence[ScalingObservation]) -> Dict[str, float]:
    """Leave-One-Scale-Out (LOSO) log-RMSE validation across expert counts E."""
    unique_e = sorted({o.num_experts for o in observations})
    log_sq_errors: List[float] = []
    rel_errors: List[float] = []

    for held_e in unique_e:
        train_obs = [o for o in observations if o.num_experts != held_e]
        test_obs = [o for o in observations if o.num_experts == held_e]
        if len(train_obs) < 5 or not test_obs:
            continue
        fit = fit_alpha_min_powerlaw(train_obs)
        for o in test_obs:
            pred = fit.predict(o.tokens_per_step, o.num_experts, o.top_k, o.tau_r)
            log_sq_errors.append(float((np.log(pred) - np.log(o.alpha_min_opt)) ** 2))
            rel_errors.append(float(abs(pred - o.alpha_min_opt) / o.alpha_min_opt))

    return {
        "loso_log_rmse": float(np.sqrt(np.mean(log_sq_errors))) if log_sq_errors else float("nan"),
        "loso_mean_rel_err": float(np.mean(rel_errors)) if rel_errors else float("nan"),
        "n_folds": float(len(unique_e)),
    }
