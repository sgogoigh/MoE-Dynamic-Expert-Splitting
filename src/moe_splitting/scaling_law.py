"""Log-space OLS power-law fitter for the non-stationary tracking floor alpha_min.

    alpha_min = C * BL^a_bl * E^a_e * K^a_k * tau^a_tau

Identifiability is checked, not assumed:
  - a feature that is constant across the observations has no identifiable exponent; it is
    dropped from the regression and reported in `unidentified` (its exponent is NaN and
    `predict` refuses other values of it);
  - collinear features (rank-deficient design) raise ValueError instead of returning the
    arbitrary minimum-norm solution of lstsq;
  - at least one residual degree of freedom is required, otherwise R^2 is meaningless.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURES: Tuple[str, ...] = ("tokens_per_step", "num_experts", "top_k", "tau_r")
_EXPONENT_NAME = {"tokens_per_step": "a_bl", "num_experts": "a_e", "top_k": "a_k", "tau_r": "a_tau"}


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
    """Fitted power law: alpha_min = C * BL^a_bl * E^a_e * K_r^a_k * tau_r^a_tau."""

    c_alpha: float
    a_bl: float
    a_e: float
    a_k: float
    a_tau: float
    r2_log: float
    n_obs: int = 0
    dof: int = 0
    unidentified: Tuple[str, ...] = ()
    fixed_values: Dict[str, float] = field(default_factory=dict)

    def predict(self, tokens_per_step: float, num_experts: float, top_k: float, tau_r: float) -> float:
        values = {"tokens_per_step": tokens_per_step, "num_experts": num_experts, "top_k": top_k, "tau_r": tau_r}
        log_val = math.log(self.c_alpha)
        for name in FEATURES:
            if name in self.unidentified:
                fixed = self.fixed_values[name]
                if not math.isclose(values[name], fixed, rel_tol=1e-9):
                    raise ValueError(f"{name} was constant ({fixed}) in the fit; its exponent is not identified")
                continue
            log_val += getattr(self, _EXPONENT_NAME[name]) * math.log(values[name])
        return float(np.clip(math.exp(log_val), 1e-6, 0.99))


def fit_alpha_min_powerlaw(observations: Sequence[ScalingObservation]) -> PowerLawFit:
    """Fit a log-linear OLS regression predicting alpha_min, with identifiability checks."""
    if not observations:
        raise ValueError("No observations.")
    cols = {name: np.array([math.log(getattr(o, name)) for o in observations]) for name in FEATURES}
    unidentified = tuple(name for name in FEATURES if np.ptp(cols[name]) < 1e-12)
    used = [name for name in FEATURES if name not in unidentified]
    n_params = 1 + len(used)
    if len(observations) < n_params + 1:
        raise ValueError(
            f"Need at least {n_params + 1} observations for {n_params} parameters "
            f"(identified features: {used}); got {len(observations)}."
        )
    X = np.column_stack([np.ones(len(observations))] + [cols[name] for name in used])
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise ValueError(f"Collinear design: rank {rank} < {X.shape[1]} columns for features {used}.")
    y = np.array([math.log(o.alpha_min_opt) for o in observations])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    exps = {name: float("nan") for name in FEATURES}
    for i, name in enumerate(used):
        exps[name] = float(beta[1 + i])
    return PowerLawFit(
        c_alpha=float(np.exp(beta[0])),
        a_bl=exps["tokens_per_step"],
        a_e=exps["num_experts"],
        a_k=exps["top_k"],
        a_tau=exps["tau_r"],
        r2_log=float(r2),
        n_obs=len(observations),
        dof=len(observations) - n_params,
        unidentified=unidentified,
        fixed_values={name: float(getattr(observations[0], name)) for name in unidentified},
    )


def evaluate_loso(observations: Sequence[ScalingObservation]) -> Dict[str, object]:
    """Leave-One-Scale-Out (LOSO) validation across expert counts E.

    A fold is skipped (with the reason recorded) when its training set cannot be fitted or
    cannot predict the held-out scale, e.g. when only one other E value remains, which leaves
    the E exponent unidentified.
    """
    unique_e = sorted({o.num_experts for o in observations})
    log_sq_errors: List[float] = []
    rel_errors: List[float] = []
    skipped: List[Dict[str, object]] = []

    for held_e in unique_e:
        train_obs = [o for o in observations if o.num_experts != held_e]
        test_obs = [o for o in observations if o.num_experts == held_e]
        try:
            fit = fit_alpha_min_powerlaw(train_obs)
            preds = [fit.predict(o.tokens_per_step, o.num_experts, o.top_k, o.tau_r) for o in test_obs]
        except ValueError as exc:
            skipped.append({"held_out_E": held_e, "reason": str(exc)})
            continue
        for o, pred in zip(test_obs, preds):
            log_sq_errors.append(float((np.log(pred) - np.log(o.alpha_min_opt)) ** 2))
            rel_errors.append(float(abs(pred - o.alpha_min_opt) / o.alpha_min_opt))

    return {
        "loso_log_rmse": float(np.sqrt(np.mean(log_sq_errors))) if log_sq_errors else float("nan"),
        "loso_mean_rel_err": float(np.mean(rel_errors)) if rel_errors else float("nan"),
        "n_folds": float(len(unique_e)),
        "n_folds_evaluated": len(unique_e) - len(skipped),
        "skipped": skipped,
    }
