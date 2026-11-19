"""
Phase A Scaling-Law Sweep Planner & Offline OLS Fitter Check.
Prints the proxy experimental grid for alpha_min = C * BL^aB * E^aE * K_r^aK * tau_r^atau
and verifies fit_alpha_min_powerlaw + LOSO validation on synthetic proxy sweeps.
"""

import itertools
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.scaling_law import (  # noqa: E402
    ScalingObservation,
    evaluate_loso,
    fit_alpha_min_powerlaw,
)


def main():
    grid_bl = [131072, 262144, 524288]        # 128k, 256k, 512k tokens/step
    grid_e = [8, 16, 32, 64]
    grid_k = [1, 2, 4]
    grid_tau = [0.5, 1.0, 2.0]

    combos = list(itertools.product(grid_bl, grid_e, grid_k, grid_tau))
    # Prune invalid K_r >= E if any
    combos = [(bl, e, k, t) for (bl, e, k, t) in combos if k < e]

    print(f"Planned Phase A Proxy Sweep Matrix: {len(combos)} configurations")
    manifest_out = ROOT / "configs" / "phase_a_sweep_manifest.json"
    manifest_out.write_text(
        json.dumps(
            [
                {"tokens_per_step": bl, "num_experts": e, "top_k": k, "tau_r": t}
                for (bl, e, k, t) in combos
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote dry-run sweep manifest -> {manifest_out}")

    # Synthetic verification of OLS log-power-law recovery + LOSO validation
    rng = np.random.default_rng(42)
    true_c, true_b, true_e, true_k, true_t = 0.008, 0.42, -0.58, 0.35, -0.20
    obs = []
    for bl, e, k, t in combos:
        y_true = true_c * (bl ** true_b) * (e ** true_e) * (k ** true_k) * (t ** true_t)
        noisy = y_true * float(np.exp(rng.normal(scale=0.05)))
        obs.append(ScalingObservation(bl, e, k, t, noisy))

    fit = fit_alpha_min_powerlaw(obs)
    loso = evaluate_loso(obs)
    print("\n[Synthetic OLS Power-Law Recovery Verification]")
    print(f"  True Law  : C={true_c:.4f}, a_B={true_b:+.2f}, a_E={true_e:+.2f}, a_K={true_k:+.2f}, a_tau={true_t:+.2f}")
    print(f"  Fitted Law: C={fit.c_alpha:.4f}, a_B={fit.a_bl:+.2f}, a_E={fit.a_e:+.2f}, a_K={fit.a_k:+.2f}, a_tau={fit.a_tau:+.2f}")
    print(f"  R^2 (log) = {fit.r2_log:.5f} | LOSO log-RMSE = {loso['loso_log_rmse']:.5f} ({loso['loso_mean_rel_err']*100:.2f}% rel err)")


if __name__ == "__main__":
    main()
