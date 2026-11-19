"""
Offline Exploratory Mini-Run (Pure NumPy):
1. Proves exact -tau_r ln(2) softmax mass invariance numerically across temperatures.
2. Simulates 400 steps of non-stationary multi-domain MoE routing comparing:
   - Unbalanced Top-K_r (rich-get-richer collapse)
   - Aux-Loss Baseline
   - DeepSeek-V3 Sign Bias Baseline
   - Ours: e-means EMA + Dynamic Donor Splitting
"""

import math
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.metrics import compute_routing_health  # noqa: E402


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def verify_exact_bias_invariance():
    print("\n[1] NUMERICAL PROOF: Softmax Probability Invariance via b <- b - tau_r * ln(2)")
    print("-" * 78)
    rng = np.random.default_rng(7)
    T, d, E = 64, 32, 16
    X = rng.normal(size=(T, d))
    W_r = rng.normal(scale=0.35, size=(E, d))
    b = rng.normal(scale=0.5, size=(E,))

    # Pretend expert j* = 3 is overloaded and expert j_dead = 11 is completely dead (b = -1e9)
    j_donor, j_dead = 3, 11
    b[j_donor] = 1.8
    b[j_dead] = -1e9

    # Random identical expert outputs BEFORE split
    expert_outputs = rng.normal(size=(E, T, d))
    expert_outputs[j_dead] = 0.0

    for tau_r in (0.5, 1.0, 2.5):
        logits_before = (X @ W_r.T + b[None, :]) / tau_r
        probs_before = _softmax(logits_before)
        y_before = np.einsum("te,etd->td", probs_before, expert_outputs)

        # Split donor -> dead WITHOUT orthogonal noise first to verify exact math identity
        W_after = W_r.copy()
        b_after = b.copy()
        W_after[j_dead] = W_after[j_donor]
        shifted = b_after[j_donor] - tau_r * math.log(2.0)
        b_after[j_donor] = shifted
        b_after[j_dead] = shifted

        exp_out_after = expert_outputs.copy()
        exp_out_after[j_dead] = exp_out_after[j_donor]

        logits_after = (X @ W_after.T + b_after[None, :]) / tau_r
        probs_after = _softmax(logits_after)
        y_after = np.einsum("te,etd->td", probs_after, exp_out_after)

        pair_prob_diff = np.max(np.abs((probs_after[:, j_donor] + probs_after[:, j_dead]) - probs_before[:, j_donor]))
        other_mask = np.ones(E, dtype=bool)
        other_mask[[j_donor, j_dead]] = False
        other_prob_diff = np.max(np.abs(probs_after[:, other_mask] - probs_before[:, other_mask]))
        output_diff = np.max(np.abs(y_after - y_before))
        print(
            f"  tau_r={tau_r:<4.1f} | max |p_pair - p_donor|={pair_prob_diff:.3e} | "
            f"max |p_others diff|={other_prob_diff:.3e} | max |MoE_out diff|={output_diff:.3e}"
        )


def simulate_routing_run(mode: str, steps: int = 400, seed: int = 123):
    """
    Simulate non-stationary multi-domain routing with realistic feedback:
      - 4 token clusters/domains arrive with uneven frequency (45%, 30%, 15%, 10%).
      - Expert skill improves smoothly with tokens seen -> positive-feedback collapse tendency!
      - Compare how each balancer behaves over 400 optimizer steps.
    """
    rng = np.random.default_rng(seed)
    E, K_r, d = 16, 2, 24
    T = 512
    tau_r = 1.0
    alpha_0, alpha_min = 0.08, 0.015
    split_interval, split_tau = 40, 0.10

    # Domain centroids in hidden space (shift slowly across steps to model non-stationarity)
    domain_centers = rng.normal(scale=1.2, size=(4, d))
    domain_probs = np.array([0.48, 0.27, 0.15, 0.10])

    # Router initialized slightly skewed so experts 0..2 initially attract hot traffic (simulating collapse risk)
    W_r = rng.normal(scale=0.25, size=(E, d))
    b = np.zeros(E)
    b[:3] = 1.6
    b[10:] = -2.2  # Deliberately starved slots

    # Expert prototype centroids track distinct data sub-clusters; overloaded experts
    # stretched across multiple clusters incur high intra-cluster interference error.
    expert_proto = rng.normal(scale=0.3, size=(E, d))
    ema_load = np.full(E, K_r / E)
    ema_energy = np.ones(E) * 0.5
    cooldown = np.zeros(E, dtype=int)
    ds_bias = np.zeros(E)

    splits_done = 0
    domain_mismatch_loss = []

    for step in range(1, steps + 1):
        # Gentle non-stationary representation drift across 4 unbalanced domains
        domain_centers += rng.normal(scale=0.008, size=domain_centers.shape)
        dom_ids = rng.choice(4, size=T, p=domain_probs)
        X = domain_centers[dom_ids] + rng.normal(scale=0.55, size=(T, d))

        logits = (X @ W_r.T + b[None, :]) / tau_r
        probs = _softmax(logits)

        sel_scores = probs + ds_bias[None, :] if mode == "deepseek_bias" else probs
        topk_idx = np.argpartition(sel_scores, -K_r, axis=-1)[:, -K_r:]

        counts = np.bincount(topk_idx.ravel(), minlength=E)
        batch_load = counts / T  # sums to K_r

        # Domain specialization update: each active expert learns toward its routed token cluster,
        # while routing entropy + cluster spread determines token reconstruction distortion.
        batch_energy = np.zeros(E)
        step_loss = 0.0
        for j in range(E):
            mask = np.any(topk_idx == j, axis=1)
            n_j = int(mask.sum())
            if n_j > 0:
                X_j = X[mask]
                mean_x = X_j.mean(axis=0)
                expert_proto[j] = 0.90 * expert_proto[j] + 0.10 * mean_x
                # Router aligns with specialized prototype (rich-get-richer feedback)
                lr_r = 0.08 * (n_j / T)
                W_r[j] = 0.97 * W_r[j] + lr_r * mean_x
                b[j] += 0.012 * (n_j / T)
                sq_err = float(np.mean(np.sum((X_j - expert_proto[j][None, :]) ** 2, axis=-1)))
                step_loss += n_j * sq_err
                batch_energy[j] = float(np.mean(probs[mask, j]) * (np.linalg.norm(mean_x) ** 2 + sq_err))
        domain_mismatch_loss.append(step_loss / (T * K_r))

        if mode == "aux_loss":
            fair = K_r / E
            grad_aux = 0.35 * E * (batch_load - fair)
            b -= 0.18 * grad_aux
        elif mode == "deepseek_bias":
            fair = K_r / E
            ds_bias -= 0.01 * np.sign(batch_load - fair)

        # Update cosine EMA tracking floor
        cos_v = 0.5 * (1.0 + math.cos(math.pi * step / steps))
        alpha = alpha_min + (alpha_0 - alpha_min) * cos_v
        ema_load = (1.0 - alpha) * ema_load + alpha * batch_load
        ema_energy = (1.0 - alpha) * ema_energy + alpha * batch_energy
        cooldown = np.maximum(cooldown - 1, 0)

        # Dynamic Expert Splitting step (checks every split_interval except final cool-down tail)
        if mode == "emeans" and step % split_interval == 0 and step <= steps - 20:
            fair = K_r / E
            dead_ids = np.where((ema_load < split_tau * fair) & (cooldown == 0))[0]
            if len(dead_ids) > 0:
                donor_pool = np.where((ema_load > 1.15 * fair) & (cooldown == 0))[0]
                if len(donor_pool) == 0:
                    donor_pool = np.where(ema_load > fair)[0]
                n_pairs = min(len(dead_ids), len(donor_pool), 4)
                if n_pairs > 0:
                    w_don = np.maximum(ema_energy[donor_pool], 1e-8)
                    w_don /= w_don.sum()
                    donors = rng.choice(donor_pool, size=n_pairs, replace=False, p=w_don)
                    for dead_i, don_i in zip(dead_ids[:n_pairs], donors):
                        # Clone router row + orthogonal break along top batch variance direction
                        noise = rng.normal(scale=0.06, size=d)
                        W_r[dead_i] = W_r[don_i] + noise
                        W_r[don_i] = W_r[don_i] - noise
                        shifted = b[don_i] - tau_r * math.log(2.0)
                        b[don_i] = shifted
                        b[dead_i] = shifted
                        expert_proto[dead_i] = expert_proto[don_i] + 0.05 * noise
                        expert_proto[don_i] = expert_proto[don_i] - 0.05 * noise
                        ema_load[don_i] *= 0.5
                        ema_load[dead_i] = ema_load[don_i]
                        ema_energy[don_i] *= 0.5
                        ema_energy[dead_i] = ema_energy[don_i]
                        cooldown[don_i] = split_interval
                        cooldown[dead_i] = split_interval
                        splits_done += 1

        health = compute_routing_health(ema_load, top_k=K_r, dead_tau=split_tau)

    final_h = compute_routing_health(ema_load, top_k=K_r, dead_tau=split_tau)
    return {
        "mode": mode,
        "splits": splits_done,
        "final_dead": int(final_h["dead_count"]),
        "final_max_mean": final_h["max_to_mean"],
        "final_entropy": final_h["norm_entropy"],
        "final_gini": final_h["gini"],
        "mean_proxy_loss_last50": float(np.mean(domain_mismatch_loss[-50:])),
    }


def main():
    verify_exact_bias_invariance()

    print("\n[2] 400-STEP MULTI-DOMAIN NON-STATIONARY ROUTING MINI-RUN (E=16, K_r=2)")
    print("-" * 88)
    print(f"{'Method':<18} {'Splits':>6} {'DeadExp':>8} {'Max/Mean':>9} {'Entropy':>9} {'Gini':>7} {'ProxyLoss':>10}")
    print("-" * 88)
    for mode in ("unbalanced", "aux_loss", "deepseek_bias", "emeans"):
        res = simulate_routing_run(mode)
        print(
            f"{res['mode']:<18} {res['splits']:>6d} {res['final_dead']:>8d} "
            f"{res['final_max_mean']:>8.2f}x {res['final_entropy']:>9.3f} "
            f"{res['final_gini']:>7.3f} {res['mean_proxy_loss_last50']:>10.4f}"
        )


if __name__ == "__main__":
    main()
