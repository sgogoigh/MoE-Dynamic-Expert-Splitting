"""
PoC 1 (Offline CPU, Zero Cost): Numerical Proof + Routing Simulation + Gemma Audit.
Writes structured telemetry to `poc/results/poc_01_results.json`.
"""

import json
import math
import pathlib
import sys
import numpy as np

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import GEMMA_PRESETS  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run_invariance_proof() -> list:
    rng = np.random.default_rng(7)
    T, d, E = 64, 32, 16
    X = rng.normal(size=(T, d))
    W_r = rng.normal(scale=0.35, size=(E, d))
    b = rng.normal(scale=0.5, size=(E,))
    j_donor, j_dead = 3, 11
    b[j_donor] = 1.8
    b[j_dead] = -1e9

    expert_outputs = rng.normal(size=(E, T, d))
    expert_outputs[j_dead] = 0.0

    rows = []
    for tau_r in (0.5, 1.0, 2.5):
        probs_before = _softmax((X @ W_r.T + b[None, :]) / tau_r)
        y_before = np.einsum("te,etd->td", probs_before, expert_outputs)

        W_after = W_r.copy()
        b_after = b.copy()
        W_after[j_dead] = W_after[j_donor]
        shifted = b_after[j_donor] - tau_r * math.log(2.0)
        b_after[j_donor] = shifted
        b_after[j_dead] = shifted

        exp_out_after = expert_outputs.copy()
        exp_out_after[j_dead] = exp_out_after[j_donor]
        probs_after = _softmax((X @ W_after.T + b_after[None, :]) / tau_r)
        y_after = np.einsum("te,etd->td", probs_after, exp_out_after)

        other_mask = np.ones(E, dtype=bool)
        other_mask[[j_donor, j_dead]] = False

        rows.append(
            {
                "tau_r": tau_r,
                "max_pair_prob_diff": float(np.max(np.abs((probs_after[:, j_donor] + probs_after[:, j_dead]) - probs_before[:, j_donor]))),
                "max_other_prob_diff": float(np.max(np.abs(probs_after[:, other_mask] - probs_before[:, other_mask]))),
                "max_moe_output_diff": float(np.max(np.abs(y_after - y_before))),
            }
        )
    return rows


def run_synthetic_simulation(mode: str, steps: int = 400, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    E, K_r, d, T, tau_r = 16, 2, 24, 512, 1.0
    alpha_0, alpha_min = 0.08, 0.015
    split_interval, split_tau = 40, 0.10

    domain_centers = rng.normal(scale=1.2, size=(4, d))
    domain_probs = np.array([0.48, 0.27, 0.15, 0.10])
    W_r = rng.normal(scale=0.25, size=(E, d))
    b = np.zeros(E)
    b[:3] = 1.6
    b[10:] = -2.2

    expert_proto = rng.normal(scale=0.3, size=(E, d))
    ema_load = np.full(E, K_r / E)
    ema_energy = np.ones(E) * 0.5
    cooldown = np.zeros(E, dtype=int)
    ds_bias = np.zeros(E)

    splits_done = 0
    losses = []

    for step in range(1, steps + 1):
        domain_centers += rng.normal(scale=0.008, size=domain_centers.shape)
        dom_ids = rng.choice(4, size=T, p=domain_probs)
        X = domain_centers[dom_ids] + rng.normal(scale=0.55, size=(T, d))

        probs = _softmax((X @ W_r.T + b[None, :]) / tau_r)
        sel = probs + ds_bias[None, :] if mode == "deepseek_bias" else probs
        topk_idx = np.argpartition(sel, -K_r, axis=-1)[:, -K_r:]
        batch_load = np.bincount(topk_idx.ravel(), minlength=E) / T

        batch_energy = np.zeros(E)
        step_loss = 0.0
        for j in range(E):
            mask = np.any(topk_idx == j, axis=1)
            n_j = int(mask.sum())
            if n_j > 0:
                X_j = X[mask]
                mean_x = X_j.mean(axis=0)
                expert_proto[j] = 0.90 * expert_proto[j] + 0.10 * mean_x
                W_r[j] = 0.97 * W_r[j] + (0.08 * (n_j / T)) * mean_x
                b[j] += 0.012 * (n_j / T)
                sq_err = float(np.mean(np.sum((X_j - expert_proto[j][None, :]) ** 2, axis=-1)))
                step_loss += n_j * sq_err
                batch_energy[j] = float(np.mean(probs[mask, j]) * (np.linalg.norm(mean_x) ** 2 + sq_err))
        losses.append(step_loss / (T * K_r))

        if mode == "aux_loss":
            b -= 0.18 * (0.35 * E * (batch_load - (K_r / E)))
        elif mode == "deepseek_bias":
            ds_bias -= 0.01 * np.sign(batch_load - (K_r / E))

        cos_v = 0.5 * (1.0 + math.cos(math.pi * step / steps))
        alpha = alpha_min + (alpha_0 - alpha_min) * cos_v
        ema_load = (1.0 - alpha) * ema_load + alpha * batch_load
        ema_energy = (1.0 - alpha) * ema_energy + alpha * batch_energy
        cooldown = np.maximum(cooldown - 1, 0)

        if mode == "emeans" and step % split_interval == 0 and step <= steps - 20:
            fair = K_r / E
            dead_ids = np.where((ema_load < split_tau * fair) & (cooldown == 0))[0]
            donor_pool = np.where((ema_load > 1.15 * fair) & (cooldown == 0))[0]
            if len(donor_pool) == 0:
                donor_pool = np.where(ema_load > fair)[0]
            n_pairs = min(len(dead_ids), len(donor_pool), 4)
            if len(dead_ids) > 0 and n_pairs > 0:
                w_don = np.maximum(ema_energy[donor_pool], 1e-8)
                w_don /= w_don.sum()
                donors = rng.choice(donor_pool, size=n_pairs, replace=False, p=w_don)
                for dead_i, don_i in zip(dead_ids[:n_pairs], donors):
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

    final_h = compute_routing_health(ema_load, top_k=K_r, dead_tau=split_tau)
    return {
        "mode": mode,
        "splits": splits_done,
        "dead_experts": int(final_h["dead_count"]),
        "max_to_mean": round(final_h["max_to_mean"], 4),
        "norm_entropy": round(final_h["norm_entropy"], 4),
        "gini": round(final_h["gini"], 4),
        "proxy_loss_last50": round(float(np.mean(losses[-50:])), 4),
    }


def profile_gemma_targets() -> list:
    specs = [
        ("gemma-3-1b", 16, 2, "sliced_granular"),
        ("gemma-3-1b", 16, 2, "noisy_clone"),
        ("gemma-2-2b", 16, 2, "sliced_granular"),
        ("gemma-2-2b", 32, 2, "sliced_granular"),
        ("gemma-2-9b", 64, 4, "sliced_granular"),
    ]
    profiles = []
    for preset_name, e_cnt, k_r, strat in specs:
        s = GEMMA_PRESETS[preset_name]
        d, ff, L = s.hidden_size, s.intermediate_size, s.num_hidden_layers
        non_mlp = (s.vocab_size * d) + L * (
            d * s.num_attention_heads * s.head_dim
            + 2 * d * s.num_key_value_heads * s.head_dim
            + s.num_attention_heads * s.head_dim * d
            + 4 * d
        ) + d
        dense_total = non_mlp + L * (3 * d * ff)
        exp_ff = ff // k_r if strat == "sliced_granular" else ff
        exp_each = 3 * d * exp_ff
        moe_total = non_mlp + L * (e_cnt * exp_each + d * e_cnt + e_cnt)
        moe_active = non_mlp + L * (k_r * exp_each + d * e_cnt + e_cnt)
        profiles.append(
            {
                "preset": preset_name,
                "E": e_cnt,
                "top_k": k_r,
                "strategy": strat,
                "dense_params_M": round(dense_total / 1e6, 2),
                "moe_total_params_M": round(moe_total / 1e6, 2),
                "moe_active_params_M": round(moe_active / 1e6, 2),
                "unsharded_train_GB": round((moe_total * 16.0) / (1024 ** 3), 2),
                "ema_sync_KB_per_step": round((L * 2 * e_cnt * 4) / 1024.0, 2),
            }
        )
    return profiles


def main():
    out_dir = POC_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    invariance = run_invariance_proof()
    simulations = [run_synthetic_simulation(m) for m in ("unbalanced", "aux_loss", "deepseek_bias", "emeans")]
    gemma_profiles = profile_gemma_targets()

    payload = {
        "poc_id": "poc_01",
        "title": "Offline Invariance Proof, Synthetic Routing Comparison & Gemma Architecture Audit",
        "invariance_proof": invariance,
        "synthetic_simulation": simulations,
        "gemma_architecture_profiles": gemma_profiles,
    }
    out_path = out_dir / "poc_01_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== PoC 1 COMPLETE ===")
    print(f"Max softmax invariance diff across all tau_r: {max(r['max_moe_output_diff'] for r in invariance):.3e}")
    for row in simulations:
        print(
            f"  {row['mode']:<15} | splits={row['splits']:2d} | dead={row['dead_experts']:2d} | "
            f"max/mean={row['max_to_mean']:.2f}x | entropy={row['norm_entropy']:.3f}"
        )
    print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
