"""
Offline Exploratory Profiler: Pretrained Gemma Dense -> MoE Upcycling Calculator.
Computes exact parameter counts, active vs total ratios, BF16 + AdamW VRAM footprints,
FLOPs/token, and EMA distributed sync traffic WITHOUT downloading or loading weights.
"""

import pathlib
import sys
from typing import Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import GEMMA_PRESETS, GemmaArchitectureSpec  # noqa: E402


def profile_gemma_upcycle(
    spec: GemmaArchitectureSpec,
    num_experts: int,
    top_k: int,
    strategy: str = "noisy_clone",
) -> Dict[str, float]:
    """Compute analytical parameter, memory, and FLOP breakdown for Dense vs. Upcycled MoE."""
    d = spec.hidden_size
    ff_dense = spec.intermediate_size
    L = spec.num_hidden_layers

    # Shared non-MLP parameters (tied embeddings + GQA + RMSNorms)
    emb_params = spec.vocab_size * d  # lm_head shares weight with embed_tokens
    q_proj = d * (spec.num_attention_heads * spec.head_dim)
    k_proj = d * (spec.num_key_value_heads * spec.head_dim)
    v_proj = d * (spec.num_key_value_heads * spec.head_dim)
    o_proj = (spec.num_attention_heads * spec.head_dim) * d
    norms_per_layer = 4 * d
    attn_params_total = L * (q_proj + k_proj + v_proj + o_proj + norms_per_layer) + d
    non_mlp_params = emb_params + attn_params_total

    # Dense GeGLU MLP: gate_proj + up_proj + down_proj = 3 * d * ff_dense per layer
    dense_mlp_per_layer = 3 * d * ff_dense
    dense_total_params = non_mlp_params + L * dense_mlp_per_layer

    # Upcycled MoE MLP
    exp_ff = ff_dense // top_k if strategy == "sliced_granular" else ff_dense
    expert_params_each = 3 * d * exp_ff
    router_params_per_layer = (d * num_experts) + num_experts  # W_r + b
    moe_mlp_per_layer = (num_experts * expert_params_each) + router_params_per_layer

    moe_total_params = non_mlp_params + L * moe_mlp_per_layer
    moe_active_params = non_mlp_params + L * ((top_k * expert_params_each) + router_params_per_layer)

    # Approximate forward+backward training FLOPs per token ~= 6 * active_non_emb_params
    active_non_emb_dense = dense_total_params - emb_params
    active_non_emb_moe = moe_active_params - emb_params
    flop_ratio = active_non_emb_moe / max(active_non_emb_dense, 1)

    # Memory estimate (weights in BF16 = 2B, grads in BF16 = 2B, AdamW m,v + master copy in FP32 = 12B -> 16B/param total)
    train_mem_gb_unsharded = (moe_total_params * 16.0) / (1024 ** 3)
    # EMA all-reduce per step per layer: 2 * E floats in FP32
    ema_sync_bytes_total = L * 2 * num_experts * 4

    return {
        "dense_params_M": dense_total_params / 1e6,
        "moe_total_params_M": moe_total_params / 1e6,
        "moe_active_params_M": moe_active_params / 1e6,
        "exp_intermediate_dim": float(exp_ff),
        "active_compute_vs_dense": flop_ratio,
        "unsharded_train_state_GB": train_mem_gb_unsharded,
        "ema_sync_KB_per_step": ema_sync_bytes_total / 1024.0,
    }


def main():
    print("=" * 98)
    print("OFFLINE GEMMA -> MoE ARCHITECTURE & VRAM EXPLORER (ZERO DOWNLOADS)")
    print("=" * 98)

    configs = [
        ("tiny-gemma-debug", 8, 2, "noisy_clone"),
        ("tiny-gemma-debug", 8, 2, "sliced_granular"),
        ("gemma-3-1b", 16, 2, "noisy_clone"),
        ("gemma-3-1b", 16, 2, "sliced_granular"),
        ("gemma-2-2b", 16, 2, "noisy_clone"),
        ("gemma-2-2b", 16, 2, "sliced_granular"),
        ("gemma-2-2b", 32, 2, "sliced_granular"),
        ("gemma-2-9b", 64, 4, "sliced_granular"),
    ]

    header = (
        f"{'Preset':<18} {'E/K_r':<7} {'Strategy':<16} {'Dense(M)':>9} "
        f"{'Total(M)':>10} {'Active(M)':>10} {'ExpFF':>6} {'FLOPx':>6} {'TrainGB':>8} {'EMA_KB':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, e_cnt, k_r, strat in configs:
        spec = GEMMA_PRESETS[name]
        res = profile_gemma_upcycle(spec, num_experts=e_cnt, top_k=k_r, strategy=strat)
        print(
            f"{name:<18} {f'{e_cnt}/{k_r}':<7} {strat:<16} "
            f"{res['dense_params_M']:>9.1f} {res['moe_total_params_M']:>10.1f} "
            f"{res['moe_active_params_M']:>10.1f} {int(res['exp_intermediate_dim']):>6d} "
            f"{res['active_compute_vs_dense']:>5.2f}x {res['unsharded_train_state_GB']:>8.2f} "
            f"{res['ema_sync_KB_per_step']:>7.2f}"
        )

    print("\nKey Takeaways:")
    print("  1. 'sliced_granular' keeps active FLOPs at ~1.00x dense Gemma while multiplying total capacity by E / K_r.")
    print("  2. Total EMA distributed all-reduce traffic per step is < 25 KB across ALL 42 layers of a 64-expert Gemma-2-9B!")
    print("  3. Unsharded TrainGB assumes naive DDP; FSDP / EP across 8x H100 divides parameter/optimizer memory by ~8x.")


if __name__ == "__main__":
    main()
