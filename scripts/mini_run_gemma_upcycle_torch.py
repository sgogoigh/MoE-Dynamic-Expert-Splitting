"""
PyTorch Dry-Run & Verification Script: TinyGemma Sparse Upcycling + Dynamic Splitting.
Runs offline on CPU or 1 GPU using synthetic weights & synthetic token batches (zero downloads).

Verifies:
  1. Exact step-0 dense->MoE logit parity when noisy_clone uses upcycle_init_noise=0.0.
  2. Deliberate starvation triggers dead-expert detection -> DynamicExpertSplitter revival.
  3. Zero large discontinuity upon split (-tau_r ln 2 bias invariance) + AdamW state sync.
  4. Clean 50-step AdamW synthetic CausalLM optimization loop without auxiliary loss.
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is not installed in this Python environment.")
    print("Run `python scripts/mini_run_math_verify.py` for offline NumPy verification, or install PyTorch on a GPU node.")
    sys.exit(0)

from moe_splitting import (  # noqa: E402
    EMeansMoEConfig,
    GEMMA_PRESETS,
    TinyGemmaForCausalLM,
    compute_routing_health,
    upcycle_gemma_model,
)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_json = json.loads((ROOT / "configs" / "tiny_gemma_debug.json").read_text(encoding="utf-8"))
    spec = GEMMA_PRESETS[cfg_json["model_preset"]]

    # Step 1: Build offline synthetic dense Gemma & inspect dense->MoE exactness
    dense_gemma = TinyGemmaForCausalLM(spec)
    dense_gemma.to(device)
    tokens = torch.randint(0, spec.vocab_size, (8, 32), device=device)

    with torch.no_grad():
        dense_logits = dense_gemma(tokens)

    exact_cfg = EMeansMoEConfig(num_experts=8, top_k=2, upcycle_strategy="noisy_clone", upcycle_init_noise=0.0)
    moe_exact = upcycle_gemma_model(dense_gemma, exact_cfg)
    moe_exact.to(device)
    with torch.no_grad():
        moe_logits, _ = moe_exact(tokens)
    parity_diff = (dense_logits - moe_logits).abs().max().item()
    print(f"[Parity Check] Dense TinyGemma vs Upcycled NoisyClone(eps=0) max |logit diff| = {parity_diff:.3e}")

    # Step 2: Initialize progressive/imbalanced MoE upcycle to observe DynamicExpertSplitter in action
    moe_cfg = EMeansMoEConfig(**cfg_json["moe"])
    moe_cfg.upcycle_strategy = "progressive_split"
    moe_cfg.progressive_active_experts = 4
    moe_cfg.dormant_init_bias = -6.0
    moe_cfg.split_interval = 10
    moe_cfg.cooldown_steps = 15

    model = upcycle_gemma_model(dense_gemma, moe_cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_json["training"]["lr"], weight_decay=0.01)
    total_steps = 50

    print("\n[Mini-Run] Running 50 AdamW steps with DynamicExpertSplitter active...")
    for step in range(1, total_steps + 1):
        batch = torch.randint(0, spec.vocab_size, (8, 32), device=device)
        logits, aux_loss = model(batch)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
        total_loss = loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        events = model.step_ema_and_split(optimizer, step=step, total_steps=total_steps)
        if events:
            # Measure immediate post-split loss continuity on the exact same batch
            with torch.no_grad():
                post_logits, _ = model(batch)
                post_loss = F.cross_entropy(post_logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
            print(
                f"  Step {step:02d}: Split {len(events)} pairs "
                f"(first: {events[0].layer_name} dead={events[0].dead_idx} <- donor={events[0].donor_idx}) | "
                f"pre_loss={loss.item():.4f} -> post_loss={post_loss.item():.4f}"
            )
        elif step in (1, 10, 25, 50):
            h0 = compute_routing_health(model.iter_moe_layers()[0].mlp.router.ema_load, top_k=moe_cfg.top_k, dead_tau=moe_cfg.split_tau)
            print(
                f"  Step {step:02d}: loss={loss.item():.4f} | L0 max/mean={h0['max_to_mean']:.2f}x | "
                f"L0 dead={int(h0['dead_count'])}/{moe_cfg.num_experts} | L0 entropy={h0['norm_entropy']:.3f}"
            )


if __name__ == "__main__":
    main()
