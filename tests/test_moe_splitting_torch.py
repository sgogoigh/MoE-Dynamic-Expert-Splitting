"""PyTorch cluster tests for DynamicExpertSplitter, TinyGemma upcycling, and AdamW state sync."""

import math
import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from moe_splitting import (  # noqa: E402
    EMeansMoEConfig,
    GEMMA_PRESETS,
    TinyGemmaForCausalLM,
    upcycle_gemma_model,
)


def test_upcycled_gemma_exact_parity_at_zero_noise():
    torch.manual_seed(0)
    spec = GEMMA_PRESETS["tiny-gemma-debug"]
    dense = TinyGemmaForCausalLM(spec)
    ids = torch.randint(0, spec.vocab_size, (4, 16))
    with torch.no_grad():
        dense_logits = dense(ids)

    cfg = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy="noisy_clone", upcycle_init_noise=0.0)
    moe = upcycle_gemma_model(dense, cfg)
    with torch.no_grad():
        moe_logits, _ = moe(ids)

    assert torch.allclose(dense_logits, moe_logits, atol=1e-5, rtol=1e-4)


def test_dynamic_split_revives_dead_expert_and_preserves_ema_sum():
    torch.manual_seed(1)
    spec = GEMMA_PRESETS["tiny-gemma-debug"]
    dense = TinyGemmaForCausalLM(spec)
    cfg = EMeansMoEConfig(
        num_experts=6,
        top_k=2,
        split_interval=1,
        split_tau=0.1,
        donor_min_mult=1.1,
        cooldown_steps=10,
        upcycle_strategy="progressive_split",
        progressive_active_experts=2,
        dormant_init_bias=-8.0,
    )
    moe = upcycle_gemma_model(dense, cfg)
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)

    ids = torch.randint(0, spec.vocab_size, (4, 16))
    logits, _ = moe(ids)
    loss = logits.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    router0 = moe.iter_moe_layers()[0].mlp.router
    sum_before = router0.ema_load.sum().item()
    bias_donor_before = router0.gate.bias[0].item()

    events = moe.step_ema_and_split(opt, step=1, total_steps=20)
    assert len(events) > 0
    # After split, dead slots receive positive EMA share while total EMA mass stays conserved
    sum_after = router0.ema_load.sum().item()
    assert abs(sum_after - sum_before) < 1e-4
    # First split event bias matches -tau_r ln 2 relative to donor before split
    ev0 = next(e for e in events if e.layer_name == "layer_0")
    assert math.isclose(
        router0.gate.bias[ev0.dead_idx].item(),
        router0.gate.bias[ev0.donor_idx].item(),
        rel_tol=1e-5,
    )
    assert router0.cooldown[ev0.dead_idx].item() == cfg.cooldown_steps
