"""PyTorch cluster tests: Gemma GeGLU/RMSNorm fidelity, AdamW sync, PoC 2 spike ablation, PoC 4 freezing."""

import math
import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from moe_splitting import (  # noqa: E402
    EMeansMoEConfig,
    GEMMA_PRESETS,
    TinyGemmaForCausalLM,
    compute_routing_health,
    upcycle_gemma_model,
)
from moe_splitting.gemma_upcycle import GemmaDenseGeGLUMLP, GemmaRMSNorm  # noqa: E402
from moe_splitting.splitting import _orthogonal_perturbation  # noqa: E402
from poc import poc_02_tiny_gemma_continuity as poc2  # noqa: E402
from poc import poc_04_gemma3_1b_frozen_upcycle as poc4  # noqa: E402


def test_gemma_rmsnorm_unit_offset_zero_weight_not_zero_output():
    torch.manual_seed(0)
    norm = GemmaRMSNorm(dim=32)
    assert torch.all(norm.weight == 0.0)
    x = torch.randn(4, 8, 32)
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


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


def test_geglu_expert_matches_dense_mlp_numerically():
    torch.manual_seed(3)
    dense_mlp = GemmaDenseGeGLUMLP(hidden_size=32, intermediate_size=64)
    cfg = EMeansMoEConfig(num_experts=4, top_k=1, upcycle_strategy="noisy_clone", upcycle_init_noise=0.0)
    from moe_splitting.gemma_upcycle import upcycle_gemma_mlp_to_moe

    moe_blk = upcycle_gemma_mlp_to_moe(dense_mlp, cfg)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        y_dense = dense_mlp(x)
        y_moe = moe_blk(x)
    assert torch.allclose(y_dense, y_moe, atol=1e-5, rtol=1e-4)


def test_orthogonal_perturbation_perpendicular_and_scaled():
    gen = torch.Generator().manual_seed(77)
    w = torch.randn(64, 32)
    delta = _orthogonal_perturbation(w, eps=0.01, gen=gen)
    cos_sim = torch.sum(w * delta) / (torch.norm(w) * torch.norm(delta))
    rel_mag = torch.norm(delta) / torch.norm(w)
    assert abs(float(cos_sim.item())) < 1e-4
    assert math.isclose(float(rel_mag.item()), 0.01, rel_tol=1e-3)


def test_dynamic_split_revives_dead_expert_and_migrates_adamw_state():
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
    experts0 = moe.iter_moe_layers()[0].mlp.experts
    sum_before = router0.ema_load.sum().item()

    events = moe.step_ema_and_split(opt, step=1, total_steps=20)
    assert len(events) > 0
    assert abs(router0.ema_load.sum().item() - sum_before) < 1e-4

    ev0 = next(e for e in events if e.layer_name == "layer_0")
    assert math.isclose(
        router0.gate.bias[ev0.dead_idx].item(),
        router0.gate.bias[ev0.donor_idx].item(),
        rel_tol=1e-5,
    )
    assert router0.cooldown[ev0.dead_idx].item() == cfg.cooldown_steps
    # Confirm AdamW exp_avg_sq was migrated to dead slot
    v_state = opt.state[experts0.gate_proj]["exp_avg_sq"]
    assert torch.allclose(v_state[ev0.dead_idx], v_state[ev0.donor_idx])


def test_poc02_bias_shift_reduces_split_loss_spike_vs_ablation():
    """Functional baseline check verifying PoC 2 spike ablation logic runs cleanly."""
    class Args:
        d_model = 64
        ff_dim = 128
        layers = 2
        num_experts = 6
        top_k = 2
        batch_size = 4
        seq_len = 16
        steps = 20
        split_interval = 5
        lr = 5e-4
        seed = 42

    spec = GEMMA_PRESETS["tiny-gemma-debug"]
    spec_small = type(spec)(
        name="test-poc2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1024,
    )
    res_shift = poc2.run_single_condition("emeans", Args(), spec_small, "cpu")
    res_noshift = poc2.run_single_condition("emeans_no_shift", Args(), spec_small, "cpu")
    assert res_shift["num_splits"] > 0
    assert res_noshift["num_splits"] > 0
    assert res_shift["max_abs_split_delta_loss"] <= res_noshift["max_abs_split_delta_loss"] + 1e-3


def test_poc04_freeze_backbone_leaves_only_top_n_moe_trainable():
    spec = GEMMA_PRESETS["tiny-gemma-debug"]
    dense = TinyGemmaForCausalLM(spec)
    cfg = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy="progressive_split", progressive_active_experts=2)
    model, moe_blocks = poc4.freeze_backbone_and_upcycle_top_layers(dense, cfg, last_n=2)
    assert len(moe_blocks) == 2
    assert not model.embed_tokens.weight.requires_grad
    assert not model.layers[0].mlp.gate_proj.weight.requires_grad
    assert model.layers[-1].mlp.router.gate.weight.requires_grad
    assert model.layers[-1].mlp.experts.gate_proj.requires_grad
