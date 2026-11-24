"""PyTorch unit tests: Gemma GeGLU/RMSNorm fidelity, upcycle parity, split + AdamW state migration.

Run:  python -m unittest tests.test_moe_splitting_torch -v      (or pytest)
PoC end-to-end smoke runs live in tests/test_poc_smoke_torch.py.
"""

import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from moe_splitting import EMeansMoEConfig, GEMMA_PRESETS, TinyGemmaForCausalLM, upcycle_gemma_model
    from moe_splitting.gemma_upcycle import GemmaDenseGeGLUMLP, GemmaRMSNorm, upcycle_gemma_mlp_to_moe
    from moe_splitting.splitting import _orthogonal_perturbation


@unittest.skipIf(torch is None, "torch not installed")
class TestGemmaFidelityAndSplit(unittest.TestCase):
    def test_gemma_rmsnorm_unit_offset_zero_weight_not_zero_output(self):
        torch.manual_seed(0)
        norm = GemmaRMSNorm(dim=32)
        self.assertTrue(bool(torch.all(norm.weight == 0.0)))
        x = torch.randn(4, 8, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)
        rms = y.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-4))

    def test_upcycled_gemma_exact_parity_at_zero_noise(self):
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
        self.assertTrue(torch.allclose(dense_logits, moe_logits, atol=1e-5, rtol=1e-4))

    def test_geglu_expert_bank_vs_dense_mlp_top1_and_top2(self):
        """K>=2 with renormalised weights reproduces the dense MLP exactly. K=1 does not renormalise
        (the router would get no gradient), so the Top-1 output is p_top1 * dense, by design."""
        torch.manual_seed(3)
        dense_mlp = GemmaDenseGeGLUMLP(hidden_size=32, intermediate_size=64)
        x = torch.randn(2, 6, 32)
        with torch.no_grad():
            y_dense = dense_mlp(x)

        cfg2 = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy="noisy_clone", upcycle_init_noise=0.0)
        blk2 = upcycle_gemma_mlp_to_moe(dense_mlp, cfg2)
        with torch.no_grad():
            self.assertTrue(torch.allclose(blk2(x), y_dense, atol=1e-5, rtol=1e-4))

        cfg1 = EMeansMoEConfig(num_experts=4, top_k=1, upcycle_strategy="noisy_clone", upcycle_init_noise=0.0)
        blk1 = upcycle_gemma_mlp_to_moe(dense_mlp, cfg1)
        with torch.no_grad():
            y1 = blk1(x)
            p_top1 = blk1.router(x.reshape(-1, 32)).topk_weights.reshape(2, 6, 1)
        self.assertTrue(torch.allclose(y1, p_top1 * y_dense, atol=1e-5, rtol=1e-4))
        self.assertTrue(bool(torch.all(p_top1 < 1.0)))

    def test_orthogonal_perturbation_perpendicular_and_scaled(self):
        gen = torch.Generator().manual_seed(77)
        w = torch.randn(64, 32)
        delta = _orthogonal_perturbation(w, eps=0.01, gen=gen)
        cos_sim = torch.sum(w * delta) / (torch.norm(w) * torch.norm(delta))
        rel_mag = torch.norm(delta) / torch.norm(w)
        self.assertLess(abs(float(cos_sim.item())), 1e-4)
        self.assertTrue(math.isclose(float(rel_mag.item()), 0.01, rel_tol=1e-3))
        self.assertTrue(torch.equal(_orthogonal_perturbation(w, eps=0.0, gen=gen), torch.zeros_like(w)))

    def test_dynamic_split_revives_dead_expert_and_migrates_adamw_state(self):
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
        bias_before = router0.gate.bias.detach().clone()

        events = moe.step_ema_and_split(opt, step=1, total_steps=20)
        self.assertGreater(len(events), 0)
        ev0 = next(e for e in events if e.layer_name == "layers.0.mlp")
        sum_after_ema = router0.ema_load.sum().item()
        self.assertAlmostEqual(sum_after_ema, cfg.top_k, places=4)   # splitting conserves total load

        # function-preserving bias pair: b_d + b_c = 2 (b_donor - tau ln2)
        b_sum = router0.gate.bias[ev0.dead_idx].item() + router0.gate.bias[ev0.donor_idx].item()
        expected = 2 * (bias_before[ev0.donor_idx].item() - cfg.tau_r * math.log(2.0))
        self.assertAlmostEqual(b_sum, expected, places=5)
        self.assertEqual(router0.cooldown[ev0.dead_idx].item(), cfg.cooldown_steps)
        self.assertEqual(int(router0.partner[ev0.dead_idx]), ev0.donor_idx)
        self.assertTrue(ev0.sibling_merged)
        # AdamW exp_avg_sq copied to the revived slot; exp_avg of the router halved across siblings
        v_state = opt.state[experts0.gate_proj]["exp_avg_sq"]
        self.assertTrue(torch.allclose(v_state[ev0.dead_idx], v_state[ev0.donor_idx]))
        m_router = opt.state[router0.gate.weight]["exp_avg"]
        self.assertTrue(torch.allclose(m_router[ev0.dead_idx], m_router[ev0.donor_idx]))

    def test_step_is_noop_without_new_training_forward(self):
        torch.manual_seed(2)
        cfg = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_init_noise=0.0)
        moe = upcycle_gemma_model(TinyGemmaForCausalLM(GEMMA_PRESETS["tiny-gemma-debug"]), cfg)
        ids = torch.randint(0, 64, (2, 8))
        moe.train()
        moe(ids)
        moe.step_ema_and_split(None, step=1, total_steps=10)
        r = moe.iter_moe_layers()[0].mlp.router
        snap = r.ema_load.clone()
        moe.step_ema_and_split(None, step=2, total_steps=10)       # stats consumed -> no update
        self.assertTrue(torch.equal(snap, r.ema_load))
        moe.eval()
        moe(ids)                                                   # eval forward records nothing
        moe.step_ema_and_split(None, step=3, total_steps=10)
        self.assertTrue(torch.equal(snap, r.ema_load))


if __name__ == "__main__":
    unittest.main()
