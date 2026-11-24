"""Upcycling: step-0 parity per strategy, dtype/device placement, HF Gemma 2 / Gemma 3 path.

Run:  python -m unittest tests.test_upcycle_torch -v      (or pytest)
"""

import os
import pathlib
import sys
import unittest

os.environ.setdefault("USE_TF", "0")            # keep transformers from importing TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None

try:
    import transformers  # noqa: F401
    from transformers import Gemma2Config, Gemma2ForCausalLM
except Exception:  # pragma: no cover
    transformers = None

if torch is not None:
    from moe_splitting import (
        DynamicExpertSplitter,
        EMeansMoEConfig,
        GEMMA_PRESETS,
        TinyGemmaForCausalLM,
        frozen_moe_stats,
        iter_moe_blocks,
        step_moe_blocks,
        upcycle_gemma_model,
    )
    from moe_splitting.gemma_upcycle import GemmaDenseGeGLUMLP, upcycle_gemma_mlp_to_moe


def _spec():
    base = GEMMA_PRESETS["tiny-gemma-debug"]
    return type(base)(name="tiny-upcycle", hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=12, vocab_size=128)


def _logits(model, ids):
    model.eval()
    with torch.no_grad():
        out = model(ids)
    if isinstance(out, tuple):
        return out[0]
    return out.logits if hasattr(out, "logits") else out


@unittest.skipIf(torch is None, "torch not installed")
class TestUpcycleParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dense = TinyGemmaForCausalLM(_spec())
        self.ids = torch.randint(0, 128, (3, 20), generator=torch.Generator().manual_seed(1))
        self.ref = _logits(self.dense, self.ids)

    def _delta(self, **cfg_kw):
        cfg = EMeansMoEConfig(upcycle_init_noise=0.0, **cfg_kw)
        moe = upcycle_gemma_model(self.dense, cfg)
        return (_logits(moe, self.ids) - self.ref).abs().max().item()

    def test_noisy_clone_exact_for_k_ge_2(self):
        for E, K in ((4, 2), (8, 4), (6, 3)):
            self.assertLess(self._delta(num_experts=E, top_k=K, upcycle_strategy="noisy_clone"), 1e-4, f"E={E} K={K}")

    def test_sliced_parity_exact(self):
        for E, K in ((8, 2), (6, 3), (16, 2)):
            self.assertLess(self._delta(num_experts=E, top_k=K, upcycle_strategy="sliced_parity"), 1e-4, f"E={E} K={K}")

    def test_sliced_granular_is_not_parity_preserving(self):
        self.assertGreater(self._delta(num_experts=16, top_k=2, upcycle_strategy="sliced_granular"), 1e-2)

    def test_progressive_split_exact_with_identical_experts(self):
        self.assertLess(self._delta(num_experts=6, top_k=2, upcycle_strategy="progressive_split",
                                    progressive_active_experts=3), 1e-4)

    def test_init_noise_breaks_parity_proportionally(self):
        cfg_a = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_init_noise=1e-4)
        cfg_b = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_init_noise=1e-3)
        da = (_logits(upcycle_gemma_model(self.dense, cfg_a), self.ids) - self.ref).abs().max().item()
        db = (_logits(upcycle_gemma_model(self.dense, cfg_b), self.ids) - self.ref).abs().max().item()
        self.assertGreater(da, 0.0)
        self.assertGreater(db, 3 * da)

    def test_invalid_strategy_and_k1_sliced_parity_rejected(self):
        mlp = GemmaDenseGeGLUMLP(16, 32)
        with self.assertRaises(ValueError):
            upcycle_gemma_mlp_to_moe(mlp, EMeansMoEConfig(upcycle_strategy="bogus"))
        with self.assertRaises(ValueError):
            upcycle_gemma_mlp_to_moe(mlp, EMeansMoEConfig(top_k=1, upcycle_strategy="sliced_parity"))
        with self.assertRaises(ValueError):
            upcycle_gemma_mlp_to_moe(GemmaDenseGeGLUMLP(16, 33), EMeansMoEConfig(top_k=2, upcycle_strategy="sliced_parity"))

    def test_sliced_parity_survives_first_splits_with_sibling_routing(self):
        cfg = EMeansMoEConfig(num_experts=8, top_k=2, upcycle_strategy="sliced_parity", upcycle_init_noise=0.0,
                              split_interval=1, cooldown_steps=0, alpha_0=0.5, alpha_min=0.5)
        moe = upcycle_gemma_model(self.dense, cfg)
        moe.train()
        for step in range(1, 7):           # record statistics, split the dead slots (no optimizer step)
            with torch.no_grad():
                moe(self.ids)
            moe.step_ema_and_split(None, step=step, total_steps=100)
        n_pairs = sum(int(b.router.merged_mask().sum()) // 2 for _, b in iter_moe_blocks(moe))
        self.assertGreater(n_pairs, 0)
        self.assertLess((_logits(moe, self.ids) - self.ref).abs().max().item(), 1e-4)


@unittest.skipIf(torch is None, "torch not installed")
class TestUpcyclePlacement(unittest.TestCase):
    def test_experts_follow_dense_dtype_router_stays_fp32(self):
        mlp = GemmaDenseGeGLUMLP(16, 32).to(torch.bfloat16)
        blk = upcycle_gemma_mlp_to_moe(mlp, EMeansMoEConfig(num_experts=4, top_k=2, upcycle_init_noise=1e-3))
        self.assertEqual(blk.experts.gate_proj.dtype, torch.bfloat16)
        self.assertEqual(blk.router.gate.weight.dtype, torch.float32)
        self.assertEqual(blk.experts.gate_proj.device, mlp.gate_proj.weight.device)
        y = blk(torch.randn(2, 3, 16, dtype=torch.bfloat16))
        self.assertEqual(y.dtype, torch.bfloat16)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA not available")
    def test_cuda_dense_weights_give_cuda_block(self):  # pragma: no cover - CPU CI
        mlp = GemmaDenseGeGLUMLP(16, 32).cuda()
        blk = upcycle_gemma_mlp_to_moe(mlp, EMeansMoEConfig(num_experts=4, top_k=2))
        self.assertTrue(blk.experts.gate_proj.is_cuda)
        self.assertTrue(blk.router.gate.weight.is_cuda)

    def test_upcycle_does_not_mutate_the_dense_model(self):
        torch.manual_seed(0)
        dense = TinyGemmaForCausalLM(_spec())
        before = {k: v.clone() for k, v in dense.state_dict().items()}
        upcycle_gemma_model(dense, EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy="sliced_parity"))
        for k, v in dense.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)


@unittest.skipIf(torch is None or transformers is None, "torch/transformers not installed")
class TestHuggingFaceGemma(unittest.TestCase):
    KW = dict(vocab_size=128, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=12, max_position_embeddings=64)

    def _models(self):
        torch.manual_seed(0)
        out = [("gemma2", Gemma2ForCausalLM(Gemma2Config(**self.KW, attn_implementation="eager")))]
        try:
            from transformers import Gemma3ForCausalLM, Gemma3TextConfig
            out.append(("gemma3", Gemma3ForCausalLM(Gemma3TextConfig(**self.KW, sliding_window=16,
                                                                    attn_implementation="eager"))))
        except ImportError:
            pass
        return out

    def test_hf_parity_backward_and_split(self):
        ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(5))
        for name, dense in self._models():
            ref = _logits(dense, ids)
            for strat in ("noisy_clone", "sliced_parity"):
                cfg = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy=strat, upcycle_init_noise=0.0)
                moe = upcycle_gemma_model(dense, cfg)
                self.assertEqual(len(iter_moe_blocks(moe)), 2, name)
                self.assertLess((_logits(moe, ids) - ref).abs().max().item(), 1e-4, f"{name}/{strat}")

            for interval in (2, 10_000):   # with splits; statistics only (for the manual split below)
                cfg = EMeansMoEConfig(num_experts=6, top_k=2, upcycle_strategy="sliced_parity",
                                      upcycle_init_noise=0.0, split_interval=interval, cooldown_steps=0,
                                      alpha_0=0.5, alpha_min=0.5)
                moe = upcycle_gemma_model(dense, cfg)
                moe.train()
                opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)
                splitter = DynamicExpertSplitter()
                n_split = 0
                for step in range(1, 7):
                    out = moe(ids, labels=ids)
                    opt.zero_grad()
                    out.loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in moe.parameters()))
                    opt.step()
                    splits, _ = step_moe_blocks(moe, splitter, opt, step, 100)
                    n_split += len(splits)
                if interval == 2:
                    self.assertGreater(n_split, 0, name)
                    self.assertTrue(torch.isfinite(_logits(moe, ids)).all())

            # a split on the trained HF model is continuous with sibling routing
            _, blk = iter_moe_blocks(moe)[0]
            moe.eval()
            with torch.no_grad(), frozen_moe_stats(moe):
                before = moe(ids).logits
                load = blk.router.ema_load
                donor, dead = int(load.argmax()), int(load.argmin())
                splitter._execute_single_split(blk.router, blk.experts, None, dead, donor, 11)
                after = moe(ids).logits
            self.assertLess((after - before).abs().max().item(), 1e-3, name)


if __name__ == "__main__":
    unittest.main()
