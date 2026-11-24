"""PoC 2/3/4 end-to-end at tiny size (results and logs in temp dirs), plus the shared PoC
helpers: token sources (train/held-out separation) and held-out evaluation.

Run:  python -m unittest tests.test_poc_smoke_torch -v      (or pytest)
"""

import gc
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from moe_splitting import (EMeansMoEConfig, GemmaArchitectureSpec, TinyGemmaForCausalLM, iter_moe_blocks,
                               step_moe_blocks, upcycle_gemma_model)
    from moe_splitting.evaluation import _mutual_information, heldout_eval
    from moe_splitting.splitting import DynamicExpertSplitter
    from moe_splitting.token_data import VAL_SEED_OFFSET, ShardSource, SyntheticSource


def _strict_json(path):
    """Load JSON and fail on NaN / Infinity (they are not valid JSON for other tools)."""
    def bad(c):
        raise ValueError(f"non-finite constant {c} in {path}")
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"), parse_constant=bad)


class _TempDirs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self._prev = os.environ.get("MOE_POC_LOGS_DIR")
        os.environ["MOE_POC_LOGS_DIR"] = str(self.tmp / "logs")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("MOE_POC_LOGS_DIR", None)
        else:
            os.environ["MOE_POC_LOGS_DIR"] = self._prev
        self._tmp.cleanup()


TINY_SPEC = dict(name="t", hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, vocab_size=128)


@unittest.skipIf(torch is None, "torch not installed")
class TestPoC2Smoke(_TempDirs):
    def test_part_a_split_is_continuous_and_ramps_complete(self):
        from poc import poc_02_tiny_gemma_continuity as poc2

        payload = poc2.main(["--part", "A", "--steps", "30", "--d-model", "32", "--ff-dim", "64", "--layers", "1",
                             "--vocab", "128", "--batch-size", "4", "--seq-len", "16", "--split-interval", "10",
                             "--merge-max-steps", "6", "--unmerge-ramp-steps", "4", "--out-dir", str(self.tmp)])
        saved = _strict_json(self.tmp / "poc_02_results.json")
        self.assertEqual(saved["diagnostics"]["sanity"], payload["diagnostics"]["sanity"])
        e = payload["part_a"]["emeans"]
        self.assertGreater(e["num_splits"], 0)
        self.assertLessEqual(e["split_dL"]["max_abs"], 1e-3)
        sanity = payload["diagnostics"]["sanity"]
        for k in ("emeans_triggered_splits", "emeans_split_continuous", "emeans_unmerge_ramp_has_no_completion_jump",
                  "part_a_finite"):
            self.assertTrue(sanity[k], k)
        ramps = [ep for ep in e["unmerge_episodes"] if ep["completed"] and ep["n_steps"] >= 2]
        self.assertTrue(ramps, "merge_max_steps=6 within 30 steps should force at least one ramped un-merge")
        self.assertIn("unmerge_cost", payload["diagnostics"]["comparison_informational"])


@unittest.skipIf(torch is None, "torch not installed")
class TestPoC3Smoke(_TempDirs):
    def test_grid_runs_and_unidentified_cells_are_not_fitted(self):
        from poc import poc_03_proxy_collapse_and_scaling as poc3

        payload = poc3.main(["--steps", "6", "--sweep-steps", "6", "--grid-experts", "4", "--grid-top-k", "1,2",
                             "--grid-batch", "4", "--val-batches", "1", "--val-batch-size", "4", "--d-model", "32",
                             "--ff-dim", "64", "--layers", "1", "--vocab", "128", "--seq-len", "16",
                             "--out-dir", str(self.tmp)])
        saved = _strict_json(self.tmp / "poc_03_results.json")
        self.assertTrue(saved["diagnostics"]["sanity_passed"])
        self.assertEqual(len(payload["part_b_cells"]), 2)
        for c in payload["part_b_cells"]:
            self.assertEqual([r["alpha_min"] for r in c["rungs"]], [0.005, 0.02, 0.08])
        fit = payload["diagnostics"]["fit_informational"]
        # 6 steps < split_interval: alpha_min cannot matter, every cell is flat, nothing is fitted
        self.assertEqual(fit["n_flat"], 2)
        self.assertEqual(fit["status"], "not_fitted")
        self.assertEqual(set(payload["part_a"]), {"unbalanced", "aux_loss", "deepseek_bias", "emeans"})
        for r in payload["part_a"].values():
            self.assertTrue(math.isfinite(r["val_loss_mean"]))
            self.assertEqual(set(r["per_domain_val_loss_mean"]), set(SyntheticSource.domain_names))


@unittest.skipIf(torch is None, "torch not installed")
class TestPoC4Smoke(_TempDirs):
    def test_synthetic_run_passes_sanity(self):
        from poc import poc_04_gemma3_1b_frozen_upcycle as poc4

        payload = poc4.main(["--synthetic-gemma", "--synthetic-layers", "3", "--upcycle-last-n", "1",
                             "--num-experts", "6", "--active-k0", "2", "--steps", "25", "--split-interval", "10",
                             "--seq-len", "16", "--batch-size", "4", "--val-batches", "2", "--lr", "1e-3",
                             "--out-dir", str(self.tmp)])
        _strict_json(self.tmp / "poc_04_results.json")
        self.assertTrue(payload["diagnostics"]["sanity_passed"], payload["diagnostics"]["sanity"])
        self.assertEqual(payload["upcycled_layers"], ["layers.2.mlp"])
        table = payload["domain_expert_share_end"]["layers.2.mlp"]
        self.assertEqual(len(table), 3)
        for row in table:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_freeze_keeps_decoder_layers_and_backbone_weights(self):
        from poc import poc_04_gemma3_1b_frozen_upcycle as poc4

        torch.manual_seed(0)
        model = TinyGemmaForCausalLM(GemmaArchitectureSpec(**TINY_SPEC))
        layers_before = list(model.layers)
        backbone = {n: p.detach().clone() for n, p in model.named_parameters() if ".mlp." not in n or "layers.0" in n}
        cfg = EMeansMoEConfig(num_experts=4, top_k=2, split_interval=2, upcycle_strategy="progressive_split",
                              progressive_active_experts=2, cooldown_steps=4)
        model, blocks = poc4.freeze_backbone_and_upcycle_top_layers(model, cfg, last_n=1)
        self.assertEqual([n for n, _ in blocks], ["layers.1.mlp"])
        self.assertTrue(all(a is b for a, b in zip(layers_before, model.layers)), "decoder layers must be kept")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        src = SyntheticSource()
        for step in range(1, 5):
            tok, _ = src.batch(step, 4, 16, 128)
            logits = model(tok)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), tok[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_moe_blocks(model, DynamicExpertSplitter(), opt, step=step, total_steps=4)
        for n, p in model.named_parameters():
            if n in backbone:
                self.assertTrue(torch.equal(p, backbone[n]), f"frozen parameter {n} changed")
        moe = blocks[0][1]
        self.assertTrue(all(p.requires_grad for p in moe.parameters()))

    def test_hf_gemma3_layers_survive_partial_upcycle(self):
        try:
            from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        except Exception:
            self.skipTest("transformers with Gemma3 not installed")
        from poc import poc_04_gemma3_1b_frozen_upcycle as poc4

        torch.manual_seed(0)
        hf = Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=128, hidden_size=48, intermediate_size=96,
                                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                                head_dim=12, max_position_embeddings=64, sliding_window=16,
                                                attn_implementation="eager")).eval()
        tok = torch.randint(0, 128, (2, 24))
        with torch.no_grad():
            dense = hf(tok, use_cache=False).logits
        layer_types = [type(layer) for layer in hf.model.layers]
        cfg = EMeansMoEConfig(num_experts=4, top_k=2, upcycle_strategy="progressive_split",
                              progressive_active_experts=2, upcycle_init_noise=0.0)
        hf, blocks = poc4.freeze_backbone_and_upcycle_top_layers(hf, cfg, last_n=1)
        self.assertEqual([type(layer) for layer in hf.model.layers], layer_types)
        with torch.no_grad():
            moe = hf(tok, use_cache=False).logits
        self.assertLess((moe - dense).abs().max().item(), 1e-4)


@unittest.skipIf(torch is None, "torch not installed")
class TestTokenSources(unittest.TestCase):
    def test_synthetic_train_and_val_never_share_a_seed(self):
        s = SyntheticSource()
        a, da = s.batch(3, 4, 16, 128, "train")
        b, db = s.batch(3, 4, 16, 128, "train")
        self.assertTrue(torch.equal(a, b) and torch.equal(da, db))
        v, _ = s.batch(3, 4, 16, 128, "val")
        from moe_splitting.synthetic_data import make_multi_domain_batch
        self.assertTrue(torch.equal(v, make_multi_domain_batch(3 + VAL_SEED_OFFSET, 4, 16, 128)[0]))
        with self.assertRaises(ValueError):
            s.batch(VAL_SEED_OFFSET, 4, 16, 128, "train")
        with self.assertRaises(ValueError):
            s.batch(0, 4, 16, 128, "test")

    def test_shards_hold_out_their_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 1000
            for i, name in enumerate(("prose.npy", "code.npy")):
                np.save(pathlib.Path(tmp) / name, (np.arange(n) % 50 + 50 * i).astype(np.uint16))
            np.save(pathlib.Path(tmp) / "pos.npy", np.arange(n))  # not a domain shard; ignored
            src = ShardSource(tmp, val_frac=0.1)
            self.assertEqual(src.domain_names, ("prose", "code"))
            cut = int(n * 0.9)
            for seed in range(30):
                tr, dtr = src.batch(seed, 8, 20, 128, "train")
                va, dva = src.batch(seed, 8, 20, 128, "val")
                for rows, doms in ((tr, dtr), (va, dva)):
                    for row, d in zip(rows.tolist(), doms.tolist()):
                        self.assertTrue(all(50 * d <= t < 50 * (d + 1) for t in row), "domain id must match shard")
            # windows from the same seeds, reconstructed from position: train ends before the cut, val after
            pos_src = ShardSource(tmp, names=("pos.npy",), weights=(1.0,), val_frac=0.1)
            for seed in range(30):
                tr, _ = pos_src.batch(seed, 8, 20, 10_000, "train")
                va, _ = pos_src.batch(seed, 8, 20, 10_000, "val")
                self.assertLess(int(tr.max()), cut)
                self.assertGreaterEqual(int(va.min()), cut)
            with self.assertRaises(ValueError):
                src.batch(0, 40, 20, 60, "train")  # 40 rows include code rows, whose ids reach 99 >= vocab 60
            del src, pos_src  # release the memory maps, or Windows cannot delete the temp dir
            gc.collect()


@unittest.skipIf(torch is None, "torch not installed")
class TestHeldoutEval(unittest.TestCase):
    def test_eval_is_side_effect_free_and_tables_are_normalised(self):
        torch.manual_seed(0)
        model = upcycle_gemma_model(TinyGemmaForCausalLM(GemmaArchitectureSpec(**TINY_SPEC)),
                                    EMeansMoEConfig(num_experts=4, top_k=2))
        model.train()
        src = SyntheticSource()
        batches = [src.batch(v, 6, 16, 128, "val") for v in range(3)]
        ev = heldout_eval(model, batches, src.domain_names)
        self.assertTrue(model.training, "previous mode must be restored")
        for _, blk in iter_moe_blocks(model):
            self.assertIsNone(blk.last_router_output, "evaluation must not leave routing statistics behind")
        self.assertEqual(ev["val_tokens"], 3 * 6 * 15)
        self.assertTrue(0.0 <= ev["routing_domain_info"] <= 1.0)
        for table in ev["domain_expert_share"].values():
            for d, row in enumerate(table):
                present = any(int((b[1] == d).sum()) > 0 for b in batches)
                self.assertAlmostEqual(sum(row), 1.0 if present else 0.0, places=6)
        # per-domain losses average back to the overall loss, weighted by token counts
        counts = torch.zeros(3)
        for _, dom in batches:
            counts += torch.bincount(dom, minlength=3).float() * 15
        per = ev["per_domain_val_loss"]
        weighted = sum(counts[i].item() * per[n] for i, n in enumerate(src.domain_names) if per[n] is not None)
        self.assertAlmostEqual(weighted / counts.sum().item(), ev["val_loss"], places=5)

    def test_mutual_information_extremes(self):
        mi, hd = _mutual_information(torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float64))
        self.assertAlmostEqual(mi / hd, 1.0, places=9)
        mi, hd = _mutual_information(torch.tensor([[1.0, 3.0], [2.0, 6.0]], dtype=torch.float64))
        self.assertAlmostEqual(mi, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
