"""
Rigorous Diagnostics, Baseline Correctness, Domain Invariants & Static Checks Suite.
Runs completely offline using Python stdlib + NumPy (no downloads, no GPU required).
"""

import ast
import json
import math
import pathlib
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from moe_splitting.config import EMeansMoEConfig, GEMMA_PRESETS  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.scaling_law import ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw  # noqa: E402
from poc import poc_01_math_and_profiler as poc1  # noqa: E402


class TestStaticAndArtifactIntegrity(unittest.TestCase):
    """Syntax, bytecode compilation, config JSONs, SVGs, and Markdown cross-link checks."""

    def test_all_python_files_parse_and_compile(self):
        py_files = sorted(ROOT.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 15)
        for p in py_files:
            src = p.read_text(encoding="utf-8")
            ast.parse(src, filename=str(p))
            compile(src, str(p), "exec")

    def test_all_json_configs_valid_and_domain_sane(self):
        cfg_dir = ROOT / "configs"
        json_files = sorted(cfg_dir.glob("*.json"))
        self.assertGreaterEqual(len(json_files), 3)
        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if "moe" in data:
                preset = data["model_preset"]
                self.assertIn(preset, GEMMA_PRESETS)
                spec = GEMMA_PRESETS[preset]
                moe = data["moe"]
                self.assertGreater(moe["top_k"], 0)
                self.assertLess(moe["top_k"], moe["num_experts"])
                self.assertGreater(moe["alpha_0"], moe["alpha_min"])
                self.assertGreater(moe["alpha_min"], 0.0)
                if moe.get("upcycle_strategy") == "sliced_granular":
                    self.assertEqual(spec.intermediate_size % moe["top_k"], 0)

    def test_all_svg_diagrams_well_formed_xml(self):
        svg_files = sorted((ROOT / "docs" / "assets" / "images").glob("*.svg"))
        self.assertEqual(len(svg_files), 3)
        for sf in svg_files:
            tree = ET.parse(str(sf))
            root_el = tree.getroot()
            self.assertTrue(root_el.tag.endswith("svg"))
            self.assertIn("viewBox", root_el.attrib)

    def test_all_markdown_relative_links_resolve(self):
        md_files = sorted(ROOT.rglob("*.md"))
        self.assertGreaterEqual(len(md_files), 10)
        link_re = re.compile(r"\[[^\]]*\]\(([^)#\s]+)")
        broken = []
        for mf in md_files:
            text = mf.read_text(encoding="utf-8")
            for m in link_re.finditer(text):
                target = m.group(1)
                if re.match(r"^(https?://|mailto:)", target):
                    continue
                resolved = (mf.parent / target).resolve()
                if not resolved.exists():
                    broken.append(f"{mf.relative_to(ROOT)} -> {target}")
        self.assertEqual(broken, [], f"Broken markdown links found: {broken}")


class TestGemmaAndMoEDomainInvariants(unittest.TestCase):
    """Domain-specific checks for Gemma configs, routing health metrics, and OLS scaling."""

    def test_gemma_presets_architectural_invariants(self):
        for name, spec in GEMMA_PRESETS.items():
            self.assertEqual(spec.hidden_size % spec.num_attention_heads, 0, name)
            self.assertEqual(spec.num_attention_heads % spec.num_key_value_heads, 0, name)
            self.assertEqual(spec.intermediate_size % 2, 0, name)
            self.assertEqual(spec.hidden_activation, "gelu_pytorch_tanh", name)
            self.assertGreater(spec.vocab_size, 0, name)

    def test_negative_ablation_without_bias_shift_violates_mass_invariance(self):
        """Confirm omitting -tau_r ln(2) breaks probability invariance significantly."""
        rng = np.random.default_rng(21)
        X = rng.normal(size=(32, 16))
        W = rng.normal(scale=0.5, size=(8, 16))
        b = rng.normal(scale=0.4, size=(8,))
        donor, dead = 1, 6
        b[donor] = 1.5
        b[dead] = -1e9

        p_before = poc1._softmax((X @ W.T + b[None, :]) / 1.0)
        W_bad = W.copy()
        b_bad = b.copy()
        W_bad[dead] = W_bad[donor]
        b_bad[dead] = b_bad[donor]  # Missing -tau_r * ln(2)!
        p_bad = poc1._softmax((X @ W_bad.T + b_bad[None, :]) / 1.0)

        err = float(np.max(np.abs((p_bad[:, donor] + p_bad[:, dead]) - p_before[:, donor])))
        self.assertGreater(err, 0.05)

    def test_routing_metrics_extreme_bounds(self):
        # Single-expert total collapse for Top-1
        c = np.array([1.0, 0.0, 0.0, 0.0])
        hc = compute_routing_health(c, top_k=1, dead_tau=0.1)
        self.assertEqual(hc["dead_count"], 3.0)
        self.assertAlmostEqual(hc["norm_entropy"], 0.0, places=6)
        self.assertAlmostEqual(hc["gini"], 0.75, places=6)
        self.assertAlmostEqual(hc["max_to_mean"], 4.0, places=6)

        # Uniform routing for Top-2 across E=8
        u = np.full(8, 2.0 / 8.0)
        hu = compute_routing_health(u, top_k=2, dead_tau=0.1)
        self.assertEqual(hu["dead_count"], 0.0)
        self.assertAlmostEqual(hu["norm_entropy"], 1.0, places=6)
        self.assertAlmostEqual(hu["gini"], 0.0, places=6)
        self.assertAlmostEqual(hu["max_to_mean"], 1.0, places=6)

    def test_cosine_floor_monotonic_bounds(self):
        cfg = EMeansMoEConfig(alpha_0=0.08, alpha_min=0.005)
        vals = [
            cfg.alpha_min + (cfg.alpha_0 - cfg.alpha_min) * 0.5 * (1.0 + math.cos(math.pi * s / 100))
            for s in range(101)
        ]
        self.assertAlmostEqual(vals[0], cfg.alpha_0, places=9)
        self.assertAlmostEqual(vals[-1], cfg.alpha_min, places=9)
        self.assertTrue(all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)))

    def test_scaling_law_recovers_expected_signs(self):
        rng = np.random.default_rng(9)
        obs = []
        for bl in (65536, 131072, 262144):
            for e in (8, 16, 32):
                for k in (1, 2):
                    for t in (0.5, 1.0, 2.0):
                        val = 0.01 * (bl ** 0.4) * (e ** -0.55) * (k ** 0.3) * (t ** -0.2)
                        obs.append(ScalingObservation(bl, e, k, t, val * float(np.exp(rng.normal(scale=0.02)))))
        fit = fit_alpha_min_powerlaw(obs)
        self.assertGreater(fit.a_bl, 0.0)
        self.assertLess(fit.a_e, 0.0)
        self.assertGreater(fit.a_k, 0.0)
        self.assertLess(fit.a_tau, 0.0)
        self.assertGreater(fit.r2_log, 0.99)
        loso = evaluate_loso(obs)
        self.assertLess(loso["loso_mean_rel_err"], 0.03)


class TestPoC1BaselinesAndCLISmoke(unittest.TestCase):
    """Multi-seed baseline correctness tests + end-to-end CLI smoke tests."""

    def test_poc01_multi_seed_baseline_dominance(self):
        for seed in (123, 456, 789):
            res_u = poc1.run_synthetic_simulation("unbalanced", steps=400, seed=seed)
            res_e = poc1.run_synthetic_simulation("emeans", steps=400, seed=seed)
            self.assertGreaterEqual(res_u["dead_experts"], 8, f"seed={seed}")
            self.assertGreater(res_u["max_to_mean"], 4.0, f"seed={seed}")
            self.assertGreaterEqual(res_e["splits"], 5, f"seed={seed}")
            self.assertLess(res_e["dead_experts"], res_u["dead_experts"], f"seed={seed}")
            self.assertGreater(res_e["norm_entropy"], res_u["norm_entropy"], f"seed={seed}")

    def test_poc01_simulation_is_deterministic(self):
        a = poc1.run_synthetic_simulation("deepseek_bias", steps=120, seed=5)
        b = poc1.run_synthetic_simulation("deepseek_bias", steps=120, seed=5)
        self.assertEqual(a, b)

    def test_poc01_topk_rows_separate_plain_from_sibling_routing(self):
        rows = poc1.run_topk_invariance()
        self.assertEqual({(r["tau_r"], r["top_k"]) for r in rows},
                         {(t, k) for t in (0.5, 1.0, 2.0) for k in (1, 2, 4)})
        self.assertLess(max(r["sibling_topk_max_output_diff"] for r in rows), 1e-12)
        self.assertGreater(min(r["plain_topk_max_output_diff"] for r in rows), 1e-3)

    def test_poc01_router_sync_counts_centroid_payload(self):
        g = next(x for x in poc1.profile_gemma_targets() if x["preset"] == "gemma-3-1b")
        spec = GEMMA_PRESETS["gemma-3-1b"]
        expected_kb = spec.num_hidden_layers * g["E"] * (spec.hidden_size + 4) * 4 / 1024.0
        self.assertAlmostEqual(g["router_stats_sync_KB_per_step"], round(expected_kb, 1), places=6)

    def test_all_poc_cli_entrypoints_write_only_to_the_given_dirs(self):
        """Every CLI runs with results and logs redirected to a temp dir; the repo is untouched."""
        import os
        import tempfile

        tracked = [ROOT / "poc" / "results" / n for n in ("poc_01_results.json", "COMPILED-POC-REPORT.md")]
        before = {t: (t.read_bytes() if t.exists() else None) for t in tracked}
        with tempfile.TemporaryDirectory() as tmp:
            res_dir, log_dir = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "logs"
            env = dict(os.environ, MOE_POC_RESULTS_DIR=str(res_dir), MOE_POC_LOGS_DIR=str(log_dir))
            cmds = [
                [sys.executable, "poc/poc_01_math_and_profiler.py"],
                [sys.executable, "poc/poc_02_tiny_gemma_continuity.py", "--dry-run"],
                [sys.executable, "poc/poc_03_proxy_collapse_and_scaling.py", "--dry-run"],
                [sys.executable, "poc/poc_04_gemma3_1b_frozen_upcycle.py", "--synthetic-gemma", "--upcycle-last-n", "2",
                 "--dry-run"],
                [sys.executable, "poc/compile_poc_report.py"],
                [sys.executable, "scripts/explore_gemma_architecture.py"],
                [sys.executable, "scripts/plan_scaling_sweep.py"],
            ]
            for cmd in cmds:
                proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False, env=env)
                self.assertEqual(proc.returncode, 0, f"{cmd}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
            p1_json = json.loads((res_dir / "poc_01_results.json").read_text(encoding="utf-8"))
            self.assertTrue(p1_json["diagnostics"]["all_passed"])
            self.assertIn("comparison_informational", p1_json)
            report = (res_dir / "COMPILED-POC-REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Top-K routing after the split", report)
            self.assertIn("PoC 2 *(not run yet)*", report)
            for sub in ("poc_01", "poc_02", "poc_03", "poc_04", "compile_report"):
                self.assertTrue((log_dir / sub / "latest.log").exists(), f"missing log for {sub}")
        after = {t: (t.read_bytes() if t.exists() else None) for t in tracked}
        self.assertEqual(before, after, "a PoC CLI wrote into poc/results despite MOE_POC_RESULTS_DIR")

    def test_poc03_invalid_ladders_are_rejected(self):
        from poc import poc_03_proxy_collapse_and_scaling as poc3

        for bad in (["--alpha-ladder", "0.01,0.02"], ["--alpha-ladder", "0.02,0.01,0.05"],
                    ["--alpha-ladder", "0.01,0.02,0.2", "--alpha-0", "0.1"], ["--ff-dim", "250", "--grid-top-k", "3"]):
            with self.assertRaises(SystemExit, msg=str(bad)):
                poc3.parse_args(bad)


class TestPoC3CellLogic(unittest.TestCase):
    """The flat / at-edge rules that decide which cells enter the alpha_min fit."""

    LADDER = [0.005, 0.02, 0.08]
    KEY = dict(num_experts=8, top_k=2, batch_size=8, tokens_per_step=256, tau_r=1.0)

    def _runs(self, losses_per_rung, seeds=1):
        runs = []
        for a, losses in zip(self.LADDER, losses_per_rung):
            for s in range(seeds):
                v = losses[s] if isinstance(losses, (list, tuple)) else losses
                runs.append({"alpha_min": a, "val_loss": v, "all_finite": v is not None})
        return runs

    def _cell(self, losses, seeds=1, tol=0.01):
        from poc import poc_03_proxy_collapse_and_scaling as poc3

        return poc3.summarize_cell(dict(self.KEY), self._runs(losses, seeds), self.LADDER, seeds, tol)

    def test_interior_optimum_is_usable(self):
        c = self._cell([3.10, 3.00, 3.08])
        self.assertTrue(c["usable"])
        self.assertEqual(c["best_alpha"], 0.02)

    def test_flat_ladder_is_excluded(self):
        c = self._cell([3.004, 3.000, 3.006])
        self.assertFalse(c["usable"])
        self.assertEqual(c["reason"], "flat")

    def test_edge_optimum_is_censored(self):
        c = self._cell([2.90, 3.00, 3.10])
        self.assertFalse(c["usable"])
        self.assertEqual(c["reason"], "at_edge")

    def test_seed_noise_raises_the_flat_threshold(self):
        # a 0.05 range clears the 0.01 tolerance, but the two seeds disagree by ~0.3
        c = self._cell([(3.05, 3.35), (3.00, 3.30), (3.05, 3.35)], seeds=2)
        self.assertEqual(c["reason"], "flat")
        self.assertGreater(c["noise_sem"], 0.1)

    def test_non_finite_run_excludes_the_cell(self):
        c = self._cell([3.1, None, 3.0])
        self.assertEqual(c["reason"], "non-finite run")

    def test_split_counts_are_recorded_per_rung_and_cell(self):
        from poc import poc_03_proxy_collapse_and_scaling as poc3

        runs = self._runs([3.004, 3.000, 3.006])
        for r, n in zip(runs, (0, 2, 4)):
            r["num_splits"] = n
        c = poc3.summarize_cell(dict(self.KEY), runs, self.LADDER, 1, 0.01)
        self.assertEqual([r["splits_mean"] for r in c["rungs"]], [0, 2, 4])
        self.assertEqual(c["total_splits"], 6)

        no_split = self._runs([3.0, 3.0, 3.0])
        for r in no_split:
            r["num_splits"] = 0
        cells = [poc3.summarize_cell(dict(self.KEY), no_split, self.LADDER, 1, 0.01), c,
                 self._cell([3.0, 3.0, 3.0])]                      # fixture without split counts
        self.assertIsNone(cells[2]["total_splits"])
        self.assertEqual(poc3.fit_usable_cells(cells)["n_no_splits"], 1)

    def test_fit_reports_why_it_was_skipped(self):
        from poc import poc_03_proxy_collapse_and_scaling as poc3

        fit = poc3.fit_usable_cells([self._cell([3.10, 3.00, 3.08]), self._cell([3.0, 3.0, 3.0])])
        self.assertEqual(fit["status"], "not_fitted")
        self.assertEqual((fit["n_usable"], fit["n_flat"]), (1, 1))
        self.assertIn("observations", fit["reason"])


class TestScalingLawIdentifiability(unittest.TestCase):
    @staticmethod
    def _obs(bls=(1024, 2048, 4096), es=(8, 16, 32), ks=(1, 2), taus=(1.0,)):
        out = []
        for bl in bls:
            for e in es:
                for k in ks:
                    for t in taus:
                        out.append(ScalingObservation(bl, e, k, t, 0.01 * bl ** 0.3 * e ** -0.5 * k ** 0.2))
        return out

    def test_constant_feature_is_unidentified_and_prediction_refuses_it(self):
        fit = fit_alpha_min_powerlaw(self._obs())
        self.assertEqual(fit.unidentified, ("tau_r",))
        self.assertTrue(math.isnan(fit.a_tau))
        self.assertAlmostEqual(fit.a_bl, 0.3, places=6)
        self.assertAlmostEqual(fit.a_e, -0.5, places=6)
        self.assertGreater(fit.predict(2048, 16, 2, 1.0), 0.0)
        with self.assertRaises(ValueError):
            fit.predict(2048, 16, 2, 2.0)

    def test_collinear_design_raises(self):
        obs = [ScalingObservation(bl, bl / 128, 2.0, 1.0, 0.01 * bl ** 0.1) for bl in (1024, 2048, 4096, 8192)]
        with self.assertRaises(ValueError):
            fit_alpha_min_powerlaw(obs)

    def test_too_few_observations_raise(self):
        with self.assertRaises(ValueError):
            fit_alpha_min_powerlaw(self._obs(bls=(1024, 2048), es=(8,), ks=(1,)))

    def test_r2_is_nan_when_target_is_constant(self):
        obs = [ScalingObservation(bl, e, 2.0, 1.0, 0.01) for bl in (1024, 2048, 4096) for e in (8, 16)]
        self.assertTrue(math.isnan(fit_alpha_min_powerlaw(obs).r2_log))

    def test_loso_skips_folds_that_leave_e_unidentified(self):
        loso = evaluate_loso(self._obs(es=(8, 16)))
        self.assertEqual(loso["n_folds_evaluated"], 0)
        self.assertEqual(len(loso["skipped"]), 2)
        loso3 = evaluate_loso(self._obs())
        self.assertEqual(loso3["n_folds_evaluated"], 3)
        self.assertLess(loso3["loso_mean_rel_err"], 1e-6)


class TestReportCompiler(unittest.TestCase):
    def test_compiler_renders_new_schemas_and_marks_missing_pocs(self):
        import os
        import tempfile

        from poc import compile_poc_report as comp

        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            p4 = {"poc_id": "poc_04", "model_id": "m", "upcycled_layers": ["layers.2.mlp"], "total_params_M": 7.0,
                  "trainable_params_M": 6.0,
                  "diagnostics": {"sanity": {"only_moe_params_trainable": True, "triggered_splits": False},
                                  "sanity_passed": False,
                                  "informational": {"heldout_loss_dense": 7.6, "heldout_loss_step0": 7.6,
                                                    "heldout_loss_end": None, "upcycle_parity_gap": 1e-4,
                                                    "dead_experts_start": 12.0, "dead_experts_end": 12.0,
                                                    "num_splits": 0, "num_unmerges": 0,
                                                    "routing_domain_info_end": None}}}
            (d / "poc_04_results.json").write_text(json.dumps(p4), encoding="utf-8")
            prev = os.environ.get("MOE_POC_LOGS_DIR")
            os.environ["MOE_POC_LOGS_DIR"] = str(d / "logs")
            try:
                out = comp.main(["--results-dir", str(d)])
            finally:
                if prev is None:
                    os.environ.pop("MOE_POC_LOGS_DIR", None)
                else:
                    os.environ["MOE_POC_LOGS_DIR"] = prev
            text = out.read_text(encoding="utf-8")
        self.assertIn("Sanity checks: FAIL", text)
        self.assertIn("`triggered_splits`: FAIL", text)
        self.assertIn("PoC 1 *(not run yet)*", text)
        self.assertIn("after training n/a", text)


if __name__ == "__main__":
    unittest.main()
