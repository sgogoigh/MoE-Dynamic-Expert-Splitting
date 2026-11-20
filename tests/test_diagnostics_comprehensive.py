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

    def test_all_poc_cli_entrypoints_and_logs(self):
        cmds = [
            [sys.executable, "poc/poc_01_math_and_profiler.py"],
            [sys.executable, "poc/poc_02_tiny_gemma_continuity.py", "--dry-run"],
            [sys.executable, "poc/poc_03_proxy_collapse_and_scaling.py", "--dry-run"],
            [sys.executable, "poc/poc_04_gemma3_1b_frozen_upcycle.py", "--synthetic-gemma", "--upcycle-last-n", "8", "--dry-run"],
            [sys.executable, "poc/compile_poc_report.py"],
            [sys.executable, "scripts/explore_gemma_architecture.py"],
            [sys.executable, "scripts/plan_scaling_sweep.py"],
        ]
        for cmd in cmds:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"Command failed: {cmd}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

        # Verify structured PoC 1 outputs, diagnostics, and logs exist cleanly
        p1_json = json.loads((ROOT / "poc" / "results" / "poc_01_results.json").read_text(encoding="utf-8"))
        self.assertTrue(p1_json["diagnostics"]["all_passed"])
        for sub in ("poc_01", "poc_02", "poc_03", "poc_04", "compile_report"):
            self.assertTrue((ROOT / "poc" / "logs" / sub / "latest.log").exists(), f"Missing log for {sub}")
        self.assertTrue((ROOT / "poc" / "results" / "COMPILED-POC-REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
