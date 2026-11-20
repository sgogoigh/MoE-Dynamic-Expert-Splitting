"""
One-Command Diagnostic Suite Runner (Syntax, Domain, Baselines & PoC Verification)
================================================================================

Runs the complete offline diagnostic battery and optionally runs PyTorch/pytest checks
if `torch` and `pytest` are installed, saving JSON telemetry & logs:
  - `poc/results/diagnostic_suite_results.json`
  - `poc/logs/diagnostics/run_<timestamp>.log` (+ `latest.log`)
"""

from datetime import datetime
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.poc_logging import tee_poc_logs  # noqa: E402


def main() -> int:
    with tee_poc_logs("diagnostics"):
        print("=" * 84)
        print("MoE-DYNAMIC-EXPERT-SPLITTING — FULL DIAGNOSTIC & BASELINE SUITE")
        print("=" * 84)

        loader = unittest.TestLoader()
        offline_suite = unittest.TestSuite()
        for test_file in sorted((ROOT / "tests").glob("test_*.py")):
            if "torch" in test_file.name:
                continue
            offline_suite.addTests(loader.discover(str(ROOT / "tests"), pattern=test_file.name))
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(offline_suite)

        torch_status = "skipped (torch/pytest not installed locally)"
        torch_ok = True
        try:
            import pytest  # noqa: F401
            import torch  # noqa: F401

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_moe_splitting_torch.py", "-v"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            print("\n--- PyTorch Test Suite Output ---")
            print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            torch_ok = proc.returncode == 0
            torch_status = f"ran (exit={proc.returncode})"
        except ImportError:
            print(f"\n[Info] Skipping PyTorch tests: {torch_status}")

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "offline_tests_run": res.testsRun,
            "offline_failures": len(res.failures),
            "offline_errors": len(res.errors),
            "offline_all_passed": res.wasSuccessful(),
            "torch_suite_status": torch_status,
            "torch_all_passed": torch_ok,
            "overall_passed": res.wasSuccessful() and torch_ok,
        }

        out_path = ROOT / "poc" / "results" / "diagnostic_suite_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\n=== DIAGNOSTIC SCORECARD ===")
        print(json.dumps(summary, indent=2))
        print(f"Saved diagnostic report -> {out_path}")
        return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
