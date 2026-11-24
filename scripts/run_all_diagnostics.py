"""
One-Command Diagnostic Suite Runner
===================================

    python scripts/run_all_diagnostics.py [--out-dir DIR]

Runs every offline test module (stdlib + numpy), then every PyTorch test module
(tests/test_*torch*.py) through unittest in a subprocess, and saves a scorecard:
  - <out-dir>/diagnostic_suite_results.json   (default poc/results, or MOE_POC_RESULTS_DIR)
  - poc/logs/diagnostics/run_<timestamp>.log (+ latest.log)

If torch is not installed the torch suite is NOT counted as passed: `torch_all_passed` is
null, `complete` is false and the exit code is 2 (1 = a test failed, 0 = everything ran and
passed).
"""

import argparse
from datetime import datetime
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from moe_splitting.poc_logging import results_root, tee_poc_logs  # noqa: E402


def _modules(torch_modules: bool):
    names = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    return [f"tests.{n}" for n in names if ("torch" in n) == torch_modules]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the offline and PyTorch test suites")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    with tee_poc_logs("diagnostics"):
        print("=" * 84)
        print("MoE-DYNAMIC-EXPERT-SPLITTING: DIAGNOSTIC SUITE")
        print("=" * 84)
        offline = _modules(torch_modules=False)
        suite = unittest.TestLoader().loadTestsFromNames(offline)
        res = unittest.TextTestRunner(verbosity=2).run(suite)

        torch_modules = _modules(torch_modules=True)
        try:
            import torch  # noqa: F401
            have_torch = True
        except ImportError:
            have_torch = False
        if have_torch:
            proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *torch_modules],
                                  cwd=str(ROOT), capture_output=True, text=True, check=False)
            print("\n--- PyTorch test suite ---")
            print(proc.stdout)
            print(proc.stderr)
            torch_ok = proc.returncode == 0
            torch_status = f"ran {len(torch_modules)} modules (exit={proc.returncode})"
        else:
            torch_ok = None
            torch_status = "NOT RUN: torch is not installed"
            print(f"\n[Warn] {torch_status}; the scorecard is incomplete")

        complete = torch_ok is not None
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "offline_modules": offline,
            "offline_tests_run": res.testsRun,
            "offline_failures": len(res.failures),
            "offline_errors": len(res.errors),
            "offline_skipped": len(res.skipped),
            "offline_all_passed": res.wasSuccessful(),
            "torch_modules": torch_modules,
            "torch_suite_status": torch_status,
            "torch_all_passed": torch_ok,
            "complete": complete,
            "overall_passed": res.wasSuccessful() and torch_ok is True,
        }
        out_dir = results_root(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "diagnostic_suite_results.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n=== DIAGNOSTIC SCORECARD ===")
        print(json.dumps(summary, indent=2))
        print(f"Saved diagnostic report -> {out_path}")
        if not res.wasSuccessful() or torch_ok is False:
            return 1
        return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
