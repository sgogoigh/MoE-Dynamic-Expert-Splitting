"""Offline mathematical unit tests (runnable right away using standard library + NumPy)."""

import math
import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import GEMMA_PRESETS  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.scaling_law import ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw  # noqa: E402


class TestMoEMathInvariants(unittest.TestCase):
    def test_softmax_mass_invariance_arbitrary_temperature(self):
        rng = np.random.default_rng(11)
        T, d, E = 40, 16, 12
        X = rng.normal(size=(T, d))
        W = rng.normal(size=(E, d))
        b = rng.normal(size=(E,))
        donor, dead = 2, 9
        b[dead] = -1e9
        b[donor] = 2.1

        for tau in (0.3, 1.0, 3.0):
            z0 = (X @ W.T + b[None, :]) / tau
            p0 = np.exp(z0 - z0.max(axis=-1, keepdims=True))
            p0 /= p0.sum(axis=-1, keepdims=True)

            W1 = W.copy()
            b1 = b.copy()
            W1[dead] = W1[donor]
            shifted = b1[donor] - tau * math.log(2.0)
            b1[donor] = shifted
            b1[dead] = shifted

            z1 = (X @ W1.T + b1[None, :]) / tau
            p1 = np.exp(z1 - z1.max(axis=-1, keepdims=True))
            p1 /= p1.sum(axis=-1, keepdims=True)

            self.assertLess(np.max(np.abs((p1[:, donor] + p1[:, dead]) - p0[:, donor])), 1e-12)

    def test_routing_health_metrics(self):
        balanced = np.full(8, 2.0 / 8.0)
        h_bal = compute_routing_health(balanced, top_k=2, dead_tau=0.1)
        self.assertAlmostEqual(h_bal["max_to_mean"], 1.0, places=6)
        self.assertEqual(h_bal["dead_count"], 0.0)
        self.assertAlmostEqual(h_bal["norm_entropy"], 1.0, places=6)

        collapsed = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h_col = compute_routing_health(collapsed, top_k=2, dead_tau=0.1)
        self.assertEqual(h_col["dead_count"], 7.0)
        self.assertAlmostEqual(h_col["max_to_mean"], 8.0, places=6)

    def test_scaling_law_fit_and_loso(self):
        obs = []
        for bl in (1e5, 2e5, 4e5):
            for e in (8, 16, 32):
                for k in (1, 2):
                    for tau in (0.5, 1.0):
                        val = 0.01 * (bl ** 0.3) * (e ** -0.5) * (k ** 0.25) * (tau ** -0.15)
                        obs.append(ScalingObservation(bl, e, k, tau, val))
        fit = fit_alpha_min_powerlaw(obs)
        self.assertGreater(fit.r2_log, 0.999999)
        loso = evaluate_loso(obs)
        self.assertLess(loso["loso_log_rmse"], 1e-6)

    def test_gemma_presets_present(self):
        for k in ("tiny-gemma-debug", "gemma-3-1b", "gemma-2-2b", "gemma-2-9b"):
            self.assertIn(k, GEMMA_PRESETS)


if __name__ == "__main__":
    unittest.main()
