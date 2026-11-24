"""Top-K function preservation of the dynamic split (sibling routing), measured on real tensors.

Run:  python -m unittest tests.test_topk_invariance_torch -v      (or pytest)
"""

import copy
import math
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch-less environments run the NumPy suites only
    torch = None

if torch is not None:
    from moe_splitting import (
        DynamicExpertSplitter,
        EMeansMoEConfig,
        EMeansMoERouter,
        GEMMA_PRESETS,
        GeGLUMoEExperts,
        TinyGemmaForCausalLM,
        frozen_moe_stats,
        upcycle_gemma_model,
    )
    from moe_splitting.gemma_upcycle import GemmaMoEMLPBlock

requires_torch = unittest.skipIf(torch is None, "torch not installed")


# ----------------------------------------------------------------------------- helpers
def make_router(E, K, d, tau=1.0, seed=0, **cfg_kw):
    base = dict(num_experts=E, top_k=K, tau_r=tau, split_interval=1, cooldown_steps=0, alpha_0=0.5, alpha_min=0.5)
    base.update(cfg_kw)
    cfg = EMeansMoEConfig(**base)
    torch.manual_seed(seed)
    r = EMeansMoERouter(d, cfg)
    with torch.no_grad():
        r.gate.weight.normal_(0.0, 0.3)
        r.gate.bias.normal_(0.0, 0.3)
    return r


def make_tokens(T, d, seed, offset=0.5):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(T, d, generator=g) + offset


def warm(router, x, steps=40):
    """Feed the same batch until the EMA statistics are converged (alpha = 0.5)."""
    for s in range(steps):
        o = router(x)
        router.update_ema(o.batch_load, torch.ones(router.E), s, 10_000, sync_distributed=False,
                          batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)


def moe_out(router, experts, x):
    with torch.no_grad():
        o = router(x)
        y, _ = experts(x, o.topk_idx, o.topk_weights)
    return y, o


def rel_change(y1, y0):
    return (y1 - y0).norm(dim=-1) / y0.norm(dim=-1).clamp_min(1e-12)


def split(router, experts, dead, donor, seed=7):
    return DynamicExpertSplitter()._execute_single_split(router, experts, None, dead_idx=dead, donor_idx=donor, pair_seed=seed)


def setup_split_case(E, K, tau, seed, d=24, T=2048, dead_bias=-30.0, **cfg_kw):
    r = make_router(E, K, d, tau=tau, seed=seed, **cfg_kw)
    dead, donor = E - 1, 1
    with torch.no_grad():
        r.gate.bias[dead] = dead_bias
        r.gate.bias[donor] += 1.0        # overloaded donor
    ex = GeGLUMoEExperts(E, d, 16)
    x = make_tokens(T, d, seed)
    warm(r, x)
    return r, ex, x, dead, donor


# ----------------------------------------------------------------------------- tests
@requires_torch
class TestSiblingTopKInvariance(unittest.TestCase):
    def test_plain_split_breaks_topk_output_negative_control(self):
        """The old -tau*ln2 split (no sibling routing) changes every donor-routed token."""
        for K in (1, 2, 3):
            r, ex, x, dead, donor = setup_split_case(8, K, 1.0, seed=K, sibling_routing=False)
            y0, o0 = moe_out(r, ex, x)
            split(r, ex, dead, donor)
            y1, _ = moe_out(r, ex, x)
            donor_tok = (o0.topk_idx == donor).any(-1)
            rc = rel_change(y1, y0)
            self.assertGreater(donor_tok.float().mean().item(), 0.05)
            self.assertGreater((rc[donor_tok] > 1e-3).float().mean().item(), 0.95, f"K={K}")
            self.assertGreater(rc[donor_tok].median().item(), 0.05, f"K={K}")

    def test_sibling_split_preserves_topk_output_grid(self):
        """K in 1..4, tau in {0.5, 1, 2}, E in {6, 16}, 2 seeds: max per-token change < 1e-4."""
        worst = 0.0
        for E in (6, 16):
            for K in (1, 2, 3, 4):
                for tau in (0.5, 1.0, 2.0):
                    for seed in (0, 1):
                        r, ex, x, dead, donor = setup_split_case(E, K, tau, seed)
                        y0, o0 = moe_out(r, ex, x)
                        split(r, ex, dead, donor)
                        y1, o1 = moe_out(r, ex, x)
                        rc = rel_change(y1, y0)
                        worst = max(worst, rc.max().item())
                        self.assertLess(rc.max().item(), 1e-4, f"E={E} K={K} tau={tau} seed={seed}")
                        # the clone really receives part of the donor's tokens
                        donor_tok = (o0.topk_idx == donor).any(-1)
                        if donor_tok.sum() > 50:
                            to_clone = (o1.topk_idx == dead).any(-1)[donor_tok].float().mean().item()
                            self.assertGreater(to_clone, 0.2, f"E={E} K={K} tau={tau} seed={seed}")
                            self.assertLess(to_clone, 0.8, f"E={E} K={K} tau={tau} seed={seed}")
        print(f"\n  [sibling grid] worst per-token relative output change: {worst:.2e}")

    def test_sibling_split_without_router_noise_is_exact_to_fp32(self):
        for K in (1, 2, 4):
            r, ex, x, dead, donor = setup_split_case(8, K, 1.0, seed=3, router_noise_eps=0.0)
            y0, _ = moe_out(r, ex, x)
            split(r, ex, dead, donor)
            y1, _ = moe_out(r, ex, x)
            self.assertLess(rel_change(y1, y0).max().item(), 2e-6, f"K={K}")

    def test_sibling_routing_equals_plain_topk_without_pairs(self):
        for K in (1, 2, 3):
            r = make_router(10, K, 16, seed=K)
            x = make_tokens(1024, 16, K)
            o_sib = r(x)
            r.cfg.sibling_routing = False
            o_plain = r(x)
            self.assertTrue(torch.equal(o_sib.topk_idx, o_plain.topk_idx))
            self.assertTrue(torch.equal(o_sib.topk_weights, o_plain.topk_weights))
            self.assertTrue(torch.all(o_sib.pair_disagree == 0))

    def test_realistic_dead_expert_only_its_own_tokens_change(self):
        """Dead slot with small but non-zero mass: with renormalised K>=2 only tokens that used it change;
        with K=1 (weight = p_top1) the change is bounded by the dead expert's softmax mass."""
        for K in (1, 2, 3):
            r, ex, x, dead, donor = setup_split_case(8, K, 1.0, seed=10 + K, dead_bias=-2.0)
            y0, o0 = moe_out(r, ex, x)
            p_dead = o0.probs[:, dead]
            split(r, ex, dead, donor)
            y1, _ = moe_out(r, ex, x)
            rc = rel_change(y1, y0)
            used_dead = (o0.topk_idx == dead).any(-1)
            if K >= 2:
                self.assertLess(rc[~used_dead].max().item(), 1e-4, f"K={K}")
            else:
                bound = p_dead / (1 - p_dead) + 1e-4
                self.assertTrue(bool(torch.all(rc[~used_dead] <= bound[~used_dead])), f"K={K}")

    def test_router_offset_error_is_second_order(self):
        """On tokens whose group routing is unchanged the output change scales ~ eps^2 (cosh term);
        group flips (near Top-K ties) are rare; |r.(x-mu)|/tau is ~ eps at split time."""
        errs = {}
        for eps in (0.05, 0.1, 0.2):
            r, ex, x, dead, donor = setup_split_case(8, 2, 1.0, seed=5, router_noise_eps=eps)
            y0, o0 = moe_out(r, ex, x)
            split(r, ex, dead, donor)
            y1, o1 = moe_out(r, ex, x)
            groups_after = torch.where(o1.topk_idx == dead, torch.full_like(o1.topk_idx, donor), o1.topk_idx)
            same = (groups_after.sort(-1).values == o0.topk_idx.sort(-1).values).all(-1)
            self.assertGreater(same.float().mean().item(), 0.99, f"eps={eps}: too many Top-K group flips")
            errs[eps] = (y1 - y0).norm(dim=-1)[same].mean().item()
            donor_tok = (o0.topk_idx == donor).any(-1)
            logits = (x @ r.gate.weight.t() + r.gate.bias) / r.tau_r
            u = 0.5 * (logits[:, donor] - logits[:, dead])[donor_tok]
            ratio = u.abs().median().item() / eps
            self.assertGreater(ratio, 0.2, f"eps={eps}")
            self.assertLess(ratio, 3.0, f"eps={eps}")
        for a, b in ((0.05, 0.1), (0.1, 0.2)):
            growth = errs[b] / max(errs[a], 1e-30)
            self.assertGreater(growth, 3.2, f"{errs}")
            self.assertLess(growth, 4.8, f"{errs}")

    def test_centroid_hyperplane_balances_siblings(self):
        """Through the donor centroid the siblings split the donor's tokens ~evenly; through the
        origin (old behaviour) the split is driven by the offset of the token cloud."""
        imbalance = {True: [], False: []}
        for through in (True, False):
            for seed in range(6):
                r = make_router(8, 2, 16, seed=seed, split_through_centroid=through)
                dead = 7
                with torch.no_grad():
                    r.gate.bias[dead] = -30.0
                ex = GeGLUMoEExperts(8, 16, 8)
                x = make_tokens(4096, 16, seed, offset=3.0)
                warm(r, x)
                donor = int(r.ema_load.argmax())
                _, o0 = moe_out(r, ex, x)
                split(r, ex, dead, donor, seed=seed)
                _, o1 = moe_out(r, ex, x)
                donor_tok = (o0.topk_idx == donor).any(-1)
                frac = (o1.topk_idx == dead).any(-1)[donor_tok].float().mean().item()
                imbalance[through].append(abs(frac - 0.5))
                if through:
                    self.assertLess(abs(frac - 0.5), 0.15, f"seed={seed} frac={frac}")
        self.assertGreater(sum(imbalance[False]) / 6, 2 * sum(imbalance[True]) / 6, f"{imbalance}")

    def test_never_routes_to_both_siblings(self):
        r, ex, x, dead, donor = setup_split_case(8, 3, 1.0, seed=2)
        split(r, ex, dead, donor)
        o = r(x)
        both = (o.topk_idx == dead).any(-1) & (o.topk_idx == donor).any(-1)
        self.assertEqual(int(both.sum().item()), 0)

    def test_straight_through_value_and_gradient(self):
        """Value = group mass P; gradient = gradient of the chosen sibling's probability."""
        r = make_router(6, 2, 12, seed=4, renorm_topk_weights=False)
        with torch.no_grad():
            r.partner[1], r.partner[4] = 4, 1
        x = make_tokens(512, 12, 4)
        c = torch.randn(512, 2, generator=torch.Generator().manual_seed(9))

        o = r(x)
        probs = F.softmax(F.linear(x, r.gate.weight, r.gate.bias) / r.tau_r, dim=-1)
        paired = (o.topk_idx == 1) | (o.topk_idx == 4)
        group_mass = probs[:, 1:2] + probs[:, 4:5]
        expected = torch.where(paired, group_mass.expand_as(paired), probs.gather(1, o.topk_idx))
        self.assertTrue(torch.allclose(o.topk_weights, expected, atol=1e-6))
        self.assertGreater(paired.any(-1).float().mean().item(), 0.1)

        (o.topk_weights * c).sum().backward()
        g_st = r.gate.weight.grad.clone()
        r.gate.weight.grad = None
        probs2 = F.softmax(F.linear(x, r.gate.weight, r.gate.bias) / r.tau_r, dim=-1)
        (probs2.gather(1, o.topk_idx) * c).sum().backward()
        self.assertTrue(torch.allclose(g_st, r.gate.weight.grad, atol=1e-6))


@requires_torch
class TestUnmerge(unittest.TestCase):
    def _separated_pair(self, sep):
        r = make_router(8, 2, 16, seed=11, unmerge_min_steps=5, unmerge_tol=0.01, merge_max_steps=0)
        x = make_tokens(4096, 16, 11)
        warm(r, x)
        a, b = 1, 6
        mu = r.centroid(a)
        direction = torch.randn(16, generator=torch.Generator().manual_seed(3))
        direction = direction / direction.norm()
        with torch.no_grad():
            row, bias = r.gate.weight[a].clone(), float(r.gate.bias[a])
            R = sep * direction
            r.gate.weight[a] = row + R
            r.gate.weight[b] = row - R
            r.gate.bias[a] = bias - math.log(2) - float(R @ mu)
            r.gate.bias[b] = bias - math.log(2) + float(R @ mu)
            r.partner[a], r.partner[b] = b, a
            r.ema_pair_disagree[a] = r.ema_pair_disagree[b] = -1.0
        return r, x, a, b

    def test_separable_pair_unmerges_and_output_change_is_bounded(self):
        r, x, a, b = self._separated_pair(sep=60.0)
        ex = GeGLUMoEExperts(8, 16, 8)
        splitter = DynamicExpertSplitter()
        events = []
        for s in range(1, 8):
            o = r(x)
            r.update_ema(o.batch_load, torch.ones(8), s, 10_000, sync_distributed=False,
                         batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)
            events += splitter.maybe_unmerge_layer(r, s)
            if events:
                break
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertFalse(ev.forced)
        self.assertEqual({ev.expert_a, ev.expert_b}, {a, b})
        self.assertGreaterEqual(ev.merge_age, 5)
        self.assertLess(ev.ema_disagreement, 0.01)
        self.assertTrue(torch.equal(r.partner, torch.arange(8)))

        # measured disagreement bounds the real output change of the un-merge
        r2, _, _, _ = self._separated_pair(sep=60.0)
        y_grouped, o_g = moe_out(r2, ex, x)
        with torch.no_grad():
            r2.partner.copy_(torch.arange(8))
        y_plain, _ = moe_out(r2, ex, x)
        h_max = max(ex(x, torch.full((x.shape[0], 1), j), torch.ones(x.shape[0], 1))[0].norm(dim=-1).max().item()
                    for j in range(8))
        mean_change = (y_plain - y_grouped).norm(dim=-1).mean().item()
        measured = o_g.pair_disagree[a].item()
        self.assertLessEqual(mean_change, measured * h_max + 1e-6)

    def test_forced_unmerge_ramp_is_linear_and_ends_at_plain_topk(self):
        N = 10
        r, ex, x, dead, donor = setup_split_case(8, 2, 1.0, seed=7, unmerge_min_steps=1, unmerge_tol=1e-6,
                                                 merge_max_steps=2, unmerge_ramp_steps=N)
        split(r, ex, dead, donor)
        splitter = DynamicExpertSplitter()
        outs, events, widths = [], [], []
        for s in range(1, 2 + N + 2):
            y, o = moe_out(r, ex, x)
            outs.append(y)
            widths.append(o.topk_idx.shape[1])
            if o.topk_idx.shape[1] > 2:
                valid = o.topk_idx >= 0
                self.assertTrue(bool(torch.all(o.topk_weights[~valid] == 0)))
                for row, ok in zip(o.topk_idx[:64], valid[:64]):
                    ids = row[ok].tolist()
                    self.assertEqual(len(ids), len(set(ids)))      # no expert twice for a token
            r.update_ema(o.batch_load, torch.ones(8), s, 10_000, sync_distributed=False,
                         batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)
            events += splitter.maybe_unmerge_layer(r, s)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].forced)
        self.assertTrue(torch.equal(r.partner, torch.arange(8)))
        self.assertIn(4, widths)                                   # 2K slots used during the ramp
        steps = [(outs[i + 1] - outs[i]).norm(dim=-1).max().item() for i in range(len(outs) - 1)]
        total = (outs[-1] - outs[0]).norm(dim=-1).max().item()
        scale = outs[0].norm(dim=-1).max().item()
        self.assertGreater(total / scale, 0.05)                    # the un-merge does change the function
        self.assertLess(max(steps), 1.5 * total / N + 1e-6, f"per-step {max(steps)} vs total {total}")
        r_plain = copy.deepcopy(r)
        r_plain.cfg = copy.deepcopy(r.cfg)
        r_plain.cfg.sibling_routing = False
        y_plain, _ = moe_out(r_plain, ex, x)
        self.assertTrue(torch.allclose(outs[-1], y_plain, atol=1e-6))

    def test_overlapping_forced_unmerges_ramp_one_at_a_time_without_jumps(self):
        """Regression: with a second merged pair present, the ramp target must keep that pair merged,
        otherwise the routing snaps at the end of the ramp (seen in PoC 2: +1.9 nats at completion)."""
        N = 8
        r, ex, x, dead, donor = setup_split_case(8, 2, 1.0, seed=12, unmerge_min_steps=1, unmerge_tol=1e-6,
                                                 merge_max_steps=4, unmerge_ramp_steps=N)
        split(r, ex, dead, donor)                                   # pair A at step 0
        splitter = DynamicExpertSplitter()
        outs, events = [], []
        for s in range(1, 4 * N):
            if s == 2:                                              # pair B two steps later
                free = [j for j in range(8) if j not in (dead, donor)]
                load = r.ema_load
                d2 = max(free, key=lambda j: float(load[j]))
                c2 = min((j for j in free if j != d2), key=lambda j: float(load[j]))
                split(r, ex, c2, d2, seed=5)
            y, o = moe_out(r, ex, x)
            outs.append(y)
            lam = r.unmerge_lambda[r.unmerge_lambda > 0]
            if lam.numel():
                self.assertTrue(bool(torch.all(lam == lam[0])), "concurrent ramps with different lambdas")
            r.update_ema(o.batch_load, torch.ones(8), s, 10_000, sync_distributed=False,
                         batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)
            events += splitter.maybe_unmerge_layer(r, s)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.forced for e in events))
        self.assertGreaterEqual(events[1].step - events[0].step, N)  # B waited for A's ramp
        totals = []
        for ev in events:
            start = ev.step - (N - 1)                              # lambda = 1/N set at the end of `start`
            window = [outs[i] for i in range(start - 1, ev.step + 1)]
            totals.append((window[-1] - window[0]).norm(dim=-1).max().item())
            per_step = max((window[i + 1] - window[i]).norm(dim=-1).max().item() for i in range(len(window) - 1))
            self.assertLess(per_step, 1.01 * totals[-1] / N + 1e-7, f"ramp ending at step {ev.step}")
        # every other step is static, except outs[0] -> outs[1], where pair B's split replaces a live
        # (low-load, not dead) expert and its tokens legitimately change
        all_steps = max((outs[i + 1] - outs[i]).norm(dim=-1).max().item() for i in range(1, len(outs) - 1))
        self.assertLess(all_steps, 1.01 * max(totals) / N + 1e-6)

    def test_fresh_split_is_not_separable_and_is_force_unmerged(self):
        r, ex, x, dead, donor = setup_split_case(8, 2, 1.0, seed=6, unmerge_min_steps=2, unmerge_tol=0.01,
                                                 merge_max_steps=6, unmerge_ramp_steps=0)
        split(r, ex, dead, donor)
        splitter = DynamicExpertSplitter()
        for s in range(1, 7):
            o = r(x)
            r.update_ema(o.batch_load, torch.ones(8), s, 10_000, sync_distributed=False,
                         batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)
            ev = splitter.maybe_unmerge_layer(r, s)
            self.assertGreater(r.ema_pair_disagree[donor].item() if not ev else ev[0].ema_disagreement, 0.01)
            if s < 6:
                self.assertEqual(ev, [], f"step {s}")
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0].forced)
        self.assertEqual(ev[0].merge_age, 6)


@requires_torch
class TestSplitSelectionAndStats(unittest.TestCase):
    def test_merged_experts_are_never_dead_or_donor(self):
        r = make_router(8, 2, 16, seed=0)
        ex = GeGLUMoEExperts(8, 16, 8)
        with torch.no_grad():
            r.ema_load.copy_(torch.tensor([0.9, 0.3, 0.3, 0.2, 0.2, 0.0, 0.0, 0.1]))
        split(r, ex, dead=5, donor=0)
        with torch.no_grad():   # make the merged pair look like the best dead slot / donor again
            r.ema_load[5] = 0.0
            r.ema_load[0] = 2.0
            r.ema_energy.fill_(1.0)
        plan = DynamicExpertSplitter()._select_split_pairs(r, step=1)
        self.assertTrue(plan)
        for dead, donor, _ in plan:
            self.assertNotIn(dead, (0, 5))
            self.assertNotIn(donor, (0, 5))

    def test_group_count_guard_keeps_at_least_k_groups(self):
        r = make_router(4, 3, 8, seed=0)
        ex = GeGLUMoEExperts(4, 8, 8)
        with torch.no_grad():
            r.ema_load.copy_(torch.tensor([2.0, 1.0, 0.0, 0.0]))
        splitter = DynamicExpertSplitter()
        plan = splitter._select_split_pairs(r, step=1)
        self.assertEqual(len(plan), 1)                  # 4 groups - K(3) = 1 merge allowed
        dead, donor, seed = plan[0]
        split(r, ex, dead, donor, seed)
        self.assertEqual(r.num_groups(), 3)
        self.assertEqual(splitter._select_split_pairs(r, step=2), [])
        o = r(make_tokens(256, 8, 0))
        self.assertTrue(bool(torch.all(o.topk_weights > 0)))

    def test_ema_centroid_and_spread_equal_routed_statistics(self):
        r = make_router(6, 2, 10, seed=1, alpha_0=1.0, alpha_min=1.0)
        x = make_tokens(777, 10, 1)
        o = r(x)
        r.update_ema(o.batch_load, torch.ones(6), 0, 10, sync_distributed=False,
                     batch_sum=o.batch_sum, batch_sqnorm=o.batch_sqnorm)
        for j in range(6):
            sel = (o.topk_idx == j).any(-1)
            if sel.sum() < 2:
                continue
            xs = x[sel]
            self.assertTrue(torch.allclose(r.centroid(j), xs.mean(0), atol=1e-5))
            spread = ((xs - xs.mean(0)) ** 2).sum(-1).mean().item()
            self.assertAlmostEqual(r.token_spread(j), spread, delta=1e-3 * max(spread, 1.0))

    def test_split_halves_statistics_and_keeps_centroid(self):
        r, ex, x, dead, donor = setup_split_case(8, 2, 1.0, seed=8)
        load, mu, spread = r.ema_load[donor].item(), r.centroid(donor).clone(), r.token_spread(donor)
        bias0 = r.gate.bias[donor].item()
        split(r, ex, dead, donor)
        self.assertAlmostEqual(r.ema_load[donor].item(), load / 2, places=6)
        self.assertAlmostEqual(r.ema_load[dead].item(), load / 2, places=6)
        self.assertTrue(torch.allclose(r.centroid(dead), mu, atol=1e-5))
        self.assertTrue(torch.allclose(r.centroid(donor), mu, atol=1e-5))
        self.assertAlmostEqual(r.token_spread(dead), spread, delta=1e-4 * max(spread, 1.0))
        self.assertEqual(int(r.partner[donor]), dead)
        self.assertEqual(int(r.partner[dead]), donor)
        # biases: b_d + b_c = 2 (b - tau ln2) and b_c - b_d = 2 r.mu with r = (W_d - W_c) / 2
        b_d, b_c = r.gate.bias[donor].item(), r.gate.bias[dead].item()
        self.assertAlmostEqual(b_d + b_c, 2 * (bias0 - r.tau_r * math.log(2.0)), places=5)
        r_vec = 0.5 * (r.gate.weight[donor] - r.gate.weight[dead])
        self.assertAlmostEqual(b_c - b_d, 2 * float(r_vec @ mu), places=5)
        self.assertGreater(r_vec.norm().item(), 0.0)


@requires_torch
class TestPrecision(unittest.TestCase):
    def test_router_stays_fp32_under_model_cast(self):
        cfg = EMeansMoEConfig(num_experts=6, top_k=2)
        blk = GemmaMoEMLPBlock(16, 32, cfg).to(torch.bfloat16)
        self.assertEqual(blk.experts.gate_proj.dtype, torch.bfloat16)
        for name, t in list(blk.router.named_parameters()) + list(blk.router.named_buffers()):
            if t.is_floating_point():
                self.assertEqual(t.dtype, torch.float32, name)
        y = blk(torch.randn(2, 5, 16, dtype=torch.bfloat16))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(y.float()).all()))

    def test_router_logits_fp32_under_cpu_autocast(self):
        r = make_router(8, 2, 16, seed=2)
        x = make_tokens(256, 16, 2)
        ref = F.softmax(F.linear(x, r.gate.weight, r.gate.bias) / r.tau_r, dim=-1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o = r(x)
        self.assertEqual(o.probs.dtype, torch.float32)
        self.assertTrue(torch.allclose(o.probs, ref, atol=1e-6))

    def test_sibling_split_with_bf16_experts(self):
        for sibling, check in ((True, "small"), (False, "large")):
            r, _, x, dead, donor = setup_split_case(8, 2, 1.0, seed=9, sibling_routing=sibling)
            ex = GeGLUMoEExperts(8, 24, 16, dtype=torch.bfloat16)
            xb = x.to(torch.bfloat16)
            y0, o0 = moe_out(r, ex, xb)
            split(r, ex, dead, donor)
            y1, _ = moe_out(r, ex, xb)
            donor_tok = (o0.topk_idx == donor).any(-1)
            rc = rel_change(y1.float(), y0.float())[donor_tok]
            if check == "small":
                self.assertLess(rc.max().item(), 2e-2)
            else:
                self.assertGreater(rc.median().item(), 0.05)


# ----------------------------------------------------------------------------- distributed
def _gloo_worker(rank, world, init_file, out_dir):
    dist.init_process_group("gloo", init_method="file:///" + init_file, rank=rank, world_size=world)
    try:
        torch.manual_seed(0)   # identical parameters on every rank
        cfg = EMeansMoEConfig(num_experts=6, top_k=2, split_interval=1, cooldown_steps=0, alpha_0=0.5,
                              alpha_min=0.5, unmerge_min_steps=1, merge_max_steps=3, unmerge_ramp_steps=3)
        r = EMeansMoERouter(16, cfg)
        ex = GeGLUMoEExperts(6, 16, 8)
        with torch.no_grad():
            r.gate.weight.normal_(0.0, 0.3)
            r.gate.bias[5] = -30.0
            r.gate.bias[2] += 1.0
        x = make_tokens(512, 16, seed=100 + rank, offset=float(rank))   # different data per rank
        splitter = DynamicExpertSplitter(seed=0)
        local_load_step1 = None
        ema_after_step1 = None
        splits, unmerges = [], []
        for step in range(1, 13):
            o = r(x)
            if step == 1:
                local_load_step1 = o.batch_load.clone()
            r.update_ema(o.batch_load, torch.full((6,), float(rank + 1)), step, 100, batch_sum=o.batch_sum,
                         batch_sqnorm=o.batch_sqnorm, pair_disagree=o.pair_disagree)
            if step == 1:
                ema_after_step1 = r.ema_load.clone()
            unmerges += [(e.step, e.expert_a, e.expert_b, e.forced) for e in splitter.maybe_unmerge_layer(r, step)]
            splits += [(e.step, e.dead_idx, e.donor_idx) for e in splitter.maybe_split_layer(r, ex, None, step)]
        torch.save({
            "state": {k: v.clone() for k, v in r.state_dict().items()},
            "experts": ex.gate_proj.detach().clone(),
            "local_load_step1": local_load_step1,
            "ema_after_step1": ema_after_step1,
            "splits": splits,
            "unmerges": unmerges,
        }, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@requires_torch
@unittest.skipUnless(torch is not None and dist.is_available(), "torch.distributed not available")
class TestDistributedGloo(unittest.TestCase):
    def test_two_rank_statistics_splits_and_unmerges_stay_in_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "init").replace("\\", "/")
            mp.spawn(_gloo_worker, args=(2, init_file, tmp), nprocs=2, join=True)
            res = [torch.load(os.path.join(tmp, f"rank{i}.pt"), weights_only=False) for i in range(2)]
        s0, s1 = res[0]["state"], res[1]["state"]
        for k in s0:
            self.assertTrue(torch.equal(s0[k], s1[k]), f"buffer/param {k} differs across ranks")
        self.assertTrue(torch.equal(res[0]["experts"], res[1]["experts"]))
        self.assertEqual(res[0]["splits"], res[1]["splits"])
        self.assertEqual(res[0]["unmerges"], res[1]["unmerges"])
        self.assertTrue(res[0]["splits"], "expected at least one split")
        self.assertTrue(res[0]["unmerges"], "expected at least one (forced) un-merge")
        # the EMA consumed the cross-rank MEAN of the local loads (SUM / world on gloo)
        mean_load = 0.5 * (res[0]["local_load_step1"] + res[1]["local_load_step1"])
        expected = 0.5 * torch.full((6,), 2 / 6) + 0.5 * mean_load
        self.assertTrue(torch.allclose(res[0]["ema_after_step1"], expected, atol=1e-6))
        self.assertFalse(torch.allclose(res[0]["local_load_step1"], res[1]["local_load_step1"]))


# ----------------------------------------------------------------------------- model level
def _tiny_spec():
    base = GEMMA_PRESETS["tiny-gemma-debug"]
    return type(base)(name="tiny-test", hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=12, vocab_size=128)


def _patterned_batch(gen, batch=8, seq=24, vocab=128):
    start = torch.randint(0, vocab, (batch, 1), generator=gen)
    steps = torch.arange(seq).unsqueeze(0)
    return (start * 5 + steps * 7) % vocab


@requires_torch
class TestModelLevel(unittest.TestCase):
    def _train(self, model, steps, lr=3e-3, seed=0, split=True):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed)
        losses, n_splits = [], 0
        model.train()
        for step in range(1, steps + 1):
            ids = _patterned_batch(gen)
            logits, aux = model(ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)) + aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if split:
                n_splits += len(model.step_ema_and_split(opt, step=step, total_steps=steps))
        return losses, n_splits

    def test_end_to_end_training_with_splits_and_unmerges(self):
        torch.manual_seed(0)
        cfg = EMeansMoEConfig(num_experts=6, top_k=2, split_interval=5, cooldown_steps=5, split_tau=0.2,
                              upcycle_strategy="progressive_split", progressive_active_experts=2,
                              dormant_init_bias=-8.0, upcycle_init_noise=0.0, unmerge_min_steps=3,
                              merge_max_steps=15, unmerge_ramp_steps=5)
        model = upcycle_gemma_model(TinyGemmaForCausalLM(_tiny_spec()), cfg)
        losses, n_splits = self._train(model, 60)
        self.assertTrue(all(math.isfinite(v) for v in losses))
        self.assertGreater(n_splits, 0)
        self.assertGreater(len(model.unmerge_log), 0)
        self.assertLess(sum(losses[-10:]) / 10, sum(losses[:10]) / 10)
        for p in model.parameters():
            self.assertTrue(bool(torch.isfinite(p).all()))

    def test_model_logits_continuous_through_split_only_with_sibling_routing(self):
        torch.manual_seed(1)
        cfg = EMeansMoEConfig(num_experts=6, top_k=2, split_interval=10_000, upcycle_strategy="progressive_split",
                              progressive_active_experts=3, dormant_init_bias=-8.0, upcycle_init_noise=0.02)
        model = upcycle_gemma_model(TinyGemmaForCausalLM(_tiny_spec()), cfg)
        self._train(model, 12, split=True)
        ids = _patterned_batch(torch.Generator().manual_seed(42))[:, :-1]

        deltas = {}
        for sibling in (True, False):
            m = copy.deepcopy(model)
            layer = m.iter_moe_layers()[0].mlp
            layer.router.cfg.sibling_routing = sibling
            m.eval()
            with torch.no_grad(), frozen_moe_stats(m):
                before, _ = m(ids)
                load = layer.router.ema_load
                donor, dead = int(load.argmax()), int(load.argmin())
                DynamicExpertSplitter()._execute_single_split(layer.router, layer.experts, None, dead, donor, 3)
                after, _ = m(ids)
            deltas[sibling] = (after - before).abs().max().item() / before.abs().max().item()
        self.assertLess(deltas[True], 1e-4, deltas)
        self.assertGreater(deltas[False], 1e-2, deltas)


if __name__ == "__main__":
    unittest.main()
