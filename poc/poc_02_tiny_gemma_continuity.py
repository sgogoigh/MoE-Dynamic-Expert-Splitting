"""
PoC 2 (CPU or 1 small GPU): split / un-merge continuity under Top-K routing, TinyGemma
=====================================================================================

RUN:
    pip install numpy torch
    python poc/poc_02_tiny_gemma_continuity.py --dry-run
    python poc/poc_02_tiny_gemma_continuity.py                       # both parts, defaults
    python poc/poc_02_tiny_gemma_continuity.py --part A --steps 200 --d-model 64 --layers 2   # quick CPU run

WHAT IT MEASURES
  Part A - continuity (GATING). Starts from `progressive_split`, so dormant experts exist by
  construction and splits are guaranteed. Each split and each un-merge is measured as the change
  of the loss on a FIXED probe batch, taken AFTER the optimizer step with routing statistics
  frozen, so the number isolates the structural change (earlier versions compared against the
  loss before the optimizer step and mixed the update into the "spike"):
      emeans             sibling routing + centroid hyperplane + measured un-merge (the method)
      emeans_plain_topk  previous method: -tau ln2 bias shift, plain Top-K routing
      emeans_no_shift    plain Top-K and no bias shift (ablation)
  Part B - balancing comparison (INFORMATIONAL, never gating). Every condition starts from
  `noisy_clone`, i.e. all experts active, so dead experts have to arise from the training
  dynamics (earlier versions started every condition with dormant experts at bias -6, which
  the baselines cannot revive):
      unbalanced, aux_loss, deepseek_bias, emeans

  This is a toy (tiny model, synthetic tokens, a few hundred steps). Part B says nothing about
  balancing at scale; it exists to catch regressions and to keep the comparison honest.

ARTIFACTS: <out-dir>/poc_02_results.json (default poc/results), logs under poc/logs/poc_02/.
"""

import argparse
import json
import math
import pathlib
import sys
import time

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import EMeansMoEConfig, GemmaArchitectureSpec  # noqa: E402
from moe_splitting.metrics import compute_routing_health, expert_pairwise_cosine, moe_expert_weights  # noqa: E402
from moe_splitting.poc_logging import results_root, tee_poc_logs  # noqa: E402

PART_A = (  # label, router_mode, sibling_routing, no_shift
    ("emeans", "emeans", True, False),
    ("emeans_plain_topk", "emeans", False, False),
    ("emeans_no_shift", "emeans", False, True),
)
PART_B = (
    ("unbalanced", "unbalanced", True, False),
    ("aux_loss", "aux_loss", True, False),
    ("deepseek_bias", "deepseek_bias", True, False),
    ("emeans", "emeans", True, False),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PoC 2: split / un-merge continuity under Top-K routing")
    p.add_argument("--part", choices=("A", "B", "both"), default="both")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--ff-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--split-interval", type=int, default=25)
    p.add_argument("--unmerge-min-steps", type=int, default=20)
    p.add_argument("--merge-max-steps", type=int, default=100)
    p.add_argument("--unmerge-ramp-steps", type=int, default=50)
    p.add_argument("--init-noise", type=float, default=1e-3, help="upcycle_init_noise for Part B noisy_clone")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--continuity-tol", type=float, default=1e-3, help="max |split dL| allowed for emeans")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print plan & config without training.")
    return p.parse_args(argv)


def make_spec(args) -> GemmaArchitectureSpec:
    return GemmaArchitectureSpec(
        name="poc2-tiny-gemma",
        hidden_size=args.d_model,
        intermediate_size=args.ff_dim,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=max(args.d_model // 4, 4),
        vocab_size=args.vocab,
    )


def _fmt(v):
    return "n/a" if v is None else f"{v:.2e}"


def _summ(values):
    if not values:
        return {"n": 0, "max_abs": None, "mean_abs": None, "max_increase": None}
    return {
        "n": len(values),
        "max_abs": max(abs(v) for v in values),
        "mean_abs": sum(abs(v) for v in values) / len(values),
        "max_increase": max(values),
    }


def _unmerge_episodes(trace):
    """Group consecutive un-merge steps (a ramp, or a 1-step immediate un-merge) into episodes.

    `total_dL` is the whole structural cost of the episode; the ramp spreads it over time but does
    not reduce it. `completion_jump_ratio` compares the final step with the largest earlier step of
    the same ramp: a value near 1 means the ramp ends where its last step pointed (no hidden jump).
    """
    episodes, cur = [], []
    for ev in trace:
        if ev["kind"] not in ("unmerge_ramp", "unmerge_done"):
            continue
        if cur and ev["step"] != cur[-1]["step"] + 1:
            episodes.append(cur)
            cur = []
        cur.append(ev)
        if ev["kind"] == "unmerge_done" and not ev["lambdas"]:
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    out = []
    for ep in episodes:
        d = [e["dL"] for e in ep]
        prev = [abs(v) for v in d[:-1]]
        out.append({
            "start": ep[0]["step"],
            "end": ep[-1]["step"],
            "n_steps": len(ep),
            "completed": ep[-1]["kind"] == "unmerge_done",
            "total_dL": sum(d),
            "max_step_abs": max(abs(v) for v in d),
            "completion_jump_ratio": (abs(d[-1]) / max(prev)) if prev and max(prev) > 0 else None,
        })
    return out


def run_condition(label, router_mode, sibling, no_shift, upcycle, args, spec, device, measure_events) -> dict:
    import torch
    import torch.nn.functional as F
    from moe_splitting import TinyGemmaForCausalLM, frozen_moe_stats, iter_moe_blocks, upcycle_gemma_model
    from moe_splitting.splitting import DynamicExpertSplitter
    from moe_splitting.synthetic_data import make_multi_domain_batch

    class NoBiasShiftSplitter(DynamicExpertSplitter):
        """Ablation: undo the -tau ln2 shift after the split (keeps the hyperplane offsets)."""

        @torch.no_grad()
        def _execute_single_split(self, router, experts, optimizer, dead_idx, donor_idx, pair_seed):
            r_norm = super()._execute_single_split(router, experts, optimizer, dead_idx, donor_idx, pair_seed)
            shift = router.tau_r * math.log(2.0)
            router.gate.bias[donor_idx] += shift
            router.gate.bias[dead_idx] += shift
            return r_norm

    t0 = time.time()
    torch.manual_seed(args.seed)
    dense = TinyGemmaForCausalLM(spec).to(device)
    cfg = EMeansMoEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_mode=router_mode,
        alpha_0=0.12,
        alpha_min=0.025,
        split_interval=args.split_interval,
        split_tau=0.10,
        donor_min_mult=1.15,
        max_splits_per_step=2,
        cooldown_steps=args.split_interval * 2,
        sibling_routing=sibling,
        unmerge_min_steps=args.unmerge_min_steps,
        merge_max_steps=args.merge_max_steps,
        unmerge_ramp_steps=args.unmerge_ramp_steps,
        upcycle_strategy=upcycle,
        upcycle_init_noise=args.init_noise if upcycle == "noisy_clone" else 1e-3,
        progressive_active_experts=max(2, args.num_experts // 2),
        dormant_init_bias=-6.0,
    )
    model = upcycle_gemma_model(dense, cfg).to(device)
    if no_shift:
        model.set_splitter(NoBiasShiftSplitter())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    V = spec.vocab_size
    probe, _ = make_multi_domain_batch(987_654, args.batch_size, args.seq_len, V, device)

    def ce_of(logits, toks):
        return F.cross_entropy(logits[:, :-1, :].reshape(-1, V), toks[:, 1:].reshape(-1))

    def probe_loss() -> float:
        model.eval()
        with torch.no_grad(), frozen_moe_stats(model):
            logits, _ = model(probe)
            val = float(ce_of(logits, probe).item())
        model.train()
        return val

    def any_merged() -> bool:
        return any(bool(b.router.merged_mask().any()) for _, b in iter_moe_blocks(model))

    def any_ramping() -> bool:
        return any(bool((b.router.unmerge_lambda > 0).any()) for _, b in iter_moe_blocks(model))

    split_dl, unmerge_dl, mixed_dl, opt_dl, loss_trace, trace = [], [], [], [], [], []
    unmerge_divergence = []
    blocks_by_name = dict(iter_moe_blocks(model))
    n_splits = n_unmerges = n_forced = 0
    model.train()
    for step in range(1, args.steps + 1):
        batch, _ = make_multi_domain_batch(step, args.batch_size, args.seq_len, V, device)
        logits, aux = model(batch)
        ce = ce_of(logits, batch)
        opt.zero_grad()
        (ce + aux).backward()
        interval_step = step % args.split_interval == 0
        pre_opt = probe_loss() if measure_events and interval_step else None
        opt.step()
        loss_trace.append(float(ce.item()))

        pre = None
        ramping_before = any_ramping()
        if measure_events and (interval_step or any_merged()):
            pre = probe_loss()
            if pre_opt is not None:
                opt_dl.append(pre - pre_opt)          # scale reference: one ordinary optimizer step
        splits = model.step_ema_and_split(opt, step=step, total_steps=args.steps)
        unmerges = model.last_unmerge_events
        for u in unmerges:  # how far the siblings' FFN weights had diverged when they were separated
            blk = blocks_by_name[u.layer_name]
            div = expert_pairwise_cosine(moe_expert_weights(blk.experts), [(u.expert_a, u.expert_b)])
            unmerge_divergence.append({"step": step, "layer": u.layer_name, "forced": u.forced,
                                       "merge_age": u.merge_age, "pair_cos": div["pair_cos"][0],
                                       "pair_rel_diff": div["pair_rel_diff"][0]})
        n_splits += len(splits)
        n_unmerges += len(unmerges)
        n_forced += sum(1 for u in unmerges if u.forced)
        unmerge_activity = bool(unmerges) or ramping_before or any_ramping()
        if pre is not None and (splits or unmerge_activity):
            d = probe_loss() - pre
            if splits and not unmerge_activity:
                split_dl.append(d)
                kind = "split"
            elif unmerge_activity and not splits:
                unmerge_dl.append(d)
                kind = "unmerge_done" if unmerges else "unmerge_ramp"
            else:
                mixed_dl.append(d)
                kind = "mixed"
            lam = [round(float(v), 3) for _, b in iter_moe_blocks(model) for v in b.router.unmerge_lambda if v > 0]
            trace.append({"step": step, "kind": kind, "dL": d, "lambdas": lam[:8]})

    health = [compute_routing_health(b.router.ema_load, top_k=args.top_k, dead_tau=cfg.split_tau)
              for _, b in iter_moe_blocks(model)]
    n = len(health)
    finite = all(math.isfinite(v) for v in loss_trace) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return {
        "label": label,
        "router_mode": router_mode,
        "sibling_routing": sibling,
        "bias_shift": not no_shift,
        "upcycle": upcycle,
        "num_splits": n_splits,
        "num_unmerges": n_unmerges,
        "num_forced_unmerges": n_forced,
        "split_dL": _summ(split_dl),
        "unmerge_dL_per_step": _summ(unmerge_dl),
        "unmerge_episodes": _unmerge_episodes(trace),
        "unmerge_pair_divergence": unmerge_divergence,
        "mixed_step_dL": _summ(mixed_dl),
        "optimizer_step_dL": _summ(opt_dl),
        "final_train_loss": sum(loss_trace[-20:]) / max(min(len(loss_trace), 20), 1),
        "final_probe_loss": probe_loss(),
        "final_dead_experts": round(sum(h["dead_count"] for h in health) / n, 2),
        "final_max_to_mean": round(sum(h["max_to_mean"] for h in health) / n, 3),
        "final_norm_entropy": round(sum(h["norm_entropy"] for h in health) / n, 4),
        "all_finite": finite,
        "wall_s": round(time.time() - t0, 1),
        "event_trace": trace,
    }


def evaluate(part_a: dict, part_b: dict, tol: float) -> dict:
    """Sanity checks gate the PoC; comparisons are reported but never gate it."""
    sanity, comparison = {}, {}
    if part_a:
        e, plain = part_a["emeans"], part_a["emeans_plain_topk"]
        e_max, p_max = e["split_dL"]["max_abs"], plain["split_dL"]["max_abs"]
        sanity["emeans_triggered_splits"] = e["num_splits"] > 0
        sanity["emeans_split_continuous"] = e_max is not None and e_max <= tol
        sanity["control_detects_plain_topk_discontinuity"] = (
            p_max is not None and e_max is not None and p_max > max(10 * e_max, tol)
        )
        # Mechanical check of the ramp only: a completed multi-step ramp must not end in a jump
        # (the bug this guards against made the last step ~16x its neighbours). It is NOT a claim
        # that un-merging is cheap: a forced un-merge of a pair that never separated costs about
        # as much in total as the plain Top-K split; that total is reported below, not gated.
        ramps = [ep for ep in e["unmerge_episodes"] if ep["completed"] and ep["n_steps"] >= 2]
        sanity["emeans_unmerge_ramp_has_no_completion_jump"] = all(
            ep["completion_jump_ratio"] is None or ep["completion_jump_ratio"] <= 2.0 for ep in ramps
        )
        sanity["part_a_finite"] = all(r["all_finite"] for r in part_a.values())
        totals = [ep["total_dL"] for ep in e["unmerge_episodes"]]
        comparison["unmerge_cost"] = {
            "n_episodes": len(totals),
            "n_ramps_checked": len(ramps),
            "forced_unmerges": e["num_forced_unmerges"],
            "natural_unmerges": e["num_unmerges"] - e["num_forced_unmerges"],
            "max_episode_total_dL": max(totals, default=None),
            "max_single_step_dL": e["unmerge_dL_per_step"]["max_abs"],
            "plain_topk_split_dL": p_max,
            "mean_optimizer_step_abs_dL": e["optimizer_step_dL"]["mean_abs"],
            "max_pair_rel_weight_diff_at_unmerge": max(
                (d["pair_rel_diff"] for d in e["unmerge_pair_divergence"]), default=None),
            "note": "a ramp spreads the cost of a forced un-merge over unmerge_ramp_steps; it does not remove it",
        }
    if part_b:
        sanity["part_b_finite"] = all(r["all_finite"] for r in part_b.values())
        e = part_b["emeans"]
        for base in ("unbalanced", "aux_loss", "deepseek_bias"):
            b = part_b[base]
            comparison[f"emeans_vs_{base}"] = {
                "dead_le": e["final_dead_experts"] <= b["final_dead_experts"],
                "entropy_ge": e["final_norm_entropy"] >= b["final_norm_entropy"],
                "max_to_mean_le": e["final_max_to_mean"] <= b["final_max_to_mean"],
                "probe_loss_le": e["final_probe_loss"] <= b["final_probe_loss"],
            }
    return {"sanity": sanity, "sanity_passed": bool(sanity) and all(sanity.values()),
            "comparison_informational": comparison}


def main(argv=None):
    args = parse_args(argv)
    tag = "dry_run" if args.dry_run else f"{args.part}_s{args.steps}_d{args.d_model}"
    with tee_poc_logs("poc_02", run_tag=tag):
        spec = make_spec(args)
        if args.dry_run:
            print("=== PoC 2 DRY-RUN PLAN ===")
            print(f"TinyGemma d={spec.hidden_size} ff={spec.intermediate_size} L={spec.num_hidden_layers} "
                  f"E={args.num_experts} K={args.top_k} steps={args.steps}")
            print("Part A (gating, progressive_split):", [c[0] for c in PART_A])
            print("Part B (informational, noisy_clone):", [c[0] for c in PART_B])
            return None
        try:
            import torch
        except ImportError:
            print("PyTorch is not installed. Re-run with --dry-run or install torch.")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PoC 2 on device={device}, {args.steps} steps per condition")
        part_a, part_b = {}, {}
        if args.part in ("A", "both"):
            print("--- Part A: continuity (progressive_split init; dL on a fixed probe batch, stats frozen)")
            for label, mode, sib, no_shift in PART_A:
                r = run_condition(label, mode, sib, no_shift, "progressive_split", args, spec, device, True)
                part_a[label] = r
                print(f"  {label:<18} splits={r['num_splits']:3d} unmerges={r['num_unmerges']:3d} "
                      f"(forced {r['num_forced_unmerges']}) | max|split dL|={_fmt(r['split_dL']['max_abs'])} "
                      f"| max|un-merge dL/step|={_fmt(r['unmerge_dL_per_step']['max_abs'])} "
                      f"(n={r['unmerge_dL_per_step']['n']}) | mean optimizer-step |dL|="
                      f"{_fmt(r['optimizer_step_dL']['mean_abs'])} | probe={r['final_probe_loss']:.4f} | {r['wall_s']}s")
                for ep in r["unmerge_episodes"]:
                    print(f"      un-merge steps {ep['start']}-{ep['end']} ({ep['n_steps']} steps): total dL="
                          f"{ep['total_dL']:+.4f} max step={ep['max_step_abs']:.4f} "
                          f"completion/prev-max={_fmt(ep['completion_jump_ratio'])}")
        if args.part in ("B", "both"):
            print("--- Part B: balancing comparison (noisy_clone init; informational only)")
            for label, mode, sib, no_shift in PART_B:
                r = run_condition(label, mode, sib, no_shift, "noisy_clone", args, spec, device, False)
                part_b[label] = r
                print(f"  {label:<14} probe={r['final_probe_loss']:.4f} dead={r['final_dead_experts']} "
                      f"max/mean={r['final_max_to_mean']} H={r['final_norm_entropy']} splits={r['num_splits']} "
                      f"| {r['wall_s']}s")

        diagnostics = evaluate(part_a, part_b, args.continuity_tol)
        out_dir = results_root(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_02_results.json"
        payload = {"poc_id": "poc_02", "config": vars(args), "part_a": part_a, "part_b": part_b,
                   "diagnostics": diagnostics}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"PoC 2 sanity passed: {diagnostics['sanity_passed']} -> {diagnostics['sanity']}")
        print(f"Informational comparison: {diagnostics['comparison_informational']}")
        print(f"Saved -> {out_path}")
        return payload


if __name__ == "__main__":
    main()
