"""
PoC 3: comparators and an alpha_min sensitivity grid with a measured power-law fit (toy by default)
==================================================================================================

RUN:
    pip install numpy torch
    python poc/poc_03_proxy_collapse_and_scaling.py --dry-run
    python poc/poc_03_proxy_collapse_and_scaling.py                        # tiny CPU defaults
    python poc/poc_03_proxy_collapse_and_scaling.py --part B --seeds 3 --sweep-steps 300
    # ~180M-parameter proxy on one GPU, real shards:
    python poc/poc_03_proxy_collapse_and_scaling.py --d-model 512 --ff-dim 1536 --layers 12 --heads 8 \
        --kv-heads 4 --vocab 32000 --seq-len 512 --batch-size 16 --grid-batch 8,16,32 \
        --steps 1500 --sweep-steps 400 --seeds 3 --data-dir /path/to/shards

WHAT IT MEASURES
  Part A - comparators (INFORMATIONAL). unbalanced / aux_loss / deepseek_bias / emeans on the
  same model, data and seeds: held-out loss overall and per domain, dead experts, load balance,
  and `routing_domain_info` = I(domain; expert) / H(domain).
  Part B - alpha_min ladder per grid cell (E, K_r, batch size, tau_r). Each cell trains one run
  per ladder value and seed (the same seeds on every rung, so the ladder comparison is paired)
  and scores it by held-out loss only. A cell's best alpha_min enters the fit only if it is
  identified:
      flat     the loss range across the ladder is below max(--flat-tol, 2 x seed SEM): at this
               budget alpha_min does not matter in that cell, so there is no optimum to fit;
      at_edge  the best rung is the smallest or largest: the optimum may lie outside the ladder.
  The fit reports which exponents are identified (a feature held constant across the grid has
  none) and is skipped, with the reason, when too few usable cells remain. Earlier versions
  padded the fit with two invented observations and scored runs by val_loss + 0.05*(max/mean-1).

  Sanity checks (gating): every run stayed finite and the grid is complete. Everything else is
  reported, never gated.

MODEL / DATA
  Random-init TinyGemma upcycled with `sliced_granular`: each expert holds a 1/K_r slice of the
  FFN, so K_r active experts cost the same FLOPs as the dense FFN at every grid point (changing
  K_r changes expert width, not compute). sliced_granular is not function-preserving at step 0,
  which does not matter for a random-init proxy (PoC 2 and PoC 4 use other strategies).
  alpha_0 is fixed across the ladder (--alpha-0), so only alpha_min varies.
  Data: synthetic 3-domain streams, or --data-dir with one tokenized flat .npy shard per domain
  (prose.npy, code.npy, math.npy; the last 10% of each shard is held out). See
  moe_splitting.token_data for what synthetic held-out loss does and does not measure.

ARTIFACTS: <out-dir>/poc_03_results.json (default poc/results), logs under poc/logs/poc_03/.
"""

import argparse
import itertools
import json
import math
import pathlib
import statistics
import sys
import time

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import EMeansMoEConfig, GemmaArchitectureSpec  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.poc_logging import results_root, tee_poc_logs  # noqa: E402
from moe_splitting.scaling_law import ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw  # noqa: E402

COMPARATORS = ("unbalanced", "aux_loss", "deepseek_bias", "emeans")
MAX_STEPS = 1_000_000  # training seeds are seed * MAX_STEPS + step


def _csv(cast):
    def parse(text):
        vals = [cast(x) for x in text.split(",") if x.strip()]
        if not vals:
            raise argparse.ArgumentTypeError("empty list")
        return vals
    return parse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PoC 3: comparators and alpha_min sensitivity grid")
    p.add_argument("--part", choices=("A", "B", "both"), default="both")
    p.add_argument("--data-dir", default=None, help="directory with prose.npy / code.npy / math.npy token shards")
    m = p.add_argument_group("model")
    m.add_argument("--d-model", type=int, default=64)
    m.add_argument("--ff-dim", type=int, default=256)
    m.add_argument("--layers", type=int, default=2)
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--kv-heads", type=int, default=2)
    m.add_argument("--vocab", type=int, default=1024)
    m.add_argument("--seq-len", type=int, default=32)
    m.add_argument("--init-noise", type=float, default=1e-2,
                   help="expert noise; breaks the symmetry of experts that share a slice")
    a = p.add_argument_group("part A (comparators)")
    a.add_argument("--steps", type=int, default=150)
    a.add_argument("--batch-size", type=int, default=8)
    a.add_argument("--num-experts", type=int, default=8)
    a.add_argument("--top-k", type=int, default=2)
    b = p.add_argument_group("part B (alpha_min grid)")
    b.add_argument("--sweep-steps", type=int, default=100)
    b.add_argument("--grid-experts", type=_csv(int), default=[8, 16])
    b.add_argument("--grid-top-k", type=_csv(int), default=[1, 2])
    b.add_argument("--grid-batch", type=_csv(int), default=[8, 16])
    b.add_argument("--grid-tau", type=_csv(float), default=[1.0])
    b.add_argument("--alpha-ladder", type=_csv(float), default=[0.005, 0.02, 0.08])
    b.add_argument("--flat-tol", type=float, default=0.01, help="nats; smaller ladder ranges count as flat")
    s = p.add_argument_group("shared")
    s.add_argument("--alpha-0", type=float, default=0.12)
    s.add_argument("--split-interval", type=int, default=25)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--val-batches", type=int, default=8)
    s.add_argument("--val-batch-size", type=int, default=16)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    _validate(p, args)
    return args


def _validate(p, args):
    lad = args.alpha_ladder
    if len(lad) < 3:
        p.error("--alpha-ladder needs >= 3 values: an interior optimum needs a neighbour on each side")
    if sorted(set(lad)) != lad:
        p.error("--alpha-ladder must be strictly increasing")
    if not all(0.0 < x < 1.0 for x in lad):
        p.error("--alpha-ladder values must be in (0, 1)")
    if args.alpha_0 < lad[-1]:
        p.error("--alpha-0 must be >= the largest ladder value (alpha decays from alpha_0 to alpha_min)")
    for k in set(args.grid_top_k + [args.top_k]):
        if args.ff_dim % k:
            p.error(f"--ff-dim {args.ff_dim} must be divisible by every top-k (got {k})")
    for e, k in itertools.product(args.grid_experts + [args.num_experts], args.grid_top_k + [args.top_k]):
        if k >= e:
            p.error(f"top-k {k} must be smaller than the expert count {e}")
    if args.d_model % args.heads or args.heads % args.kv_heads:
        p.error("--d-model must be divisible by --heads, and --heads by --kv-heads")
    if not 1 <= args.seeds <= 999:
        p.error("--seeds must be in [1, 999]")
    if max(args.steps, args.sweep_steps) >= MAX_STEPS:
        p.error(f"steps must be < {MAX_STEPS}")


def make_spec(args) -> GemmaArchitectureSpec:
    return GemmaArchitectureSpec(
        name="poc3-proxy",
        hidden_size=args.d_model,
        intermediate_size=args.ff_dim,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.d_model // args.heads,
        vocab_size=args.vocab,
    )


def make_cfg(args, mode: str, num_experts: int, top_k: int, tau: float, alpha_min: float) -> EMeansMoEConfig:
    return EMeansMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        router_mode=mode,
        tau_r=tau,
        alpha_0=args.alpha_0,
        alpha_min=alpha_min,
        split_interval=args.split_interval,
        cooldown_steps=2 * args.split_interval,
        upcycle_strategy="sliced_granular",
        upcycle_init_noise=args.init_noise,
    )


def train_run(args, spec, cfg, steps: int, batch_size: int, seed: int, source, device: str) -> dict:
    import torch
    import torch.nn.functional as F
    from moe_splitting import TinyGemmaForCausalLM, iter_moe_blocks, upcycle_gemma_model
    from moe_splitting.evaluation import heldout_eval

    t0 = time.time()
    torch.manual_seed(seed)
    model = upcycle_gemma_model(TinyGemmaForCausalLM(spec).to(device), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    V = spec.vocab_size
    finite, losses, n_splits, n_unmerges = True, [], 0, 0
    model.train()
    for step in range(1, steps + 1):
        tokens, _ = source.batch(seed * MAX_STEPS + step, batch_size, args.seq_len, V, "train", device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, aux = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, V), tokens[:, 1:].reshape(-1))
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        opt.zero_grad()
        (loss + aux).backward()
        opt.step()
        n_splits += len(model.step_ema_and_split(opt, step=step, total_steps=steps))
        n_unmerges += len(model.last_unmerge_events)
        losses.append(float(loss.item()))
    finite = finite and all(bool(torch.isfinite(p).all()) for p in model.parameters())

    ev = None
    if finite:
        val_batches = (source.batch(v, args.val_batch_size, args.seq_len, V, "val", device)
                       for v in range(args.val_batches))
        ev = heldout_eval(model, val_batches, source.domain_names)
        finite = math.isfinite(ev["val_loss"])
    health = [compute_routing_health(b.router.ema_load, top_k=cfg.top_k, dead_tau=cfg.split_tau)
              for _, b in iter_moe_blocks(model)]
    n = len(health)
    tail = losses[-20:]
    return {
        "router_mode": cfg.router_mode,
        "num_experts": cfg.num_experts,
        "top_k": cfg.top_k,
        "batch_size": batch_size,
        "tokens_per_step": batch_size * args.seq_len,
        "tau_r": cfg.tau_r,
        "alpha_min": cfg.alpha_min,
        "seed": seed,
        "steps_run": len(losses),
        "all_finite": finite,
        "final_train_loss": sum(tail) / len(tail) if tail else None,
        "val_loss": ev["val_loss"] if finite else None,
        "per_domain_val_loss": ev["per_domain_val_loss"] if finite else None,
        "routing_domain_info": ev["routing_domain_info"] if finite else None,
        "dead_experts": sum(h["dead_count"] for h in health) / n,
        "max_to_mean": sum(h["max_to_mean"] for h in health) / n,
        "norm_entropy": sum(h["norm_entropy"] for h in health) / n,
        "num_splits": n_splits,
        "num_unmerges": n_unmerges,
        "wall_s": round(time.time() - t0, 1),
    }


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def summarize_cell(cell_key: dict, runs: list, ladder, seeds: int, flat_tol: float) -> dict:
    """Per-rung mean/std over seeds, the best rung, and whether that optimum is identified."""
    rungs = []
    for a in ladder:
        rs = [r for r in runs if r["alpha_min"] == a]
        mean, std = _mean_std([r["val_loss"] for r in rs])
        splits_mean, _ = _mean_std([r.get("num_splits") for r in rs])
        rungs.append({"alpha_min": a, "val_loss_mean": mean, "val_loss_std": std,
                      "splits_mean": splits_mean,
                      "n_finite": sum(1 for r in rs if r["all_finite"]), "n_runs": len(rs)})
    # alpha_min acts only through split decisions: with zero splits in the cell, a flat ladder says
    # nothing about alpha_min, only that the run never collapsed far enough to trigger a split.
    split_counts = [r.get("num_splits") for r in runs]
    total_splits = sum(split_counts) if runs and None not in split_counts else None
    out = dict(cell_key, rungs=rungs, total_splits=total_splits)
    if any(r["n_finite"] < seeds for r in rungs):
        out.update(best_alpha=None, ladder_range=None, noise_sem=None, flat=None, at_edge=None,
                   usable=False, reason="non-finite run")
        return out
    means = [r["val_loss_mean"] for r in rungs]
    best = min(range(len(means)), key=means.__getitem__)
    spread = max(means) - min(means)
    sem = None
    if seeds >= 2:
        pooled = math.sqrt(statistics.fmean([r["val_loss_std"] ** 2 for r in rungs]))
        sem = pooled / math.sqrt(seeds)
    flat = spread < max(flat_tol, 2.0 * sem if sem is not None else 0.0)
    at_edge = best in (0, len(ladder) - 1)
    out.update(best_alpha=ladder[best], ladder_range=spread, noise_sem=sem, flat=flat, at_edge=at_edge,
               usable=not flat and not at_edge,
               reason="flat" if flat else ("at_edge" if at_edge else None))
    return out


def fit_usable_cells(cells: list) -> dict:
    usable = [c for c in cells if c["usable"]]
    obs = [ScalingObservation(float(c["tokens_per_step"]), float(c["num_experts"]), float(c["top_k"]),
                              float(c["tau_r"]), float(c["best_alpha"])) for c in usable]
    result = {"n_cells": len(cells), "n_usable": len(usable),
              "n_flat": sum(1 for c in cells if c["flat"]),
              "n_at_edge": sum(1 for c in cells if c["at_edge"] and not c["flat"]),
              "n_non_finite": sum(1 for c in cells if c["reason"] == "non-finite run"),
              "n_no_splits": sum(1 for c in cells if c.get("total_splits") == 0)}
    try:
        fit = fit_alpha_min_powerlaw(obs)
    except ValueError as exc:
        result.update(status="not_fitted", reason=str(exc))
        return result
    result.update(status="fitted", c_alpha=fit.c_alpha, a_bl=fit.a_bl, a_e=fit.a_e, a_k=fit.a_k, a_tau=fit.a_tau,
                  r2_log=fit.r2_log, n_obs=fit.n_obs, dof=fit.dof, unidentified=list(fit.unidentified),
                  loso=evaluate_loso(obs))
    return result


def _clean(o):
    """JSON-safe: NaN / inf become null."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _f(v, fmt=".4f"):
    return "n/a" if v is None else format(v, fmt)


def grid_cells(args):
    return [dict(num_experts=e, top_k=k, batch_size=bs, tokens_per_step=bs * args.seq_len, tau_r=t)
            for e, k, bs, t in itertools.product(args.grid_experts, args.grid_top_k, args.grid_batch, args.grid_tau)]


def main(argv=None):
    args = parse_args(argv)
    tag = "dry_run" if args.dry_run else f"{args.part}_s{args.steps}_d{args.d_model}"
    with tee_poc_logs("poc_03", run_tag=tag):
        spec = make_spec(args)
        cells = grid_cells(args)
        n_a = len(COMPARATORS) * args.seeds if args.part in ("A", "both") else 0
        n_b = len(cells) * len(args.alpha_ladder) * args.seeds if args.part in ("B", "both") else 0
        print(f"PoC 3: TinyGemma d={spec.hidden_size} ff={spec.intermediate_size} L={spec.num_hidden_layers} "
              f"V={spec.vocab_size} seq={args.seq_len} | data={'synthetic' if args.data_dir is None else args.data_dir}")
        print(f"  Part A: {COMPARATORS} x {args.seeds} seed(s), {args.steps} steps -> {n_a} runs")
        print(f"  Part B: {len(cells)} cells (E={args.grid_experts}, K={args.grid_top_k}, batch={args.grid_batch}, "
              f"tau={args.grid_tau}) x ladder {args.alpha_ladder} x {args.seeds} seed(s), "
              f"{args.sweep_steps} steps -> {n_b} runs")
        if args.dry_run:
            return None
        try:
            import torch
        except ImportError:
            print("PyTorch is not installed. Re-run with --dry-run or install torch.")
            sys.exit(1)
        from moe_splitting.token_data import make_source

        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = make_source(args.data_dir)
        print(f"device={device}")
        part_a, cell_summaries, all_runs = {}, [], []
        mid_alpha = args.alpha_ladder[len(args.alpha_ladder) // 2]

        if args.part in ("A", "both"):
            print("--- Part A: comparators (informational)")
            for mode in COMPARATORS:
                cfg = make_cfg(args, mode, args.num_experts, args.top_k, 1.0, mid_alpha)
                runs = [train_run(args, spec, cfg, args.steps, args.batch_size, s, source, device)
                        for s in range(args.seeds)]
                all_runs.extend(runs)
                val_mean, val_std = _mean_std([r["val_loss"] for r in runs])
                part_a[mode] = {
                    "val_loss_mean": val_mean,
                    "val_loss_std": val_std,
                    "per_domain_val_loss_mean": {
                        d: _mean_std([r["per_domain_val_loss"][d] if r["per_domain_val_loss"] else None
                                      for r in runs])[0] for d in source.domain_names},
                    "routing_domain_info_mean": _mean_std([r["routing_domain_info"] for r in runs])[0],
                    "dead_experts_mean": statistics.fmean(r["dead_experts"] for r in runs),
                    "max_to_mean_mean": statistics.fmean(r["max_to_mean"] for r in runs),
                    "norm_entropy_mean": statistics.fmean(r["norm_entropy"] for r in runs),
                    "num_splits": [r["num_splits"] for r in runs],
                    "runs": runs,
                }
                pa = part_a[mode]
                print(f"  {mode:<14} val={_f(val_mean)} (sd {_f(val_std)}) dead={pa['dead_experts_mean']:.2f} "
                      f"max/mean={pa['max_to_mean_mean']:.3f} H={pa['norm_entropy_mean']:.3f} "
                      f"domain-info={_f(pa['routing_domain_info_mean'], '.3f')} splits={pa['num_splits']} "
                      f"| {sum(r['wall_s'] for r in runs):.0f}s")

        if args.part in ("B", "both"):
            print("--- Part B: alpha_min ladder per cell (scored by held-out loss only)")
            for key in cells:
                runs = []
                for a_min, s in itertools.product(args.alpha_ladder, range(args.seeds)):
                    cfg = make_cfg(args, "emeans", key["num_experts"], key["top_k"], key["tau_r"], a_min)
                    runs.append(train_run(args, spec, cfg, args.sweep_steps, key["batch_size"], s, source, device))
                all_runs.extend(runs)
                c = summarize_cell(key, runs, args.alpha_ladder, args.seeds, args.flat_tol)
                cell_summaries.append(c)
                ladder_txt = " ".join(_f(r["val_loss_mean"]) for r in c["rungs"])
                splits_txt = " ".join("-" if r["splits_mean"] is None else f"{r['splits_mean']:g}" for r in c["rungs"])
                print(f"  E={key['num_experts']:<3} K={key['top_k']} B={key['batch_size']:<3} tau={key['tau_r']}: "
                      f"[{ladder_txt}] splits/run=[{splits_txt}] best={c['best_alpha']} range={_f(c['ladder_range'])} "
                      f"-> {'usable' if c['usable'] else c['reason']}"
                      f"{' (no splits: alpha_min had nothing to act on)' if c['total_splits'] == 0 else ''}")

        fit = fit_usable_cells(cell_summaries) if cell_summaries else None
        expected = len(cells) * len(args.alpha_ladder) * args.seeds if args.part in ("B", "both") else 0
        sanity = {
            "all_runs_finite": all(r["all_finite"] for r in all_runs),
            "grid_complete": sum(len(c["rungs"]) * args.seeds for c in cell_summaries) == expected
            and all(r["n_runs"] == args.seeds for c in cell_summaries for r in c["rungs"]),
        }
        diagnostics = {"sanity": sanity, "sanity_passed": all(sanity.values()),
                       "fit_informational": fit, "comparators_informational": {
                           m: {k: v for k, v in r.items() if k != "runs"} for m, r in part_a.items()}}
        if fit is not None:
            if fit["status"] == "fitted":
                print(f"  fit on {fit['n_usable']}/{fit['n_cells']} cells: a_bl={_f(fit['a_bl'], '.3f')} "
                      f"a_e={_f(fit['a_e'], '.3f')} a_k={_f(fit['a_k'], '.3f')} a_tau={_f(fit['a_tau'], '.3f')} "
                      f"R2={_f(fit['r2_log'], '.3f')} dof={fit['dof']} unidentified={fit['unidentified']} "
                      f"LOSO rel err={_f(fit['loso']['loso_mean_rel_err'], '.3f')} "
                      f"({fit['loso']['n_folds_evaluated']} folds)")
            else:
                print(f"  no fit ({fit['n_usable']}/{fit['n_cells']} usable cells; flat={fit['n_flat']}, "
                      f"at_edge={fit['n_at_edge']}, cells with zero splits={fit['n_no_splits']}): {fit['reason']}")

        out_dir = results_root(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_03_results.json"
        payload = _clean({"poc_id": "poc_03", "config": vars(args), "part_a": part_a,
                          "part_b_cells": cell_summaries, "diagnostics": diagnostics})
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"PoC 3 sanity passed: {diagnostics['sanity_passed']} -> {sanity}")
        print(f"Saved -> {out_path}")
        return payload


if __name__ == "__main__":
    main()
