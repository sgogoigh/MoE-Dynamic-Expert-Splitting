"""
PoC Result Compiler (Any Machine, Python Stdlib Only): Aggregate PoC 1-4 Telemetry
===================================================================================

RUN (from `MoE-Dynamic-Expert-Splitting/`, after at least one PoC wrote its JSON):
    python poc/compile_poc_report.py                       # reads and writes poc/results
    python poc/compile_poc_report.py --results-dir DIR     # or MOE_POC_RESULTS_DIR=DIR

Every PoC separates SANITY checks (gating: properties the code must have) from INFORMATIONAL
comparisons (measurements on toy or small runs that do not gate anything). The report keeps
that separation so a toy comparison is never read as a passed test.

ARTIFACTS: <results-dir>/COMPILED-POC-REPORT.md, log under poc/logs/compile_report/.
"""

import argparse
from datetime import datetime
import json
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.poc_logging import results_root, tee_poc_logs  # noqa: E402


def _num(v, fmt=".4f"):
    return "n/a" if v is None else format(v, fmt)


def _sanity_lines(obj) -> list:
    d = obj.get("diagnostics") or {}
    if "sanity" in d:
        checks, passed = d["sanity"], d.get("sanity_passed")
    else:
        checks = {k: v for k, v in d.items() if k != "all_passed"}
        passed = d.get("all_passed")
    if not checks:
        return []
    out = [f"**Sanity checks: {'PASS' if passed else 'FAIL'}**", ""]
    out += [f"- `{k}`: {'pass' if v else 'FAIL'}" for k, v in checks.items()]
    return out + [""]


def _load_json(results_dir: pathlib.Path, name: str):
    p = results_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pending(title: str, cmd: str) -> list:
    return ["---", "", f"## {title} *(not run yet)*", f"Run `{cmd}`.", ""]


def section_poc1(p1) -> list:
    lines = ["## PoC 1: Invariance proofs, toy routing simulation, Gemma audit", "",
             "### Dense softmax mixture after the split (`b <- b - tau_r ln 2`)",
             "| tau_r | Max pair prob diff | Max other prob diff | Max MoE output diff |",
             "| :---: | :---: | :---: | :---: |"]
    for r in p1["invariance_proof"]:
        lines.append(f"| {r['tau_r']} | `{r['max_pair_prob_diff']:.3e}` | `{r['max_other_prob_diff']:.3e}` | "
                     f"`{r['max_moe_output_diff']:.3e}` |")
    if p1.get("topk_invariance"):
        lines += ["", "### Top-K routing after the split: plain vs sibling routing",
                  "| tau_r | K | Plain Top-K max output diff | Sibling routing max output diff |",
                  "| :---: | :---: | :---: | :---: |"]
        for r in p1["topk_invariance"]:
            lines.append(f"| {r['tau_r']} | {r['top_k']} | `{r['plain_topk_max_output_diff']:.3e}` | "
                         f"`{r['sibling_topk_max_output_diff']:.3e}` |")
    lines += ["", "### Toy routing simulation (informational; plain Top-K, hand-written router dynamics)",
              "| Seed | Method | Splits | Dead | Max/Mean | Norm entropy |",
              "| :---: | :--- | :---: | :---: | :---: | :---: |"]
    per_seed = p1.get("synthetic_simulation_per_seed") or {"-": p1["synthetic_simulation"]}
    for seed, rows in per_seed.items():
        for r in rows:
            lines.append(f"| {seed} | `{r['mode']}` | {r['splits']} | {r['dead_experts']} | {r['max_to_mean']}x | "
                         f"{r['norm_entropy']} |")
    lines += ["", "### Gemma upcycling footprint",
              "| Preset | E / K_r | Strategy | Dense (M) | MoE total (M) | MoE active (M) | Router stats sync (KB/step) | % of bf16 grad sync |",
              "| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for g in p1["gemma_architecture_profiles"]:
        sync = g.get("router_stats_sync_KB_per_step", g.get("ema_sync_KB_per_step"))
        frac = g.get("router_sync_frac_of_grad_sync")
        lines.append(f"| `{g['preset']}` | {g['E']}/{g['top_k']} | `{g['strategy']}` | {g['dense_params_M']} | "
                     f"{g['moe_total_params_M']} | {g['moe_active_params_M']} | {sync} | "
                     f"{'n/a' if frac is None else f'{100 * frac:.3f}%'} |")
    lines += [""] + _sanity_lines(p1)
    comp = p1.get("comparison_informational")
    if comp:
        lines += ["Informational (toy, per seed): " + "; ".join(f"`{k}` {v}" for k, v in comp.items()), ""]
    return lines


def section_poc2(p2) -> list:
    c = p2["config"]
    lines = ["---", "", "## PoC 2: split / un-merge continuity under Top-K routing (TinyGemma)", "",
             f"TinyGemma d={c['d_model']}, layers={c['layers']}, E={c['num_experts']}, K_r={c['top_k']}, "
             f"{c['steps']} steps. dL is measured on a fixed probe batch after the optimizer step with routing "
             "statistics frozen, so it isolates the structural change.", ""]
    if p2.get("part_a"):
        lines += ["| Condition | Splits | Un-merges (forced) | Max abs split dL | Max abs un-merge dL/step | "
                  "Mean abs optimizer-step dL | Final probe loss |",
                  "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for label, r in p2["part_a"].items():
            lines.append(f"| `{label}` | {r['num_splits']} | {r['num_unmerges']} ({r['num_forced_unmerges']}) | "
                         f"{_num(r['split_dL']['max_abs'], '.2e')} | {_num(r['unmerge_dL_per_step']['max_abs'], '.2e')} | "
                         f"{_num(r['optimizer_step_dL']['mean_abs'], '.2e')} | {r['final_probe_loss']:.4f} |")
        eps = p2["part_a"].get("emeans", {}).get("unmerge_episodes") or []
        if eps:
            lines += ["", "Un-merge episodes of `emeans` (a ramp spreads a forced un-merge; it does not make it free):", "",
                      "| Steps | Length | Total dL | Max step | Last step / largest earlier step |",
                      "| :--- | :---: | :---: | :---: | :---: |"]
            for ep in eps:
                lines.append(f"| {ep['start']}-{ep['end']} | {ep['n_steps']} | {ep['total_dL']:+.4f} | "
                             f"{ep['max_step_abs']:.4f} | {_num(ep['completion_jump_ratio'], '.2f')} |")
    if p2.get("part_b"):
        lines += ["", "Part B (informational, noisy_clone init):", "",
                  "| Condition | Final probe loss | Dead | Max/Mean | Entropy | Splits |",
                  "| :--- | :---: | :---: | :---: | :---: | :---: |"]
        for label, r in p2["part_b"].items():
            lines.append(f"| `{label}` | {r['final_probe_loss']:.4f} | {r['final_dead_experts']} | "
                         f"{r['final_max_to_mean']} | {r['final_norm_entropy']} | {r['num_splits']} |")
    return lines + [""] + _sanity_lines(p2)


def section_poc3(p3) -> list:
    lines = ["---", "", "## PoC 3: comparators and alpha_min sensitivity grid", ""]
    if p3.get("part_a"):
        lines += ["Comparators (informational; mean over seeds):", "",
                  "| Method | Held-out loss | sd | Dead | Max/Mean | Entropy | Routing domain info |",
                  "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for mode, r in p3["part_a"].items():
            lines.append(f"| `{mode}` | {_num(r['val_loss_mean'])} | {_num(r['val_loss_std'])} | "
                         f"{r['dead_experts_mean']:.2f} | {r['max_to_mean_mean']:.3f} | {r['norm_entropy_mean']:.3f} | "
                         f"{_num(r['routing_domain_info_mean'], '.3f')} |")
    cells = p3.get("part_b_cells") or []
    if cells:
        lines += ["", "alpha_min ladder per cell (held-out loss per rung):", "",
                  "| E | K | batch | tau | Loss per rung | Splits (all runs) | Best | Range | Verdict |",
                  "| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |"]
        for c in cells:
            rungs = ", ".join(f"{r['alpha_min']}: {_num(r['val_loss_mean'])}" for r in c["rungs"])
            splits = c.get("total_splits")
            lines.append(f"| {c['num_experts']} | {c['top_k']} | {c['batch_size']} | {c['tau_r']} | {rungs} | "
                         f"{'n/a' if splits is None else splits} | "
                         f"{c['best_alpha']} | {_num(c['ladder_range'])} | {'usable' if c['usable'] else c['reason']} |")
        n_zero = sum(1 for c in cells if c.get("total_splits") == 0)
        if n_zero:
            lines += ["", f"{n_zero}/{len(cells)} cells never split. alpha_min acts only through split decisions, "
                          "so a flat ladder there means the run never collapsed, not that alpha_min does not matter."]
    fit = (p3.get("diagnostics") or {}).get("fit_informational")
    if fit:
        if fit.get("status") == "fitted":
            lines += ["", f"Fit on {fit['n_usable']}/{fit['n_cells']} usable cells: a_bl={_num(fit['a_bl'], '.3f')}, "
                          f"a_e={_num(fit['a_e'], '.3f')}, a_k={_num(fit['a_k'], '.3f')}, a_tau={_num(fit['a_tau'], '.3f')} "
                          f"(R^2 {_num(fit['r2_log'], '.3f')}, dof {fit['dof']}, unidentified {fit['unidentified']}, "
                          f"LOSO mean rel err {_num(fit['loso']['loso_mean_rel_err'], '.3f')})."]
        else:
            lines += ["", f"No fit: {fit['n_usable']}/{fit['n_cells']} usable cells (flat {fit['n_flat']}, "
                          f"at edge {fit['n_at_edge']}). {fit.get('reason', '')}"]
    return lines + [""] + _sanity_lines(p3)


def section_poc4(p4) -> list:
    info = (p4.get("diagnostics") or {}).get("informational", {})
    lines = ["---", "", "## PoC 4: frozen-backbone partial upcycling", "",
             f"- Model `{p4['model_id']}`, upcycled `{', '.join(p4.get('upcycled_layers', []))}`, "
             f"trainable {p4['trainable_params_M']}M of {p4.get('total_params_M', 'n/a')}M parameters",
             f"- Held-out loss: dense {_num(info.get('heldout_loss_dense'))} -> upcycled step 0 "
             f"{_num(info.get('heldout_loss_step0'))} (gap {_num(info.get('upcycle_parity_gap'), '.2e')}) -> after training "
             f"{_num(info.get('heldout_loss_end'))}",
             f"- Dead experts per layer {_num(info.get('dead_experts_start'), '.2f')} -> {_num(info.get('dead_experts_end'), '.2f')}; "
             f"splits {info.get('num_splits')}, un-merges {info.get('num_unmerges')}; routing domain info "
             f"{_num(info.get('routing_domain_info_end'), '.3f')}", ""]
    return lines + _sanity_lines(p4)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compile PoC results into one markdown report")
    ap.add_argument("--results-dir", default=None, help="default: MOE_POC_RESULTS_DIR or poc/results")
    args = ap.parse_args(argv)
    with tee_poc_logs("compile_report"):
        results_dir = results_root(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        p = {i: _load_json(results_dir, f"poc_0{i}_results.json") for i in (1, 2, 3, 4)}
        done = [f"PoC {i}" for i in (1, 2, 3, 4) if p[i] is not None]
        lines = ["# Compiled Proof-of-Concept Report", "",
                 f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
                 f"**Completed**: {', '.join(done) if done else 'none yet'}", "",
                 "Sanity checks gate each PoC. Everything labelled informational is a measurement on toy or "
                 "small runs and is not evidence at scale.", "", "---", ""]
        lines += section_poc1(p[1]) if p[1] else _pending("PoC 1", "python poc/poc_01_math_and_profiler.py")
        lines += section_poc2(p[2]) if p[2] else _pending("PoC 2", "python poc/poc_02_tiny_gemma_continuity.py")
        lines += section_poc3(p[3]) if p[3] else _pending("PoC 3", "python poc/poc_03_proxy_collapse_and_scaling.py")
        lines += section_poc4(p[4]) if p[4] else _pending(
            "PoC 4", "python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 2")
        out_md = results_dir / "COMPILED-POC-REPORT.md"
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"Compiled PoC report -> {out_md}")
        return out_md


if __name__ == "__main__":
    main()
