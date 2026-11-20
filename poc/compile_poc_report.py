"""
Compile PoC 1-4 JSON telemetry into a single grant/proposal-ready Markdown report:
  poc/results/COMPILED-POC-REPORT.md
Runs cleanly offline with zero dependencies beyond Python stdlib.
"""

import json
import pathlib
from datetime import datetime


POC_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = POC_DIR / "results"


def _load_json(name: str):
    p = RESULTS_DIR / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = _load_json("poc_01_results.json")
    p2 = _load_json("poc_02_results.json")
    p3 = _load_json("poc_03_results.json")
    p4 = _load_json("poc_04_results.json")

    lines = [
        "# Compiled Proof-of-Concept Evidence Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Completed Tiers**: "
        + ", ".join(
            [lbl for lbl, obj in [("PoC 1", p1), ("PoC 2", p2), ("PoC 3", p3), ("PoC 4", p4)] if obj is not None]
            or ["None yet"]
        ),
        "",
        "---",
        "",
    ]

    if p1:
        lines += [
            "## PoC 1: Exact Mathematical Invariance & Gemma Systems Audit",
            "",
            "### Softmax Mass Conservation Verification (`b <- b - tau_r ln 2`)",
            "| $\\tau_r$ | Max Pair Prob Diff | Max Other Prob Diff | Max MoE Output Diff |",
            "| :---: | :---: | :---: | :---: |",
        ]
        for r in p1["invariance_proof"]:
            lines.append(
                f"| {r['tau_r']} | `{r['max_pair_prob_diff']:.3e}` | `{r['max_other_prob_diff']:.3e}` | `{r['max_moe_output_diff']:.3e}` |"
            )

        lines += [
            "",
            "### Multi-Domain Synthetic 400-Step Comparison (`E=16, K_r=2`)",
            "| Method | Splits | Dead Experts | Max/Mean Load | Norm Entropy | Gini |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in p1["synthetic_simulation"]:
            lines.append(
                f"| `{r['mode']}` | {r['splits']} | {r['dead_experts']} | {r['max_to_mean']}x | {r['norm_entropy']} | {r['gini']} |"
            )

        lines += [
            "",
            "### Gemma Upcycling Resource Footprint",
            "| Preset | `E / K_r` | Strategy | Dense (M) | MoE Total (M) | MoE Active (M) | Unsharded Train (GB) | EMA Sync (KB/step) |",
            "| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for g in p1["gemma_architecture_profiles"]:
            lines.append(
                f"| `{g['preset']}` | `{g['E']}/{g['top_k']}` | `{g['strategy']}` | {g['dense_params_M']} | "
                f"{g['moe_total_params_M']} | {g['moe_active_params_M']} | {g['unsharded_train_GB']} | {g['ema_sync_KB_per_step']} |"
            )
        lines.append("")

    if p2:
        lines += [
            "---",
            "",
            "## PoC 2: PyTorch Split-Step Continuity & Loss-Spike Ablation",
            "",
            f"- **Model**: `TinyGemma` (`d={p2['config']['d_model']}, layers={p2['config']['layers']}, E={p2['config']['num_experts']}, K_r={p2['config']['top_k']}`)",
            f"- **Steps**: {p2['config']['steps']}",
            "",
            "| Method | Splits | Max $|\\Delta \\mathcal{L}|$ at Split | Final Loss | Final Dead Experts | Final Max/Mean |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for row in p2["runs"]:
            spike = f"`{row['max_abs_split_delta_loss']:.5f}`" if row["max_abs_split_delta_loss"] is not None else "N/A"
            lines.append(
                f"| `{row['label']}` | {row['num_splits']} | {spike} | {row['final_loss']:.4f} | "
                f"{row['final_dead_experts']} | {row['final_max_to_mean']:.2f}x |"
            )
        lines.append("")
    else:
        lines += [
            "---",
            "",
            "## PoC 2: PyTorch Split-Step Continuity *(Pending Execution)*",
            "Run `python poc/poc_02_tiny_gemma_continuity.py` on a GPU or Colab T4 to populate `poc/results/poc_02_results.json`.",
            "",
        ]

    if p3:
        lines += [
            "---",
            "",
            "## PoC 3: Real Multi-Domain Proxy & Mini Scaling Law",
            "",
            "| Method | Final Val Loss | Val PPL | Dead Experts | Max/Mean | Entropy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for row in p3.get("domain_comparison", []):
            lines.append(
                f"| `{row['mode']}` | {row['val_loss']:.4f} | {row['val_ppl']:.2f} | {row['dead_experts']} | "
                f"{row['max_to_mean']:.2f}x | {row['norm_entropy']:.3f} |"
            )
        if p3.get("scaling_law_fit"):
            sf = p3["scaling_law_fit"]
            lines += [
                "",
                f"**Fitted Mini Power Law**: $\\alpha_{{\\min}} = {sf['c_alpha']:.4f} (BL)^{{{sf['a_bl']:+.2f}}} E^{{{sf['a_e']:+.2f}}} K_r^{{{sf['a_k']:+.2f}}} \\tau_r^{{{sf['a_tau']:+.2f}}}$ "
                f"($R^2 = {sf['r2_log']:.4f}$, LOSO rel err = `{sf.get('loso_mean_rel_err_pct', 'N/A')}%`)",
                "",
            ]
    else:
        lines += [
            "---",
            "",
            "## PoC 3: Real Multi-Domain Proxy & Mini Scaling Law *(Pending Execution)*",
            "Run `python poc/poc_03_proxy_collapse_and_scaling.py --run-all` on 1× 4090/A100 to populate `poc/results/poc_03_results.json`.",
            "",
        ]

    if p4:
        lines += [
            "---",
            "",
            "## PoC 4: Frozen-Backbone Gemma-3-1B Partial Upcycling",
            "",
            f"- **Model**: `{p4['model_id']}` | **Upcycled Upper Layers**: {p4['upcycle_last_n']} | **Trainable Params**: `{p4['trainable_params_M']:.1f}M`",
            f"- **Step-0 Parity Max Diff**: `{p4.get('step0_parity_diff', 'N/A')}`",
            f"- **Total Dynamic Splits Triggered**: {p4['total_splits']}",
            f"- **Final Dormant Experts**: {p4['final_dead_experts']} (started at {p4['initial_dormant_experts']})",
            f"- **Validation Loss Before -> After**: `{p4['val_loss_start']:.4f}` -> `{p4['val_loss_end']:.4f}`",
            "",
        ]
    else:
        lines += [
            "---",
            "",
            "## PoC 4: Frozen-Backbone Gemma-3-1B Partial Upcycling *(Pending Execution)*",
            "Run `python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path <local_gemma_dir>` on 1× A100/4090 to populate `poc/results/poc_04_results.json`.",
            "",
        ]

    out_md = RESULTS_DIR / "COMPILED-POC-REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Compiled PoC report -> {out_md}")


if __name__ == "__main__":
    main()
