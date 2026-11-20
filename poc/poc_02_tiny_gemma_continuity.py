"""
PoC 2 (1 Free Colab T4 / 1x GPU or CPU Dry-Run): Split-Step Continuity & Loss-Spike Ablation
=============================================================================================

STANDALONE RUN INSTRUCTIONS (FRESH GPU MACHINE / COLAB / RUNPOD):
  - Hardware Needed : 1x GPU with >= 2 GB VRAM (runs on Colab T4, or CPU for tiny runs).
  - Dependencies    : Python >= 3.10, `numpy`, `torch` (`pip install numpy torch`).
  - Steps:
      1. Copy/clone `MoE-Dynamic-Expert-Splitting/` and `cd` into its root folder.
      2. Install minimal deps if missing:
             pip install -r requirements.txt  (or `pip install numpy torch`)
      3. Verify setup cleanly with a dry-run (zero training):
             python poc/poc_02_tiny_gemma_continuity.py --dry-run
      4. Execute the 5-condition PyTorch ablation benchmark:
             python poc/poc_02_tiny_gemma_continuity.py --steps 1000 --d-model 256 --layers 6
         (For a 3-minute smoke run on slow GPU/CPU, use `--steps 300 --d-model 128 --layers 4`.)
      5. Refresh the compiled markdown report:
             python poc/compile_poc_report.py

WHAT TO EXPECT:
  - Compares 5 modes: `unbalanced`, `aux_loss`, `deepseek_bias`, `emeans_no_shift`
    (ablation WITHOUT `-tau_r ln 2`), and `emeans` (full method WITH `-tau_r ln 2`).
  - Runtime         : ~5-15 minutes on Colab T4 at `--steps 1000` (depends on `--d-model`).
  - Console Output  : Per-condition summary showing `splits`, `max_split_deltaL` (near zero
                      for `emeans` vs larger jump for `emeans_no_shift`), `final_loss`,
                      and `dead` expert count dropping toward 0.
  - Saved Artifacts :
      * Telemetry JSON : `poc/results/poc_02_results.json`
      * Run Log File   : `poc/logs/poc_02/run_<YYYYMMDD_HHMMSS>[_dry_run].log` (+ `latest.log`)
"""

import argparse
import json
import math
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import EMeansMoEConfig, GemmaArchitectureSpec  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.poc_logging import tee_poc_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 2: TinyGemma Split Continuity & Loss-Spike Ablation")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--split-interval", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Print plan & config without training.")
    return p.parse_args()


def _make_structured_multi_domain_batch(step: int, batch_size: int, seq_len: int, vocab_size: int, device):
    """Create asymmetric 3-domain synthetic token streams (high repetition in Domain A)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(10000 + step)
    batch = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        u = float(torch.rand((), generator=g))
        if u < 0.65:
            # Domain A (frequent tokens 0..255 with modular pattern)
            start = int(torch.randint(0, 64, (), generator=g))
            seq = (start + torch.arange(seq_len) * 3) % 256
        elif u < 0.88:
            # Domain B (code-like band 256..640)
            start = int(torch.randint(256, 384, (), generator=g))
            seq = 256 + ((start + torch.arange(seq_len) * 5) % 384)
        else:
            # Domain C (math-like sparse band 640..vocab)
            seq = torch.randint(640, vocab_size, (seq_len,), generator=g)
        batch[b] = seq
    return batch.to(device)


def run_single_condition(label: str, args, spec: GemmaArchitectureSpec, device: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from moe_splitting import TinyGemmaForCausalLM, upcycle_gemma_model
    from moe_splitting.splitting import DynamicExpertSplitter

    torch.manual_seed(args.seed)
    dense = TinyGemmaForCausalLM(spec)
    dense.to(device)

    mode = "emeans" if label == "emeans_no_shift" else label
    moe_cfg = EMeansMoEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_mode=mode,
        tau_r=1.0,
        alpha_0=0.12,
        alpha_min=0.025,
        split_interval=args.split_interval,
        split_tau=0.10,
        donor_min_mult=1.15,
        max_splits_per_step=2,
        cooldown_steps=args.split_interval * 2,
        upcycle_strategy="progressive_split",
        progressive_active_experts=max(2, args.num_experts // 2),
        dormant_init_bias=-6.0,
    )

    model = upcycle_gemma_model(dense, moe_cfg)
    model.to(device)
    if label == "emeans_no_shift":
        # Ablation subclass that omits the -tau_r ln(2) bias shift during splitting
        class NoBiasShiftSplitter(DynamicExpertSplitter):
            @torch.no_grad()
            def _execute_single_split(self, router, experts, optimizer, dead_idx, donor_idx, pair_seed):
                saved_bias = router.gate.bias[donor_idx].item() if router.gate.bias is not None else None
                super()._execute_single_split(router, experts, optimizer, dead_idx, donor_idx, pair_seed)
                if router.gate.bias is not None and saved_bias is not None:
                    router.gate.bias[donor_idx] = saved_bias
                    router.gate.bias[dead_idx] = saved_bias

        model.set_splitter(NoBiasShiftSplitter())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    split_deltas = []
    loss_trace = []

    for step in range(1, args.steps + 1):
        batch = _make_structured_multi_domain_batch(step, args.batch_size, args.seq_len, spec.vocab_size, device)
        logits, aux = model(batch)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
        total_loss = ce + aux

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_trace.append(float(ce.item()))
        events = model.step_ema_and_split(optimizer, step=step, total_steps=args.steps)
        if events:
            with torch.no_grad():
                post_logits, _ = model(batch)
                post_ce = F.cross_entropy(post_logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
            split_deltas.append(abs(float(post_ce.item()) - float(ce.item())))

    # Aggregate final routing health across layers
    dead_counts = []
    max_means = []
    entropies = []
    for layer in model.iter_moe_layers():
        h = compute_routing_health(layer.mlp.router.ema_load, top_k=args.top_k, dead_tau=moe_cfg.split_tau)
        dead_counts.append(h["dead_count"])
        max_means.append(h["max_to_mean"])
        entropies.append(h["norm_entropy"])

    return {
        "label": label,
        "num_splits": len(split_deltas),
        "max_abs_split_delta_loss": max(split_deltas) if split_deltas else None,
        "mean_abs_split_delta_loss": (sum(split_deltas) / len(split_deltas)) if split_deltas else None,
        "final_loss": sum(loss_trace[-20:]) / min(len(loss_trace), 20),
        "final_dead_experts": round(sum(dead_counts) / len(dead_counts), 2),
        "final_max_to_mean": round(sum(max_means) / len(max_means), 3),
        "final_norm_entropy": round(sum(entropies) / len(entropies), 4),
    }


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else f"s{args.steps}_d{args.d_model}"
    with tee_poc_logs("poc_02", run_tag=tag):
        spec = GemmaArchitectureSpec(
            name="poc2-tiny-gemma",
            hidden_size=args.d_model,
            intermediate_size=args.ff_dim,
            num_hidden_layers=args.layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=args.d_model // 4,
            vocab_size=1024,
        )

        if args.dry_run:
            print("=== PoC 2 DRY-RUN PLAN ===")
            print(f"TinyGemma: d={spec.hidden_size}, ff={spec.intermediate_size}, L={spec.num_hidden_layers}, E={args.num_experts}, K_r={args.top_k}")
            print(f"Conditions: unbalanced, aux_loss, deepseek_bias, emeans_no_shift, emeans ({args.steps} steps each)")
            return

        try:
            import torch
        except ImportError:
            print("PyTorch is not installed in this local environment. Re-run with `--dry-run` or install PyTorch on a GPU.")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running PoC 2 on device={device} ({args.steps} steps/condition)...")
        runs = []
        for label in ("unbalanced", "aux_loss", "deepseek_bias", "emeans_no_shift", "emeans"):
            r = run_single_condition(label, args, spec, device)
            runs.append(r)
            print(
                f"  {label:<16} | splits={r['num_splits']:2d} | max_split_deltaL={r['max_abs_split_delta_loss']} | "
                f"loss={r['final_loss']:.4f} | dead={r['final_dead_experts']} | max/mean={r['final_max_to_mean']}x"
            )

        by_lbl = {r["label"]: r for r in runs}
        e_spike = by_lbl["emeans"]["max_abs_split_delta_loss"]
        noshift_spike = by_lbl["emeans_no_shift"]["max_abs_split_delta_loss"]
        diagnostics = {
            "emeans_triggered_splits": by_lbl["emeans"]["num_splits"] > 0,
            "bias_shift_reduces_split_spike": (
                e_spike is not None and noshift_spike is not None and e_spike <= noshift_spike
            ),
            "emeans_fewer_dead_than_unbalanced": (
                by_lbl["emeans"]["final_dead_experts"] <= by_lbl["unbalanced"]["final_dead_experts"]
            ),
            "emeans_higher_entropy_than_unbalanced": (
                by_lbl["emeans"]["final_norm_entropy"] >= by_lbl["unbalanced"]["final_norm_entropy"]
            ),
        }
        diagnostics["all_passed"] = all(diagnostics.values())

        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_02_results.json"
        out_path.write_text(
            json.dumps(
                {
                    "poc_id": "poc_02",
                    "config": vars(args),
                    "runs": runs,
                    "diagnostics": diagnostics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"PoC 2 Baseline Diagnostics Passed: {diagnostics['all_passed']} -> {diagnostics}")
        print(f"Saved PoC 2 telemetry -> {out_path}")


if __name__ == "__main__":
    main()
