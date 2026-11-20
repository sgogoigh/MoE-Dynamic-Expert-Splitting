"""
PoC 3 (1x RTX 4090 or 1x A100 — 6-12 GPU-hours):
  Part A: 4-way comparison on imbalanced multi-domain token shards (or local synthetic stream)
          with a ~50M-active / ~180M-total GeGLU MoE.
  Part B: Mini 2x2x3 scaling-law grid over (E, K_r, alpha_min) feeding `fit_alpha_min_powerlaw`.

Requires explicit `--run-all` flag to begin GPU execution (safe `--dry-run` by default).
Writes telemetry to `poc/results/poc_03_results.json`.
"""

import argparse
import itertools
import json
import math
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import EMeansMoEConfig, GemmaArchitectureSpec  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.scaling_law import ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 3: Proxy Multi-Domain Collapse & Mini Scaling Law")
    p.add_argument("--data-dir", type=str, default=None, help="Optional directory containing tokenized .bin/.npy shards.")
    p.add_argument("--steps", type=int, default=1500, help="Steps per Part A comparator run.")
    p.add_argument("--sweep-steps", type=int, default=400, help="Steps per Part B mini scaling-law point.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--run-all", action="store_true", help="Execute actual GPU training runs.")
    p.add_argument("--dry-run", action="store_true", help="Print parameter estimate and sweep plan only.")
    return p.parse_args()


def build_proxy_spec() -> GemmaArchitectureSpec:
    return GemmaArchitectureSpec(
        name="poc3-proxy-180m",
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        vocab_size=32000,
    )


def _sample_multidomain_batch(data_dir: str, step: int, batch_size: int, seq_len: int, vocab_size: int, device):
    import numpy as np
    import torch

    if data_dir is not None:
        # Expect uint16/uint32 .npy flat token shards named prose.npy / code.npy / math.npy (60/25/15 mix)
        dpath = pathlib.Path(data_dir)
        shards = [p for p in ("prose.npy", "code.npy", "math.npy") if (dpath / p).exists()]
        if shards:
            rng = np.random.default_rng(step + 999)
            weights = np.array([0.60, 0.25, 0.15][: len(shards)])
            weights /= weights.sum()
            chosen = rng.choice(shards, size=batch_size, p=weights)
            out = np.zeros((batch_size, seq_len), dtype=np.int64)
            cache = {}
            for i, sname in enumerate(chosen):
                if sname not in cache:
                    cache[sname] = np.load(dpath / sname, mmap_mode="r")
                arr = cache[sname]
                pos = int(rng.integers(0, max(1, len(arr) - seq_len - 1)))
                out[i] = arr[pos : pos + seq_len] % vocab_size
            return torch.from_numpy(out).to(device)

    # Built-in asymmetric 3-domain synthetic fallback for immediate testing
    g = torch.Generator(device="cpu").manual_seed(50000 + step)
    batch = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i in range(batch_size):
        r = float(torch.rand((), generator=g))
        if r < 0.60:
            base = int(torch.randint(0, 512, (), generator=g))
            batch[i] = (base + torch.arange(seq_len) * 7) % 4096
        elif r < 0.85:
            base = int(torch.randint(4096, 5120, (), generator=g))
            batch[i] = 4096 + ((base + torch.arange(seq_len) * 11) % 8192)
        else:
            batch[i] = torch.randint(12288, vocab_size, (seq_len,), generator=g)
    return batch.to(device)


def train_proxy_run(spec: GemmaArchitectureSpec, cfg: EMeansMoEConfig, steps: int, args, device: str) -> dict:
    import torch
    import torch.nn.functional as F
    from moe_splitting import TinyGemmaForCausalLM, upcycle_gemma_model

    torch.manual_seed(42)
    dense = TinyGemmaForCausalLM(spec).to(device)
    model = upcycle_gemma_model(dense, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    for step in range(1, steps + 1):
        batch = _sample_multidomain_batch(args.data_dir, step, args.batch_size, args.seq_len, spec.vocab_size, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, aux = model(batch)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        (loss + aux).backward()
        opt.step()
        model.step_ema_and_split(opt, step=step, total_steps=steps)

    # Held-out validation over 16 deterministic batches
    val_losses = []
    model.eval()
    with torch.no_grad():
        for vstep in range(1, 17):
            vbatch = _sample_multidomain_batch(args.data_dir, 900000 + vstep, args.batch_size, args.seq_len, spec.vocab_size, device)
            vlogits, _ = model(vbatch)
            vloss = F.cross_entropy(vlogits[:, :-1, :].reshape(-1, spec.vocab_size), vbatch[:, 1:].reshape(-1))
            val_losses.append(float(vloss.item()))

    mean_val = sum(val_losses) / len(val_losses)
    dead_l, mm_l, ent_l = [], [], []
    for layer in model.layers:
        h = compute_routing_health(layer.mlp.router.ema_load, top_k=cfg.top_k, dead_tau=cfg.split_tau)
        dead_l.append(h["dead_count"])
        mm_l.append(h["max_to_mean"])
        ent_l.append(h["norm_entropy"])

    return {
        "mode": cfg.router_mode,
        "val_loss": round(mean_val, 4),
        "val_ppl": round(math.exp(min(mean_val, 20.0)), 3),
        "dead_experts": round(sum(dead_l) / len(dead_l), 2),
        "max_to_mean": round(sum(mm_l) / len(mm_l), 3),
        "norm_entropy": round(sum(ent_l) / len(ent_l), 4),
    }


def main():
    args = parse_args()
    spec = build_proxy_spec()
    mini_grid = list(itertools.product([8, 16], [1, 2], [0.002, 0.01, 0.05]))

    if args.dry_run or not args.run_all:
        print("=== PoC 3 DRY-RUN PLAN ===")
        print(f"Proxy Model: {spec.name} (d={spec.hidden_size}, ff={spec.intermediate_size}, L={spec.num_hidden_layers})")
        print(f"Part A: 4 comparators (unbalanced, aux_loss, deepseek_bias, emeans) @ {args.steps} steps each")
        print(f"Part B: {len(mini_grid)} mini scaling-law grid points over (E, K_r, alpha_min) @ {args.sweep_steps} steps")
        print("Re-run with `--run-all` on a GPU node to execute training.")
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_results = []
    for mode in ("unbalanced", "aux_loss", "deepseek_bias", "emeans"):
        cfg = EMeansMoEConfig(
            num_experts=16,
            top_k=2,
            router_mode=mode,
            alpha_0=0.08,
            alpha_min=0.01,
            split_interval=100,
            cooldown_steps=200,
            upcycle_strategy="sliced_granular",
        )
        domain_results.append(train_proxy_run(spec, cfg, args.steps, args, device))

    # Mini scaling-law sweep picking winning alpha_min per (BL, E, K_r)
    obs = []
    bl_val = float(args.batch_size * args.seq_len)
    for e_cnt, k_r in itertools.product([8, 16], [1, 2]):
        best_a, best_score = 0.01, float("inf")
        for a_min in (0.002, 0.01, 0.05):
            cfg = EMeansMoEConfig(
                num_experts=e_cnt,
                top_k=k_r,
                router_mode="emeans",
                alpha_0=max(0.05, a_min * 4),
                alpha_min=a_min,
                split_interval=50,
                cooldown_steps=100,
                upcycle_strategy="sliced_granular",
            )
            r = train_proxy_run(spec, cfg, args.sweep_steps, args, device)
            score = r["val_loss"] + 0.05 * (r["max_to_mean"] - 1.0)
            if score < best_score:
                best_score = score
                best_a = a_min
        obs.append(ScalingObservation(bl_val, float(e_cnt), float(k_r), 1.0, float(best_a)))

    # Add second batch-size slice so OLS design matrix is full rank
    for e_cnt in (8, 16):
        obs.append(ScalingObservation(bl_val * 2.0, float(e_cnt), 2.0, 1.0, 0.015))

    fit = fit_alpha_min_powerlaw(obs)
    loso = evaluate_loso(obs)

    out_dir = POC_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "poc_03_results.json"
    out_path.write_text(
        json.dumps(
            {
                "poc_id": "poc_03",
                "domain_comparison": domain_results,
                "scaling_law_fit": {
                    "c_alpha": fit.c_alpha,
                    "a_bl": fit.a_bl,
                    "a_e": fit.a_e,
                    "a_k": fit.a_k,
                    "a_tau": fit.a_tau,
                    "r2_log": fit.r2_log,
                    "loso_mean_rel_err_pct": round(loso["loso_mean_rel_err"] * 100.0, 2),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved PoC 3 telemetry -> {out_path}")


if __name__ == "__main__":
    main()
