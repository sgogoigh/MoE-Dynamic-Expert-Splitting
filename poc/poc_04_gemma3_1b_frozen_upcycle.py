"""
PoC 4: frozen-backbone partial upcycling of Gemma (Gemma-3-1B on one GPU, or a synthetic TinyGemma)
===================================================================================================

RUN:
    pip install numpy torch transformers safetensors
    # offline, zero downloads (CPU is fine):
    python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 2 --steps 200 --dry-run
    python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 2 --steps 200
    # real checkpoint, after accepting the Gemma license and downloading it yourself:
    python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path /path/to/gemma-3-1b-pt --upcycle-last-n 8 \
        --steps 500 --data-dir /path/to/gemma-tokenized-shards
  Hardware: A100 40/80GB for --upcycle-last-n 8..13 on Gemma-3-1B, RTX 4090 for 4..6.
  Automatic HuggingFace downloads are disabled (local_files_only=True).

WHAT IT DOES
  Freezes the whole model, then replaces `layer.mlp` (and nothing else, so HF attention, masks,
  caches and sliding windows are untouched) in the top N layers with an e-means MoE block
  (`progressive_split`: K0 active clones + dormant experts that DynamicExpertSplitter revives).
  Only the MoE blocks train. Splits and un-merges go through `step_moe_blocks`.

  Held-out loss is measured three times on the same fixed batches: dense model, right after
  upcycling (step 0), and after training; per domain, plus the domain x expert routing table.

  Sanity checks (gating): only MoE parameters are trainable; upcycling changes the held-out loss
  by at most --parity-tol nats; training stays finite; at least one split happened.
  Informational: held-out loss before/after, dormant experts before/after, routing table.

  Data: synthetic 3-domain token streams by default (see moe_splitting.token_data); with a real
  checkpoint pass --data-dir shards tokenized with the model's own tokenizer, otherwise the
  model is fine-tuned on token patterns that are not text.

ARTIFACTS: <out-dir>/poc_04_results.json (default poc/results), logs under poc/logs/poc_04/.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import GEMMA_PRESETS, EMeansMoEConfig  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.poc_logging import results_root, tee_poc_logs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PoC 4: frozen-backbone partial Gemma upcycling")
    p.add_argument("--model-path", default=None, help="local directory of a Gemma 2/3 checkpoint")
    p.add_argument("--synthetic-gemma", action="store_true", help="offline TinyGemma (tiny-gemma-debug preset)")
    p.add_argument("--synthetic-layers", type=int, default=None,
                   help="layers of the synthetic model (default: max(4, upcycle_last_n + 2))")
    p.add_argument("--upcycle-last-n", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--active-k0", type=int, default=4, help="active experts at step 0 (the rest are dormant)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--split-interval", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None, help="prose.npy / code.npy / math.npy shards (model's tokenizer)")
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--parity-tol", type=float, default=0.02, help="nats; max held-out change from upcycling")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="build and count parameters, no training")
    args = p.parse_args(argv)
    if not args.synthetic_gemma and args.model_path is None:
        p.error("pass --model-path /local/gemma-3-1b-pt or --synthetic-gemma")
    if args.active_k0 <= args.top_k - 1 or args.active_k0 > args.num_experts:
        p.error("--active-k0 must be in [top_k, num_experts]")
    return args


def model_layers(model):
    return getattr(getattr(model, "model", model), "layers")


def freeze_backbone_and_upcycle_top_layers(model, moe_cfg: EMeansMoEConfig, last_n: int):
    """In place: freeze every parameter, then replace `layer.mlp` of the top `last_n` layers with
    a trainable MoE block. Works for TinyGemma and HF Gemma 2/3 (the decoder layer is kept)."""
    from moe_splitting.gemma_upcycle import upcycle_gemma_mlp_to_moe

    for p in model.parameters():
        p.requires_grad_(False)
    layers = model_layers(model)
    start = max(0, len(layers) - last_n)
    blocks = []
    for idx in range(start, len(layers)):
        moe = upcycle_gemma_mlp_to_moe(layers[idx].mlp, moe_cfg, layer_idx=idx)
        for p in moe.parameters():
            p.requires_grad_(True)
        layers[idx].mlp = moe
        blocks.append((f"layers.{idx}.mlp", moe))
    return model, blocks


def main(argv=None):
    args = parse_args(argv)
    tag = "dry_run" if args.dry_run else ("synthetic" if args.synthetic_gemma else "hf_local")
    with tee_poc_logs("poc_04", run_tag=tag):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            spec = GEMMA_PRESETS["gemma-3-1b"]
            per_layer = args.num_experts * 3 * spec.hidden_size * spec.intermediate_size
            trainable = args.upcycle_last_n * per_layer
            print("PyTorch is not installed; estimate only (noisy_clone/progressive experts hold the full FFN).")
            print(f"Gemma-3-1B top-{args.upcycle_last_n} upcycle: ~{trainable / 1e6:.0f}M trainable params, "
                  f"~{trainable * 16 / 1024 ** 3:.1f} GB for weights + fp32 AdamW state")
            return None
        from moe_splitting import DynamicExpertSplitter, TinyGemmaForCausalLM, iter_moe_blocks, step_moe_blocks
        from moe_splitting.evaluation import heldout_eval
        from moe_splitting.token_data import make_source

        device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"
        torch.manual_seed(args.seed)
        is_hf = not args.synthetic_gemma
        if args.synthetic_gemma:
            layers = args.synthetic_layers or max(4, args.upcycle_last_n + 2)
            spec = dataclasses.replace(GEMMA_PRESETS["tiny-gemma-debug"], num_hidden_layers=layers)
            model = TinyGemmaForCausalLM(spec).to(device)
            model_id = f"synthetic-{spec.name}-L{layers}"
            vocab = spec.vocab_size
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, local_files_only=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device)
            model_id = args.model_path
            vocab = int(model.config.vocab_size)
        if args.upcycle_last_n > len(model_layers(model)):
            raise SystemExit(f"--upcycle-last-n {args.upcycle_last_n} > {len(model_layers(model))} layers")

        def forward(tokens):
            return model(tokens, use_cache=False) if is_hf else model(tokens)

        def logits_of(out):
            return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

        source = make_source(args.data_dir)
        val_set = [source.batch(v, args.batch_size, args.seq_len, vocab, "val", device) for v in range(args.val_batches)]

        def evaluate():
            return heldout_eval(model, val_set, source.domain_names, forward=forward)

        moe_cfg = EMeansMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            router_mode="emeans",
            alpha_0=0.06,
            alpha_min=0.008,
            split_interval=args.split_interval,
            cooldown_steps=2 * args.split_interval,
            upcycle_strategy="progressive_split",
            progressive_active_experts=args.active_k0,
            dormant_init_bias=-6.0,
        )
        dense_eval = None if args.dry_run else evaluate()
        model, blocks = freeze_backbone_and_upcycle_top_layers(model, moe_cfg, args.upcycle_last_n)
        moe_param_ids = {id(p) for _, b in blocks for p in b.parameters()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in trainable)
        print(f"Model: {model_id} | total {total_params / 1e6:.2f}M params | trainable (MoE) {n_trainable / 1e6:.2f}M "
              f"| upcycled layers {[n for n, _ in blocks]}")
        if args.dry_run:
            print("PoC 4 dry-run finished (no training).")
            return None

        t0 = time.time()
        step0_eval = evaluate()
        dead_start = sum(compute_routing_health(b.router.ema_load, args.top_k, moe_cfg.split_tau)["dead_count"]
                         for _, b in blocks) / len(blocks)
        opt = torch.optim.AdamW(trainable, lr=args.lr)
        splitter = DynamicExpertSplitter()
        n_splits = n_unmerges = 0
        finite = True
        losses = []
        model.train()
        for step in range(1, args.steps + 1):
            tokens, _ = source.batch(args.seed * 1_000_000 + step, args.batch_size, args.seq_len, vocab, "train", device)
            logits = logits_of(forward(tokens)).float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1))
            aux = sum((b.last_aux_loss for _, b in blocks if b.last_aux_loss is not None), loss.new_zeros(()))
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            opt.zero_grad()
            (loss + aux).backward()
            opt.step()
            splits, unmerges = step_moe_blocks(model, splitter, opt, step=step, total_steps=args.steps)
            n_splits += len(splits)
            n_unmerges += len(unmerges)
            losses.append(float(loss.item()))
            if step % max(1, args.steps // 10) == 0:
                print(f"  step {step:5d} train loss {losses[-1]:.4f} | splits {n_splits} un-merges {n_unmerges}")
        finite = finite and all(bool(torch.isfinite(p).all()) for p in trainable)
        end_eval = evaluate() if finite else None
        dead_end = sum(compute_routing_health(b.router.ema_load, args.top_k, moe_cfg.split_tau)["dead_count"]
                       for _, b in iter_moe_blocks(model)) / len(blocks)

        parity_gap = abs(step0_eval["val_loss"] - dense_eval["val_loss"])
        sanity = {
            "only_moe_params_trainable": all(id(p) in moe_param_ids for p in trainable) and n_trainable < total_params,
            "upcycle_preserves_heldout_loss": parity_gap <= args.parity_tol,
            "training_finite": finite,
            "triggered_splits": n_splits > 0,
        }
        informational = {
            "heldout_loss_dense": dense_eval["val_loss"],
            "heldout_loss_step0": step0_eval["val_loss"],
            "heldout_loss_end": end_eval["val_loss"] if end_eval else None,
            "heldout_improved": bool(end_eval) and end_eval["val_loss"] < step0_eval["val_loss"],
            "upcycle_parity_gap": parity_gap,
            "dead_experts_start": dead_start,
            "dead_experts_end": dead_end,
            "num_splits": n_splits,
            "num_unmerges": n_unmerges,
            "routing_domain_info_end": end_eval["routing_domain_info"] if end_eval else None,
        }
        payload = {
            "poc_id": "poc_04",
            "model_id": model_id,
            "config": vars(args),
            "upcycled_layers": [n for n, _ in blocks],
            "total_params_M": round(total_params / 1e6, 3),
            "trainable_params_M": round(n_trainable / 1e6, 3),
            "per_domain_val_loss": {"dense": dense_eval["per_domain_val_loss"],
                                    "step0": step0_eval["per_domain_val_loss"],
                                    "end": end_eval["per_domain_val_loss"] if end_eval else None},
            "domain_expert_share_end": end_eval["domain_expert_share"] if end_eval else None,
            "domain_names": list(source.domain_names),
            "wall_s": round(time.time() - t0, 1),
            "diagnostics": {"sanity": sanity, "sanity_passed": all(sanity.values()),
                            "informational": informational},
        }
        payload = json.loads(json.dumps(payload, default=float), parse_constant=lambda _c: None)
        print(f"held-out loss: dense {dense_eval['val_loss']:.4f} -> upcycled {step0_eval['val_loss']:.4f} "
              f"(gap {parity_gap:.2e}) -> end {informational['heldout_loss_end']}")
        print(f"dead experts per layer: {dead_start:.2f} -> {dead_end:.2f} | splits {n_splits} un-merges {n_unmerges}")
        if end_eval:
            name, table = next(iter(end_eval["domain_expert_share"].items()))
            print(f"domain x expert routing share, {name} (rows: {list(source.domain_names)}):")
            for dom, row in zip(source.domain_names, table):
                print(f"  {dom:<12} " + " ".join(f"{v:.2f}" for v in row))
        out_dir = results_root(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_04_results.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"PoC 4 sanity passed: {payload['diagnostics']['sanity_passed']} -> {sanity}")
        print(f"Saved -> {out_path}")
        return payload


if __name__ == "__main__":
    main()
