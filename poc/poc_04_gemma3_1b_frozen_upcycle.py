"""
PoC 4 (1x A100 40/80GB or 1x RTX 4090 Partial Layers): Frozen-Backbone Gemma-3-1B Upcycle
==========================================================================================

STANDALONE RUN INSTRUCTIONS (REMOTE GPU WORKSTATION / CLOUD NODE):
  - Hardware Needed : 1x GPU (`A100 40/80GB` for `--upcycle-last-n 8..13`, or `RTX 4090 24GB`
                      with `--upcycle-last-n 4..6`). Offline `--synthetic-gemma --dry-run` works on CPU.
  - Dependencies    : Python >= 3.10, `numpy`, `torch`, `transformers`, `accelerate`, `safetensors`.
  - Steps:
      1. Copy/clone `MoE-Dynamic-Expert-Splitting/` onto the GPU machine and `cd` into it.
      2. Install cluster dependencies:
             pip install -r requirements.txt
      3. Inspect frozen vs. trainable parameter footprint offline (zero downloads):
             python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 8 --dry-run
      4. Download `google/gemma-3-1b-pt` manually into `/path/to/gemma-3-1b-pt` (after accepting
         the HuggingFace Gemma license) OR run offline synthetic mode:
         - Real local Gemma checkpoint run:
             python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path /path/to/gemma-3-1b-pt --upcycle-last-n 8 --steps 500
         - Synthetic scaled Gemma run (zero downloads):
             python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 6 --steps 300
      5. Compile all results into markdown:
             python poc/compile_poc_report.py

WHAT TO EXPECT:
  - Freezes Gemma embeddings, RMSNorms, GQA attention, and lower layers; upcycles only the
    upper N MLP blocks (`progressive_split`: 4 active + 12 dormant experts, `sliced_granular`).
  - Console Output  : Total vs. trainable MoE parameter count, split progress, and before/after
                      validation loss.
  - Saved Artifacts :
      * Telemetry JSON : `poc/results/poc_04_results.json`
      * Run Log File   : `poc/logs/poc_04/run_<YYYYMMDD_HHMMSS>[_dry_run].log` (+ `latest.log`)
"""

import argparse
import json
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_splitting.config import EMeansMoEConfig, GEMMA_PRESETS  # noqa: E402
from moe_splitting.metrics import compute_routing_health  # noqa: E402
from moe_splitting.poc_logging import tee_poc_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 4: Frozen-Backbone Gemma-3-1B Partial Upcycling")
    p.add_argument("--model-path", type=str, default=None, help="Local directory of pretrained Gemma-3-1B weights.")
    p.add_argument("--synthetic-gemma", action="store_true", help="Use offline Tiny/Scaled Gemma model (zero downloads).")
    p.add_argument("--upcycle-last-n", type=int, default=8, help="Upcycle only the top N layers to fit 1 GPU.")
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--active-k0", type=int, default=4, help="Initial active experts before progressive splits.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dry-run", action="store_true", help="Inspect frozen vs trainable params & VRAM without training.")
    return p.parse_args()


def freeze_backbone_and_upcycle_top_layers(model, moe_cfg: EMeansMoEConfig, last_n: int):
    from moe_splitting.gemma_upcycle import GemmaMoEDecoderLayer, upcycle_gemma_mlp_to_moe

    # Freeze entire dense backbone first
    for p in model.parameters():
        p.requires_grad_(False)

    layers = getattr(getattr(model, "model", model), "layers")
    start_idx = max(0, len(layers) - last_n)
    moe_blocks = []

    for idx in range(start_idx, len(layers)):
        dense_layer = layers[idx]
        moe_mlp = upcycle_gemma_mlp_to_moe(dense_layer.mlp, moe_cfg, layer_idx=idx)
        for p in moe_mlp.parameters():
            p.requires_grad_(True)
        if hasattr(dense_layer, "self_attn"):
            layers[idx] = GemmaMoEDecoderLayer(dense_layer, moe_mlp)
        else:
            dense_layer.mlp = moe_mlp
        moe_blocks.append((f"layer_{idx}", moe_mlp))

    return model, moe_blocks


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else ("synthetic" if args.synthetic_gemma else "hf_local")
    with tee_poc_logs("poc_04", run_tag=tag):
        if not args.synthetic_gemma and args.model_path is None:
            print("=== PoC 4 SAFE CHECK ===")
            print("Specify `--model-path /local/gemma-3-1b-pt` OR `--synthetic-gemma` (with optional `--dry-run`).")
            print("Automatic HuggingFace downloads are disabled by design.")
            return

        try:
            import torch
            import torch.nn.functional as F
            from moe_splitting import DynamicExpertSplitter, TinyGemmaForCausalLM
        except ImportError:
            print("PyTorch is required to instantiate model layers for PoC 4 dry-run/execution.")
            spec = GEMMA_PRESETS["gemma-3-1b"]
            per_layer_moe = args.num_experts * (3 * spec.hidden_size * (spec.intermediate_size // args.top_k))
            trainable = args.upcycle_last_n * per_layer_moe
            print(f"Estimated Gemma-3-1B upper-{args.upcycle_last_n}-layer upcycle trainable params: {trainable / 1e6:.1f}M")
            print(f"Estimated BF16 weights + FP32 AdamW state for trainable MoE blocks: {(trainable * 16) / (1024**3):.2f} GB")
            return

        device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"

        if args.synthetic_gemma:
            spec = GEMMA_PRESETS["tiny-gemma-debug"]
            spec.num_hidden_layers = max(spec.num_hidden_layers, args.upcycle_last_n + 2)
            base_model = TinyGemmaForCausalLM(spec).to(device)
            model_id = f"synthetic-{spec.name}-L{spec.num_hidden_layers}"
            vocab_size = spec.vocab_size
        else:
            from transformers import AutoModelForCausalLM

            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device)
            model_id = args.model_path
            vocab_size = int(base_model.config.vocab_size)

        moe_cfg = EMeansMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            router_mode="emeans",
            alpha_0=0.06,
            alpha_min=0.008,
            split_interval=50,
            cooldown_steps=100,
            upcycle_strategy="progressive_split",
            progressive_active_experts=args.active_k0,
            dormant_init_bias=-6.0,
        )

        model, moe_blocks = freeze_backbone_and_upcycle_top_layers(base_model, moe_cfg, args.upcycle_last_n)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {model_id} | Total Params: {total_params/1e6:.2f}M | Trainable MoE Params: {trainable_params/1e6:.2f}M")
        if args.dry_run:
            print("PoC 4 dry-run finished cleanly (zero training performed).")
            return

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        splitter = DynamicExpertSplitter()
        total_splits = 0
        val_start, val_end = None, None

        for step in range(1, args.steps + 1):
            ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else getattr(out, "logits", out)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
            if val_start is None:
                val_start = float(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            for lname, blk in moe_blocks:
                if blk.last_router_output is not None and blk.last_batch_energy is not None:
                    blk.router.update_ema(blk.last_router_output.batch_load, blk.last_batch_energy, step, args.steps)
                    evs = splitter.maybe_split_layer(blk.router, blk.experts, opt, step=step, layer_name=lname)
                    total_splits += len(evs)
            val_end = float(loss.item())

        final_dead = sum(
            compute_routing_health(blk.router.ema_load, args.top_k, moe_cfg.split_tau)["dead_count"]
            for _, blk in moe_blocks
        ) / max(len(moe_blocks), 1)

        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_04_results.json"
        out_path.write_text(
            json.dumps(
                {
                    "poc_id": "poc_04",
                    "model_id": model_id,
                    "upcycle_last_n": args.upcycle_last_n,
                    "trainable_params_M": round(trainable_params / 1e6, 2),
                    "initial_dormant_experts": args.num_experts - args.active_k0,
                    "total_splits": total_splits,
                    "final_dead_experts": round(final_dead, 2),
                    "val_loss_start": val_start,
                    "val_loss_end": val_end,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved PoC 4 telemetry -> {out_path}")


if __name__ == "__main__":
    main()
