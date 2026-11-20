# Proof-of-Concept (PoC) Suite — De-Risking Compute Requests

This folder holds the **4-Tier Proof-of-Concept (PoC) suite** designed to verify every theoretical and systems claim of **MoE Dynamic Expert Splitting** (`e-means` + Gemma Sparse Upcycling) using **minimal resources** before applying for an `8× H100` node (`1,100–1,400 GPU-hours`).

> **Safe Defaults**: Scripts never download HuggingFace checkpoints or datasets behind your back. GPU training scripts require explicit dataset/model flags (and include `--dry-run` / offline synthetic modes for validation).

---

## PoC Tiers at a Glance

| Tier | Script | Target Hardware | Est. Runtime / Cost | Core Evidence Produced |
| :---: | :--- | :--- | :--- | :--- |
| **PoC 1** | [`poc_01_math_and_profiler.py`](./poc_01_math_and_profiler.py) | Local CPU (NumPy only) | `< 5s` • `\$0` | `< 1e-15` $-\tau_r \ln(2)$ softmax invariance proof + Gemma FLOP/VRAM/`<25 KB` sync audit |
| **PoC 2** | [`poc_02_tiny_gemma_continuity.py`](./poc_02_tiny_gemma_continuity.py) | Free Colab T4 / 1× GPU (or CPU dry) | `~15 min` • `\$0–\$0.5` | Split-step loss continuity ($\Delta \mathcal{L} \approx 0$ vs. unshifted spike) + `4 -> 0` dead-expert revival on `TinyGemma` |
| **PoC 3** | [`poc_03_proxy_collapse_and_scaling.py`](./poc_03_proxy_collapse_and_scaling.py) | 1× RTX 4090 / 1× A100 | `6–12 hrs` • `\$5–\$20` | Real multi-domain Perplexity & Imbalance (`Unbalanced` vs `Switch` vs `DeepSeek` vs `Ours`) + `2x2x3` $\alpha_{\min}$ scaling fit |
| **PoC 4** | [`poc_04_gemma3_1b_frozen_upcycle.py`](./poc_04_gemma3_1b_frozen_upcycle.py) | 1× A100 (40/80GB) or 4090 partial | `8–15 hrs` • `\$10–\$30` | Pretrained `Gemma-3-1B` frozen-attention partial upcycling (`progressive_split`) + domain specialization heatmap |
| **Report** | [`compile_poc_report.py`](./compile_poc_report.py) | Local CPU | `< 1s` | Aggregates `poc/results/poc_0*.json` into `poc/results/COMPILED-POC-REPORT.md` |

---

## Automatic Run Logging (`poc/logs/`)

Every PoC script automatically tees console output + system/GPU environment headers into structured subfolders via [`poc_logging.py`](../src/moe_splitting/poc_logging.py):
- `poc/logs/poc_01/run_<YYYYMMDD_HHMMSS>.log` & `latest.log`
- `poc/logs/poc_02/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/poc_03/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/poc_04/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/compile_report/run_<YYYYMMDD_HHMMSS>.log` & `latest.log`

---

## Step-by-Step Execution Workflow

### 1. Run Offline PoC 1 Immediately
```bash
python poc/poc_01_math_and_profiler.py
python poc/compile_poc_report.py
```

### 2. Run PyTorch Continuity Ablation (PoC 2 on Free T4 / Single GPU)
```bash
# Dry-run inspection (zero training):
python poc/poc_02_tiny_gemma_continuity.py --dry-run

# Full PoC 2 run comparing emeans vs emeans_no_bias_shift vs aux_loss vs deepseek_bias vs unbalanced:
python poc/poc_02_tiny_gemma_continuity.py --steps 1000 --d-model 256 --layers 6
```

### 3. Run Multi-Domain Proxy & Mini Scaling Law (PoC 3 on 1× 4090 / A100)
```bash
# Dry-run grid & memory estimate:
python poc/poc_03_proxy_collapse_and_scaling.py --dry-run

# Real multi-domain run (supply local tokenized/jsonl shards or synthetic domain benchmark):
python poc/poc_03_proxy_collapse_and_scaling.py --run-all --steps 2000 --data-dir /path/to/multi_domain_shards
```

### 4. Run Low-VRAM Frozen-Backbone Gemma-3-1B Upcycle (PoC 4 on 1× A100 / 4090)
```bash
# Dry-run using offline config-faithful Gemma architecture (zero downloads):
python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 8 --dry-run

# With locally downloaded or explicit HF Gemma checkpoint on GPU cluster:
python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path /path/to/gemma-3-1b-pt --upcycle-last-n 8 --steps 800
```

### 5. Compile All Results Into Grant / Cluster Justification
```bash
python poc/compile_poc_report.py
```
Then pair `poc/results/COMPILED-POC-REPORT.md` with [`PROPOSAL-TEMPLATE.md`](./PROPOSAL-TEMPLATE.md).

### 6. Run Full Diagnostics, Syntax, Baseline & Domain Verification
```bash
python scripts/run_all_diagnostics.py
```
Checks syntax/bytecode across all `.py` files, config JSON/SVG/Markdown integrity, negative invariance ablations, multi-seed PoC 1 baseline dominance, and PyTorch tests when installed (`poc/results/diagnostic_suite_results.json` & `poc/logs/diagnostics/latest.log`).

---

## Related Docs
- [Base Terms & Concepts](../docs/basic-terms/00-INDEX.md)
- [Full Gemma Implementation Plan](../docs/architecture/IMPLEMENTATION-PLAN-GEMMA.md)
