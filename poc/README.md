# Proof-of-Concept (PoC) Suite — De-Risking Compute Requests

This folder holds the **4-tier proof-of-concept (PoC) suite** for **MoE Dynamic Expert Splitting** (`e-means` + Gemma sparse upcycling), run with minimal resources before applying for an `8× H100` node (`1,100–1,400 GPU-hours`).

Every PoC separates **sanity checks** (gating: properties the code must have, e.g. a split does not change the loss) from **informational comparisons** (measurements on toy or small runs, e.g. e-means vs DeepSeek bias balancing). A passed PoC means the sanity checks passed; it is not evidence that the method wins at scale. Every PoC takes `--out-dir` (or `MOE_POC_RESULTS_DIR`), and `MOE_POC_LOGS_DIR` redirects the logs.

> **Known limit of the CPU defaults**: e-means splits only experts whose EMA load is below `split_tau` (0.1) × fair share. In the CPU runs of PoC 2 Part B and PoC 3 Part A, no expert ever fell that low, so e-means never split and its numbers equal the unbalanced baseline. In PoC 3 Part B it split only 1–2 times per run (the `splits/run` column), and every α_min cell was flat. Those comparisons therefore say almost nothing about e-means yet. Splitting is exercised in PoC 2 Part A (forced `progressive_split`) and PoC 4 (upcycling starts experts dead). Before these PoCs can compare balancing methods, they need a setup where collapse actually happens: longer runs, larger scale, or a deliberately skewed router init.

> **Safe Defaults**: Scripts never download HuggingFace checkpoints or datasets behind your back. GPU training scripts require explicit dataset/model flags (and include `--dry-run` / offline synthetic modes for validation).

---

## PoC Tiers at a Glance

| Tier | Script | Target Hardware | Est. Runtime / Cost | Core Evidence Produced |
| :---: | :--- | :--- | :--- | :--- |
| **PoC 1** | [`poc_01_math_and_profiler.py`](./poc_01_math_and_profiler.py) | Local CPU (NumPy only) | `< 10s` • `\$0` | Dense softmax invariance (~1e-15), Top-$K_r$ invariance: plain routing breaks it, sibling routing keeps it (~1e-15); Gemma FLOP/VRAM audit; router statistics sync (~2–37 MB/step, ~0.02% of gradient sync); toy routing simulation over 3 seeds (informational) |
| **PoC 2** | [`poc_02_tiny_gemma_continuity.py`](./poc_02_tiny_gemma_continuity.py) | CPU (tiny) or 1× GPU | `~7 min` CPU at `--d-model 64 --layers 2 --steps 240` | Part A (gating): split dL on a fixed probe batch, sibling routing vs plain Top-$K_r$ vs no bias shift; un-merge episodes (total cost of a forced un-merge, ramp without a completion jump). Part B (informational): balancing comparison from `noisy_clone` |
| **PoC 3** | [`poc_03_proxy_collapse_and_scaling.py`](./poc_03_proxy_collapse_and_scaling.py) | CPU (tiny defaults) or 1× RTX 4090 / A100 (proxy) | CPU defaults `~30 min` (`--seeds 2`) | Comparators on held-out loss per domain (informational); $\alpha_{\min}$ ladder per (E, K_r, batch, tau_r) cell with flat / at-edge flags; power-law fit only on cells with an identified optimum |
| **PoC 4** | [`poc_04_gemma3_1b_frozen_upcycle.py`](./poc_04_gemma3_1b_frozen_upcycle.py) | CPU (`--synthetic-gemma`) or 1× A100 / 4090 | synthetic `~1–3 min` | Frozen-backbone partial upcycling (`.mlp` replaced only): held-out loss dense -> step 0 (parity) -> after training, domain x expert routing table |
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

# Quick CPU run (Part A gating + Part B informational):
python poc/poc_02_tiny_gemma_continuity.py --steps 240 --d-model 64 --ff-dim 256 --layers 2 --batch-size 8 --seq-len 32 --merge-max-steps 60 --unmerge-ramp-steps 40

# Larger run on a GPU:
python poc/poc_02_tiny_gemma_continuity.py --steps 1000 --d-model 256 --layers 6
```

### 3. Run Multi-Domain Proxy & Mini Scaling Law (PoC 3 on 1× 4090 / A100)
```bash
# Dry-run grid & memory estimate:
python poc/poc_03_proxy_collapse_and_scaling.py --dry-run

# Tiny CPU defaults (synthetic 3-domain data), 2 seeds so the flat test has a noise estimate:
python poc/poc_03_proxy_collapse_and_scaling.py --seeds 2

# ~180M proxy on one GPU with real shards (prose.npy / code.npy / math.npy, last 10% held out):
python poc/poc_03_proxy_collapse_and_scaling.py --d-model 512 --ff-dim 1536 --layers 12 --heads 8 --kv-heads 4 --vocab 32000 --seq-len 512 --steps 1500 --sweep-steps 400 --seeds 3 --data-dir /path/to/multi_domain_shards
```

### 4. Run Low-VRAM Frozen-Backbone Gemma-3-1B Upcycle (PoC 4 on 1× A100 / 4090)
```bash
# Dry-run using offline config-faithful Gemma architecture (zero downloads):
python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 2 --dry-run
python poc/poc_04_gemma3_1b_frozen_upcycle.py --synthetic-gemma --upcycle-last-n 2 --steps 200 --seq-len 64

# With locally downloaded or explicit HF Gemma checkpoint on GPU cluster:
python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path /path/to/gemma-3-1b-pt --upcycle-last-n 8 --steps 800 --data-dir /path/to/gemma-tokenized-shards
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
Runs every offline test module, then every PyTorch test module (`tests/test_*torch*.py`) through unittest. Without torch the scorecard says `complete: false` and the exit code is 2; a skipped torch suite is never counted as passed (`poc/results/diagnostic_suite_results.json` & `poc/logs/diagnostics/latest.log`).

---

## Related Docs
- [Base Terms & Concepts](../docs/basic-terms/00-INDEX.md)
- [Full Gemma Implementation Plan](../docs/architecture/IMPLEMENTATION-PLAN-GEMMA.md)
