# MoE-Dynamic-Expert-Splitting
Research - Eliminating Routing Collapse in Sparse MoEs without Auxiliary Losses (e_means vs k_means)

**Eliminating Routing Collapse in Sparse MoEs without Auxiliary Losses** — translating `e-means` sufficient statistics $(N_j, S_j)$, non-stationary cosine EMA floors ($\alpha_{\min}$), and function-preserving donor splitting ($b \leftarrow b - \tau_r \ln 2$) into **GeGLU Sparse MoE** architectures upcycled from **Gemma 2 / Gemma 3**.

> **Offline & Safe Setup**: Nothing in this repository automatically downloads Gemma checkpoints or launches full training runs. All exploratory profilers, mathematical proofs, and synthetic mini-runs execute locally out of the box.

---

## Repository Map

| Folder / File | Purpose |
| :--- | :--- |
| [**`docs/basic-terms/`**](./docs/basic-terms/00-INDEX.md) | Structured Markdown primers covering all base terms: `e-means` EMAs, Top-$K_r$ routing, routing collapse, $-\tau_r \ln(2)$ invariance, Gemma `GeGLU`/`RMSNorm`/`GQA`, scaling laws, and A–Z glossary |
| [**`docs/architecture/`**](./docs/architecture/IMPLEMENTATION-PLAN-GEMMA.md) | Full Gemma Dense-to-MoE upcycling blueprint, VRAM/FLOP breakdown, and Phase 0–3 execution plan |
| [**`docs/assets/images/`**](./docs/assets/images/) | Vector SVG architecture diagrams (`gemma-to-moe-upcycling.svg`, `split-invariance-geometry.svg`, `distributed-ep-dp-ema-sync.svg`) |
| **`src/moe_splitting/`** | Core modules: `config.py`, `router.py`, `experts.py`, `splitting.py`, `gemma_upcycle.py`, `metrics.py`, `scaling_law.py` |
| **`scripts/`** | Offline profiler (`explore_gemma_architecture.py`), NumPy verification mini-run (`mini_run_math_verify.py`), PyTorch TinyGemma upcycling dry-run (`mini_run_gemma_upcycle_torch.py`), and sweep planner (`plan_scaling_sweep.py`) |
| **`configs/`** | Ready JSON specs for `tiny-gemma-debug`, `gemma-3-1b`, and `gemma-2-2b` |
| [**`poc/`**](./poc/README.md) | 4-Tier Proof-of-Concept suite + result compiler & grant proposal template (`poc_01`..`poc_04` + `compile_poc_report.py`) |
| **`tests/`** | Automated math & PyTorch verification tests |

---

## Visual Architecture: Gemma Upcycling & Dynamic Splitting

![Gemma Dense-to-MoE Upcycling](./docs/assets/images/gemma-to-moe-upcycling.svg)

---

## Quick Start (Runs Immediately Without Downloads or Training)

```bash
# 1. Profile Gemma-3-1B / Gemma-2-2B / Gemma-2-9B Dense -> MoE shapes, VRAM & FLOPs
python scripts/explore_gemma_architecture.py

# 2. Prove -tau_r ln(2) softmax mass invariance + run 400-step multi-domain routing simulation
python scripts/mini_run_math_verify.py

# 3. Preview Phase-A scaling-law sweep matrix & verify OLS log-power-law fitter
python scripts/plan_scaling_sweep.py

# 4. Run offline mathematical unit tests
python -m unittest discover -s tests -p "*numpy*.py" -v      # numpy-only modules
```

### PyTorch (CPU is enough for everything below)
```bash
# CPU-only torch in a local venv (Windows paths; on Linux/macOS use .venv/bin/python)
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install numpy transformers safetensors pytest

# Upcycles offline TinyGemma -> parity check -> 50 AdamW steps, printing each split's same-batch loss change
python scripts/mini_run_gemma_upcycle_torch.py

# Every test module (unittest; pytest also works). The torch suites take a few minutes on CPU.
python -m unittest discover -s tests -v
python scripts/run_all_diagnostics.py      # same, plus a JSON scorecard; exit 2 if torch is missing
```

### What is and is not established
- **Split continuity under Top-K routing**: the `-tau_r ln 2` bias shift preserves only the dense softmax mixture. Under hard Top-K it needs **sibling routing**: the split pair holds one Top-K slot and the token goes to the more probable sibling. With it, the measured split loss change is 0, against up to 0.047 nats for plain Top-K (PoC 2, tiny Gemma). See [`docs/basic-terms/04`](./docs/basic-terms/04-Function-Preserving-Splitting-and-Adam-State.md), section 3b.
- **Open: un-merging.** In the CPU runs, sibling experts diverged in weight space (relative difference 0.23 to 0.36) but the router never separated them, so every un-merge was forced. Forced un-merges are ramped, so no single step jumps. Their total cost (+0.104 and −0.002 nats per 4-pair episode) is the same order as a plain Top-K split. Merged experts cannot donate again until they un-merge, so active experts at most double per `merge_max_steps` cycle (PoC 4: 4 → 8, then no further splits within 200 steps).
- **Open: the CPU proxies barely collapse.** e-means splits only experts below the dead threshold (0.1× fair share). With the HF-matching init:
  - In PoC 2 Part B and PoC 3 Part A, no expert got that low, so e-means never split and matched the unbalanced baseline exactly.
  - In PoC 3 Part B it split only 1–2 times per run, and all 8 α_min cells were flat, so no scaling law was fitted.
  - Splitting is properly exercised only in PoC 2 Part A (forced `progressive_split`) and PoC 4 (upcycling starts experts dead).
- **Comparisons are toy-scale.** In the PoC 1 toy simulation (3 seeds), e-means beats the unbalanced baseline. DeepSeek-V3 bias balancing beats e-means on dead experts and balance on every seed, and the aux-loss stand-in matches or beats it on most. PoC 2 Part B and PoC 3 are informational CPU runs. None of this is evidence at scale.
