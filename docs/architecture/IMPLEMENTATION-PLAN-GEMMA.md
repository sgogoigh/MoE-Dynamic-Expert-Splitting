# Gemma Upcycling & Dynamic Expert Splitting Implementation Plan

> **Important Constraint Honored**: This repository is set up for **offline exploration, verification, and cluster-ready preparation**. No pretrained Gemma weights are downloaded and no full pretraining jobs are launched locally. All math, routing dynamics, upcycling shapes, and mini-runs can be inspected cleanly right now.

---

## 1. Visual System Architecture

![Gemma Dense-to-MoE Upcycling & Dynamic Splitting](../assets/images/gemma-to-moe-upcycling.svg)

---

## 2. Why Pretrained Gemma + `e-means` Dynamic Expert Splitting Works Cleanly

Upcycling a dense pretrained Transformer into a Sparse MoE (**Komatsuzaki et al., ICLR 2023**) avoids paying full scratch-pretraining cost while testing routing mechanics on real pretrained hidden manifolds:
1. **GeGLU 3-Matrix Compatiblity**: Both `Gemma 2` and `Gemma 3` use `GeGLU(x) = (gelu_pytorch_tanh(gate_proj(x)) * up_proj(x)) @ down_proj.T`, matching modern gated MoE expert blocks (`SwiGLU`/`GeGLU`) cleanly.
2. **Step-0 Preservation**: `sliced_parity` (FLOP-matched; the default in `configs/gemma_*`) and `noisy_clone` (K_r x dense FLOPs) reproduce the dense MLP exactly at zero noise for $K_r \ge 2$ with renormalised weights (`tests/test_upcycle_torch.py`, HF Gemma 2 and Gemma 3). `sliced_granular` does NOT (a token whose Top-$K_r$ holds two experts with the same slice drops part of the FFN; ~0.7 logit error measured) and is kept only for random-init proxies (PoC 3)—while **`e-means` Dynamic Expert Splitting** removes the need for auxiliary loss gradients that would distort Gemma's delicate pretrained residual geometry!
3. **Progressive Specialization Testbed (`progressive_split`)**: By placing $K_0$ active clones and $(E - K_0)$ dormant slots (or inducing domain skew during continual training on code/math/prose), we directly measure how fast `DynamicExpertSplitter` carves dead slots from overloaded donor experts. Splits are loss-continuous under Top-$K_r$ only with sibling routing (see `docs/basic-terms/04`, section 3b); un-merging a pair that has not separated is not free and is measured separately.

---

## 3. Repository Architecture (`src/moe_splitting/` & `scripts/`)

```mermaid
flowchart TD
    CFG["src/moe_splitting/config.py<br/>Presets: tiny-gemma, gemma-3-1b, gemma-2-2b, gemma-2-9b"] --> UPC["src/moe_splitting/gemma_upcycle.py<br/>TinyGemma + HF Gemma Dense->MoE Converter"]
    CFG --> RTR["src/moe_splitting/router.py<br/>EMeansMoERouter + Switch/DeepSeek Baselines"]
    CFG --> EXP["src/moe_splitting/experts.py<br/>GeGLUMoEExperts (gelu_pytorch_tanh)"]
    RTR & EXP --> SPL["src/moe_splitting/splitting.py<br/>DynamicExpertSplitter (Bias Shift + Orth Perturb + AdamW Sync)"]
    RTR --> MET["src/moe_splitting/metrics.py<br/>Load Imbalance, Dead Count, Gini, Entropy"]
    MET --> SCL["src/moe_splitting/scaling_law.py<br/>OLS Power-Law Fitter & LOSO Validator"]
```

---

## 4. Target Gemma Configurations Profiled (Zero-Download Estimates)

Run `python scripts/explore_gemma_architecture.py` to inspect exact per-layer tensor shapes, BF16 weights, AdamW optimizer states, and active FLOPs/token for:
1. **`tiny-gemma-debug`** ($d=128, D_{ff}=512, L=4$): Instant CPU/GPU unit-test & mini-run model.
2. **`gemma-3-1b`** ($d=1152, D_{ff}=6912, L=26$) upcycled to `E=16, K_r=2` (`noisy_clone` & `sliced_granular`).
3. **`gemma-2-2b`** ($d=2304, D_{ff}=9216, L=26$) upcycled to `E=16, K_r=2` & `E=32, K_r=2`.
4. **`gemma-2-9b`** ($d=3584, D_{ff}=14336, L=42$) upcycled to `E=64, K_r=4`.

---

## 5. Multi-Phase Research & Execution Workflow

### Phase 0: Local Exploratory & Mathematical Verification *(Runnable Now — Offline)*
1. `python scripts/explore_gemma_architecture.py` — Prints parameter counts, VRAM budgets, `sliced_granular` vs `noisy_clone` FLOPs, for all Gemma targets. (Router statistics all-reduce per step is $L \cdot E \cdot (d+4)$ fp32 values, about 1.9 MB for Gemma-3-1B at $E=16$, dominated by the token sums used for the centroid split; ~0.02% of the bf16 gradient all-reduce, see PoC 1.)
2. `python scripts/mini_run_math_verify.py` — Executes:
   - Machine-precision numerical check (`< 1e-13`) of the $-\tau_r \ln(2)$ softmax mass invariance theorem.
   - 400-step multi-domain non-stationary simulation comparing **No-Balance (Collapse)**, **Switch Aux-Loss**, **DeepSeek-V3 Bias**, and **`e-means` Dynamic Expert Splitting**.
3. `python -m unittest discover -s tests -p "*numpy*.py"` — Unit tests on all mathematical guarantees.

### Phase 1: Cluster Dry-Run on PyTorch `TinyGemma` *(Zero Download Needed)*
When running in an environment with PyTorch installed:
- `python scripts/mini_run_gemma_upcycle_torch.py` builds a random-weight `TinyGemmaForCausalLM`, runs a dense forward pass, converts it into `TinyGemmaMoE` via `upcycle_gemma_model`, checks logit parity, runs 50 AdamW steps with `DynamicExpertSplitter` active and prints each split's same-batch loss change (measured after the optimizer step, statistics frozen).

### Phase 2: Scaling-Law Proxy Sweeps on Gemma (`~160 H100 GPU-hours`)
- Use `python scripts/plan_scaling_sweep.py` to generate the exact `(BL, E, K_r, tau_r, alpha_min)` sweep grid.
- Continually pretrain small proxy MoEs on mixed **Code + Math + FineWeb-Edu** corpora (intentionally multi-domain to stress specialization vs. collapse).
- Fit $\alpha_{\min}^* = C_\alpha (BL)^{a_B} E^{a_E} K_r^{a_K} \tau_r^{a_\tau}$ via `src/moe_splitting/scaling_law.py`.

### Phase 3: Full Gemma-2-2B / 7B Upcycled Comparison (`~640–950 H100 GPU-hours`)
Compare matched-FLOP baselines over **30B–100B continual tokens**:
1. **Unbalanced Top-$K_r$** (collapse baseline)
2. **Switch / Mixtral Aux Loss** ($\lambda_{\text{aux}} = 0.01$)
3. **DeepSeek-V3 Dynamic Bias Balancing** ($\gamma_{\text{bias}} = 0.001$)
4. **Ours (`EMeansMoERouter` + `DynamicExpertSplitter`)**

Evaluate: Validation perplexity across domains, split-step and un-merge loss changes (measured as in PoC 2), max-to-mean load ratio, dead-expert count, tokens/sec throughput, and downstream zero-shot benchmarks (`MMLU`, `GSM8K`, `HumanEval`, `ARC-C`, `HellaSwag`).

---

## 6. Distributed Architecture Diagram

![Distributed EP/DP Async EMA Sync](../assets/images/distributed-ep-dp-ema-sync.svg)
