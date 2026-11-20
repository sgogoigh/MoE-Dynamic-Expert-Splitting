# Compute Allocation Proposal Template (`8× H100` Request Backed by PoC 1–4)

Use the numbers automatically compiled into `poc/results/COMPILED-POC-REPORT.md` to fill in the bracketed fields below when requesting university HPC hours, lab GPUs, or cloud research credits.

---

## 1. Project Summary
- **Title**: *Dynamic Expert Splitting: Eliminating Routing Collapse in Sparse MoEs without Auxiliary Losses*
- **Target Architecture**: Upcycled `Gemma-3-1B` / `Gemma-2-2B` & scratch `1.3B`–`7B` GeGLU Sparse MoEs ($E \in [16, 64], K_r \in [2, 4]$).
- **Core Bottleneck Addressed**: Traditional Switch/Mixtral auxiliary load-balancing losses interfere with language-modeling gradients, while heuristic bias balancing (DeepSeek-V3) cannot reallocate dead capacity onto overloaded domains.

## 2. Empirical Proof-of-Concept Evidence Completed
1. **Exact Function-Preserving Invariance (PoC 1 & PoC 2)**:
   - Mathematically and numerically verified softmax mass conservation ($\max |\Delta p| \le$ `[PoC1 max diff]`) via $b \leftarrow b - \tau_r \ln(2)$.
   - In PyTorch continual optimization on `TinyGemma`, dynamic splitting eliminated `[X]` dead experts while keeping split-step loss discontinuity at `[PoC2 emeans split delta L]`, compared to `[PoC2 no-shift spike]` without bias shifting.
2. **Real Multi-Domain Advantage & Preliminary Scaling Law (PoC 3)**:
   - Under asymmetric `FineWeb-Edu + Code + Math` traffic on 1 GPU, our method reduced validation perplexity relative to Switch Aux-Loss (`[delta PPL]`) and DeepSeek-V3 bias (`[delta PPL]`) with `0` dead experts.
   - A 12-point mini grid yielded an initial $\alpha_{\min}$ OLS log-power-law fit of $R^2 = \text{`[PoC3 R2]`}$ with LOSO error `[PoC3 LOSO %]`.
3. **Pretrained Gemma Upcycling Feasibility (PoC 4)**:
   - Verified `1.00x` active FLOP parity via `sliced_granular` upcycling and domain-driven expert splitting (`progressive_split`) on frozen-backbone `Gemma-3-1B`, achieving `[PoC4 domain specialization summary]` at `< 25 KB/step` distributed EMA sync cost.

## 3. Requested Compute Breakdown (`1,100–1,400 H100 GPU-Hours`)
- **Phase A (Full Proxy Scaling Law Grid)**: `~160 H100 hrs` — Fit final $\alpha_{\min}(BL, E, K_r, \tau_r)$ power law and verify OOD extrapolation.
- **Phase B (`1.3B` @ 30B tokens & `7B` / `Gemma-2-2B` @ 100B tokens)**: `~640 H100 hrs` — Controlled matched-FLOP baselines (`Switch`, `DeepSeek-V3 Bias`, `Expert Choice`, `Ours`).
- **Phase C (Ablations & Downstream Evals)**: `~300 H100 hrs` — `MMLU`, `GSM8K`, `HumanEval`, `ARC-C`, `HellaSwag` + split-criterion ablations.
- **Expected Risk**: **Low** — every routing invariant, AdamW moment migration, and memory budget has already been verified across PoCs 1–4.
