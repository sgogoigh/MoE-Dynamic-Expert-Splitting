# Compute Allocation Proposal Template (`8× H100` Request Backed by PoC 1–4)

Use the numbers automatically compiled into `poc/results/COMPILED-POC-REPORT.md` to fill in the bracketed fields below when requesting university HPC hours, lab GPUs, or cloud research credits.

---

## 1. Project Summary
- **Title**: *Dynamic Expert Splitting: Eliminating Routing Collapse in Sparse MoEs without Auxiliary Losses*
- **Target Architecture**: Upcycled `Gemma-3-1B` / `Gemma-2-2B` & scratch `1.3B`–`7B` GeGLU Sparse MoEs ($E \in [16, 64], K_r \in [2, 4]$).
- **Core Bottleneck Addressed**: Traditional Switch/Mixtral auxiliary load-balancing losses interfere with language-modeling gradients, while heuristic bias balancing (DeepSeek-V3) cannot reallocate dead capacity onto overloaded domains.

## 2. Empirical Proof-of-Concept Evidence Completed
1. **Function-Preserving Splits under Top-$K_r$ Routing (PoC 1 & PoC 2)**:
   - Mathematically and numerically verified softmax mass conservation ($\max |\Delta p| \le$ `[PoC1 max diff]`) via $b \leftarrow b - \tau_r \ln(2)$.
   - The bias shift alone preserves only the dense mixture; with plain Top-$K_r$ the split changes outputs by `[PoC1 plain Top-K diff]`. Sibling routing (the split pair holds one Top-$K_r$ slot) restores exact invariance (`[PoC1 sibling diff]`).
   - On `TinyGemma` (fixed probe batch, after the optimizer step, statistics frozen): split loss change `[PoC2 emeans split dL]` vs `[PoC2 plain Top-K split dL]` with the bias shift and plain Top-$K_r$, and `[PoC2 no-shift dL]` without the shift.
   - Open: un-merging a sibling pair that has not separated costs `[PoC2 un-merge episode total dL]` in total (spread over a ramp); `[PoC2 natural]/[PoC2 total]` un-merges were natural.
2. **Real Multi-Domain Advantage & Preliminary Scaling Law (PoC 3)**:
   - Under asymmetric `FineWeb-Edu + Code + Math` traffic on 1 GPU, held-out loss relative to Switch aux-loss: `[delta, mean +/- sd over seeds]`; relative to DeepSeek-V3 bias: `[delta, mean +/- sd]`; dead experts `[n per method]`. (Report whichever direction the result goes; in the PoC 1 toy simulation DeepSeek-V3 bias and the aux-loss stand-in both balance better than e-means.)
   - $\alpha_{\min}$ grid: `[n usable]/[n cells]` cells showed an interior optimum above seed noise (the rest were flat or at the ladder edge); fit `[exponents of the identified features, R2, dof, LOSO]` or `not identified at this budget`.
3. **Pretrained Gemma Upcycling Feasibility (PoC 4)**:
   - `1.00x` active FLOPs via `sliced_parity` upcycling (exact step-0 parity) and `progressive_split` on frozen-backbone `Gemma-3-1B`: held-out loss `[dense -> step 0 -> end]`, routing domain information `[PoC4 I(domain;expert)/H(domain)]`, router statistics sync `[PoC1 KB/step]` per step (`[PoC1 %]` of the gradient all-reduce).

## 3. Requested Compute Breakdown (`1,100–1,400 H100 GPU-Hours`)
- **Phase A (Full Proxy Scaling Law Grid)**: `~160 H100 hrs` — Fit final $\alpha_{\min}(BL, E, K_r, \tau_r)$ power law and verify OOD extrapolation.
- **Phase B (`1.3B` @ 30B tokens & `7B` / `Gemma-2-2B` @ 100B tokens)**: `~640 H100 hrs` — Controlled matched-FLOP baselines (`Switch`, `DeepSeek-V3 Bias`, `Expert Choice`, `Ours`).
- **Phase C (Ablations & Downstream Evals)**: `~300 H100 hrs` — `MMLU`, `GSM8K`, `HumanEval`, `ARC-C`, `HellaSwag` + split-criterion ablations.
- **Expected Risk**: **Medium**. Verified: split continuity under Top-$K_r$ (PoC 1, 2), AdamW moment migration, upcycling parity, memory budget. Open: whether sibling pairs separate on their own (in the CPU PoCs the experts diverged but the router did not separate them, so every un-merge was forced and cost the same order as a plain split); whether routing collapse appears at all in the proxy setup (in the CPU proxies almost no expert died, so e-means split 0 to 2 times per run there); whether $\alpha_{\min}$ has an identifiable optimum at scale; and whether the method beats DeepSeek-V3 bias balancing at all.
