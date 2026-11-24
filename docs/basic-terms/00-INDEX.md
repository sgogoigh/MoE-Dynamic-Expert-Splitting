# Base Terms & Core Concepts Index (`docs/basic-terms/`)

This directory breaks down every foundational term, equation, system bottleneck, and architectural concept required before reading or running the **MoE Dynamic Expert Splitting** (`e-means` + Gemma upcycling) codebase.

---

## Recommended Learning Sequence

| Order | Document | Core Topics Covered |
| :---: | :--- | :--- |
| **01** | [**`e-means` Primitives & EMA**](./01-e-means-Primitives-and-EMA.md) | Why `e-means` replaces `k-means`, sufficient statistics $(N_j, S_j)$, cosine decay, non-stationary tracking floor $\alpha_{\min}$, donor splitting |
| **02** | [**Sparse MoE Routing Fundamentals**](./02-Sparse-MoE-Routing-Fundamentals.md) | Linear router $(W_r, b, \tau_r)$, Softmax + Top-$K_r$ gating, token-choice vs. expert-choice dispatch, All-to-All, capacity factor & token dropping |
| **03** | [**Routing Collapse & Load Balancing**](./03-Routing-Collapse-and-Load-Balancing.md) | Rich-get-richer dead experts, Switch/GShard auxiliary loss, z-loss, DeepSeek-V3 bias balancing, imbalance / Gini / routing entropy metrics |
| **04** | [**Function-Preserving Splitting & Adam State**](./04-Function-Preserving-Splitting-and-Adam-State.md) | Net2Net splitting, exact $-\tau_r \ln(2)$ softmax invariance proof (dense mixture), why it fails under Top-$K_r$ and how sibling routing restores it, centroid hyperplane, un-merging and its measured cost, orthogonal perturbation $\delta W_\perp$, cooldown windows, Adam $(m, v)$ migration |
| **05** | [**Gemma Architecture & Sparse Upcycling**](./05-Gemma-Architecture-and-Sparse-Upcycling.md) | Gemma 2 / Gemma 3 blocks (`GeGLU`, `(1+w)` RMSNorm, GQA, logit soft-cap), Sparse Upcycling from dense checkpoints, splitting strategies |
| **06** | [**Scaling Laws & Master A–Z Glossary**](./06-Scaling-Laws-and-Glossary-A-to-Z.md) | Power-law prediction of $\alpha_{\min}(BL, E, K_r, \tau_r)$, log-space OLS, LOSO/OOD validation, complete symbol & term reference |

---

## Visual Architecture Guides (`../assets/images/` & `../architecture/`)
- [**Gemma Dense-to-MoE Upcycling & Splitting Diagram**](../assets/images/gemma-to-moe-upcycling.svg)
- [**Softmax Probability Invariance Geometry**](../assets/images/split-invariance-geometry.svg)
- [**Distributed EP/DP Async EMA Sync Architecture**](../assets/images/distributed-ep-dp-ema-sync.svg)
- [**Full Gemma Implementation & Experimental Plan**](../architecture/IMPLEMENTATION-PLAN-GEMMA.md)
- [**Repository README**](../../README.md)
