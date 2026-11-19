# 03. Routing Collapse & Load-Balancing Approaches

> **Core Problem**: Without active balancing, unconstrained Top-$K_r$ routers suffer positive-feedback collapse onto a tiny subset of experts—yet traditional auxiliary penalties distort the language modeling objective.

---

## 1. Anatomy of Routing Collapse ("Rich-Get-Richer" Positive Feedback)

Routing collapse is the exact neural-network counterpart of **empty ("dead") clusters in `k-means`**:
1. At initialization (or early upcycling), expert $A$ happens to receive slightly more tokens than expert $B$ (or already has better FFN loss gradient alignment).
2. Because $A$ sees more tokens, $A$'s weights improve faster via AdamW, reducing token reconstruction loss when $A$ is chosen.
3. Router parameters $W_r$ receive gradients proportional to the downstream reward of lowering $\mathcal{L}_{\text{LM}}$, further raising row $w_A$ and starving $w_B$.
4. Once expert $B$ drops out of everyone's Top-$K_r$, **both FFN gradient AND router gradient for $B$ vanish** ($\mathbb{I}(g_B > 0) = 0$)—expert $B$ becomes permanently **dead**, burning VRAM for zero compute utility!

---

## 2. Existing Solutions & Their Limitations

### A. Auxiliary Load-Balancing Loss (GShard / Switch / Mixtral)
Adds a differentiable penalty weighted by $\lambda_{\text{aux}}$ (typically $0.01$):
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{aux}} \cdot E \sum_{j=1}^{E} f_j \cdot P_j
$$
where:
- $f_j = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}(j \in \mathcal{T}(x_t))$ is the **actual fraction of tokens** routed to expert $j$ (non-differentiable),
- $P_j = \frac{1}{T} \sum_{t=1}^{T} p_j(x_t)$ is the **average predicted router probability** (differentiable).

**Bottleneck — Gradient Interference**: $\nabla_{W_r} \mathcal{L}_{\text{aux}}$ pushes probabilities $P_j$ toward uniform $1/E$ even when token distributions are genuinely uneven (e.g., Python syntax vs. English prose). Too large $\lambda_{\text{aux}}$ hurts perplexity; too small $\lambda_{\text{aux}}$ permits collapse.

### B. Router Z-Loss (ST-MoE, Zoph et al. 2022)
Stabilizes numerical softmax overflow by penalizing large logit scales:
$$
\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \left(\log \sum_{j=1}^{E} \exp(z_j(x_t))\right)^2
$$
Prevents float16/bfloat16 logit blow-up, **does not fix dead-expert starvation**.

### C. DeepSeek-V3 Auxiliary-Loss-Free Dynamic Bias Balancing (2024)
Removes $\mathcal{L}_{\text{aux}}$ from backpropagation and instead adds a **detached bias $b_j^{\text{bal}}$ only during Top-$K_r$ selection** (not multiplied into the output gate weights!):
$$
g_j^{\text{select}} = s_j + b_j^{\text{bal}}, \qquad b_j^{\text{bal}} \leftarrow b_j^{\text{bal}} - \gamma_{\text{bias}} \cdot \operatorname{sign}(\text{load}_j - \bar{N})
$$
**Remaining Limitations**:
1. **No architectural capacity reallocation**: Pushing tokens into a poorly initialized/unhelpful expert via artificial bias degrades representation quality, while genuinely dense domains (math/code) never **split** into finer specialized experts.
2. **Hand-tuned $\gamma_{\text{bias}}$**: Fixed bias step lacks scaling laws over batch tokens $BL$, expert count $E$, Top-$K_r$, and temperature $\tau_r$.

---

## 3. Quantitative Routing Health Metrics Used in Our Code (`src/moe_splitting/metrics.py`)

| Metric | Formula | Healthy Target |
| :--- | :--- | :--- |
| **Max-to-Mean Load Ratio** | $\frac{\max_j N_j}{\bar{N}} = \frac{\max_j N_j}{K_r / E}$ | Close to $1.0\text{–}1.5\times$ (collapse: $>4\times$ to $E\times$) |
| **Min-to-Fair Load Ratio** | $\frac{\min_j N_j}{K_r / E}$ | $> 0.1$ ($\tau = 0.1$ dead threshold) |
| **Dead Expert Count** | $\sum_{j=1}^E \mathbb{I}(N_j < \tau \cdot K_r / E)$ | **`0` dead experts** |
| **Normalized Routing Entropy** | $H_{\text{norm}} = -\frac{1}{\ln E} \sum_{j=1}^E \hat{f}_j \ln \hat{f}_j$ ($\hat{f}_j = f_j / K_r$) | Near $0.90\text{–}1.00$ (0 = total single-expert collapse) |
| **Gini Coefficient** | $\frac{\sum_{i,j} |f_i - f_j|}{2 E \sum_j f_j}$ | Near $0.0\text{–}0.15$ |

---

## 4. Sources & Videos
- **ST-MoE (Z-loss)**: Zoph et al. (2022), *Designing Effective Sparse Expert Models* ([arXiv:2202.08906](https://arxiv.org/abs/2202.08906))
- **DeepSeek-V3**: DeepSeek-AI (2024), *Auxiliary-Loss-Free Load Balancing* ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
- **Video Search**: [Vizuara DeepSeek Aux-Loss-Free MoE Explained (YouTube)](https://www.youtube.com/results?search_query=Vizuara+DeepSeek+Auxiliary+Loss+Free+MoE)
