# 06. Scaling Laws & Complete A–Z Glossary

> **Goal**: Define the scaling-law methodology predicting optimal tracking floor $\alpha_{\min}$ and serve as an A–Z quick lookup for every mathematical symbol and engineering term used in `MoE-Dynamic-Expert-Splitting`.

---

## 1. Non-Stationary Tracking Floor Scaling Law

Following the empirical power-law methodology in `e-means` (Bessa et al., 2026, Eq. 6–7), we avoid dataset-by-dataset manual tuning of EMA reactivity $\alpha_{\min}$ by fitting a log-linear Ordinary Least Squares (OLS) predictor on proxy runs:

$$
\alpha_{\min}^* = C_\alpha \cdot (BL)^{a_B} \cdot E^{a_E} \cdot K_r^{a_K} \cdot \tau_r^{a_\tau}
$$

Taking natural logarithms converts this into standard multiple linear regression:
$$
\ln \alpha_{\min}^* = \ln C_\alpha + a_B \ln(BL) + a_E \ln E + a_K \ln K_r + a_\tau \ln \tau_r
$$

### Expected Exponent Signs (Intuition Check)
- **$a_B > 0$** *(batch tokens $BL$)*: Larger global batches average out per-token stochasticity while performing fewer optimizer steps per epoch, allowing higher EMA responsiveness per step (just as `e-means` found $B^{0.90}$!).
- **$a_E < 0$** *(expert count $E$)*: More experts reduce average per-expert token hits per batch ($BL \cdot K_r / E$), requiring smaller $\alpha_{\min}$ (longer step memory) to avoid false dead-expert triggers.
- **$a_K > 0$** *(Top-$K_r$)*: Routing each token to more active experts increases hits per expert, supporting larger $\alpha_{\min}$.

We validate generalization using **Leave-One-Scale-Out (LOSO)** and **Out-of-Distribution (OOD)** extrapolation from small proxy models (`~150M–300M` or `tiny-gemma`) up to `1.3B–7B` / `Gemma-2-2B` MoEs (see `src/moe_splitting/scaling_law.py`).

---

## 2. Complete A–Z Glossary & Symbol Table

| Symbol / Term | Meaning & Role in This Repo |
| :--- | :--- |
| **$\alpha_0, \alpha_{\min}, \alpha(t)$** | Initial, floor, and step-$t$ EMA step size ($\alpha(t) = 1 - \gamma_t$). |
| **$\gamma_t$** | EMA decay factor ($\gamma_t = 1 - \alpha(t)$); weight placed on prior EMA history. |
| **$N_j$ (`ema_load`)** | Smoothed FP32 token selection fraction for expert $j$; sums to $\approx K_r$. |
| **$S_j$ (`ema_energy`)** | Smoothed FP32 routing energy $\sum g_j(x) \|h_j(x)\|_2^2$ weighting active gate + activation strength. |
| **$E$ (`num_experts`)** | Total number of experts in each MoE Transformer layer. |
| **$K_r$ (`top_k`)** | Active experts executed per token ($K_r \ll E$). |
| **$\tau_r$ (`tau_r`)** | Router softmax temperature dividing logits $(W_r x + b) / \tau_r$. |
| **$\tau$ (`split_tau`)** | Dead-expert threshold fraction (default `0.1`): flags $N_j < \tau \cdot (K_r / E)$. |
| **$M_{\text{split}}$** | Split-check cadence in optimizer steps (e.g., every `100–250` steps). |
| **$\delta$ (`orth_eps`)** | Relative scale of orthogonal perturbation $\pm \delta W_\perp$ breaking child symmetry. |
| **All-to-All (`A2A`)** | Distributed NCCL collective routing tokens across Expert-Parallel (`EP`) GPUs. |
| **Aux Loss ($\mathcal{L}_{\text{aux}}$)** | Switch/GShard penalty $\lambda_{\text{aux}} E \sum f_j P_j$ omitted by our method. |
| **Capacity Factor ($CF$)** | Buffer size multiplier per expert causing dropped tokens when overloaded. |
| **Cooldown Window** | Lockout timer (`2 * M_split` steps) protecting newborn child experts from immediate re-splitting. |
| **Donor Expert ($j^*$)** | Overloaded expert ($N_{j^*} > K_r/E$) cloned to revive a dead slot ($j_{\text{dead}}$). |
| **GeGLU** | Gemma's 3-matrix gated MLP using `gelu_pytorch_tanh` on `gate_proj(x) * up_proj(x)`. |
| **GQA** | Grouped-Query Attention sharing Key/Value heads across Query head groups. |
| **LOSO / OOD** | Leave-One-Scale-Out / Out-of-Distribution checks proving scaling-law generalization. |
| **Max-to-Mean Ratio** | $\max_j N_j / (K_r/E)$; primary measure of routing straggler imbalance. |
| **Net2Net Splitting** | Function-preserving network growth/splitting preserving initial forward outputs. |
| **`RMSNorm (1 + w)`** | Gemma's unit-offset Root Mean Square layer normalization (`weight` initialized at `0`). |
| **Sparse Upcycling** | Converting a pretrained dense Transformer's MLP weights into MoE experts (Komatsuzaki 2023). |
| **Z-Loss ($\mathcal{L}_z$)** | Optional tiny router logit norm stabilizer $\text{mean}((\log\sum e^{z_j})^2)$. |

---

## 3. Sources
- **OLS Scaling Methodology**: Bessa et al. (2026), *e-means* ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501), Section 4 & Appendix C)
- **Repository Companion Guide**: `../architecture/IMPLEMENTATION-PLAN-GEMMA.md`
