# 04. Function-Preserving Splitting & AdamW State Migration

> **Core Mathematical Contribution**: How to dynamically replace a dead expert slot $j_{\text{dead}}$ by cloning an overloaded donor $j^*$ so the network's forward predictions and training loss remain **mathematically invariant at the exact split step** ($\Delta \mathcal{L}_{\text{LM}} = 0$).

---

## 1. Visual Intuition

![Softmax Probability Invariance & Splitting Geometry](../assets/images/split-invariance-geometry.svg)

---

## 2. Why Naive Splitting Causes Catastrophic Loss Spikes

Suppose donor expert $j^*$ has pre-temperature router score $u_{j^*} = w_{j^*}^\top x + b_{j^*}$ and logit $z_{j^*} = u_{j^*} / \tau_r$.
If you naively copy $w_{j_{\text{dead}}} \leftarrow w_{j^*}$ and $b_{j_{\text{dead}}} \leftarrow b_{j^*}$ **without adjusting the biases**, then the softmax denominator gains an extra duplicate term $e^{z_{j^*}}$:
- The pair's **combined routing probability suddenly surges** from $\frac{e^{z_{j^*}}}{Z}$ to $\frac{2 e^{z_{j^*}}}{Z + e^{z_{j^*}}}$!
- Worse, in Top-$K_r$ ($K_r = 2$) routing, **both slots in Top-2 immediately become the two clones**, blocking the token's true 2nd expert $k_2$ and injecting huge downstream activation shocks! *(Note: even for $K_r=2$, let's check both Top-$K_r$ renormalization and probability mass!)*

---

## 3. Exact Proof of Softmax Mass Invariance (`b <- b - tau_r ln 2`)

Set the router bias of **both** children (`j^*` and `j_dead`) to:
$$
b_{j^*}^{\text{new}} = b_{j^*}^{\text{old}} - \tau_r \ln(2), \qquad b_{j_{\text{dead}}}^{\text{new}} = b_{j^*}^{\text{old}} - \tau_r \ln(2)
$$
(Equivalently, if the router uses linear bias *before* dividing by $\tau_r$, $(w^\top x + b) / \tau_r$, subtracting $\tau_r \ln 2$ shifts the post-temperature logit by exact $-\ln 2$. If the bias is added *after* dividing by $\tau_r$, subtract $\ln 2$ directly.)

Then for any input hidden state $x$ (with tiny symmetry-breaking $\xi_r = 0$):
$$
\exp\!\left(\frac{w_{j^*}^\top x + (b_{j^*} - \tau_r \ln 2)}{\tau_r}\right) = e^{z_{j^*}} \cdot e^{-\ln 2} = \frac{1}{2} e^{z_{j^*}}
$$
Therefore, summing over both sibling experts:
$$
\exp(z_{j^*}^{\text{new}}) + \exp(z_{j_{\text{dead}}}^{\text{new}}) = \frac{1}{2} e^{z_{j^*}} + \frac{1}{2} e^{z_{j^*}} = \boxed{e^{z_{j^*}}}
$$

### Two Immediate Consequences
1. **Unchanged Denominator & Other Experts**: The total softmax partition function $Z = \sum_{m=1}^{E} e^{z_m}$ stays invariant (ignoring negligible $e^{z_{\text{dead}}} \approx 0$ before split), so **all other $E-2$ experts keep their exact softmax weights**!
2. **Function Preservation Under Weighted Sum**: When both siblings compute identical FFN outputs $f_{\text{split}}(x) = f_{j^*}(x)$ and their raw softmax weights add up to $\frac{1}{2}p_{j^*} + \frac{1}{2}p_{j^*} = p_{j^*}$, total contribution is identical!
   - *Important Top-$K_r$ Implementation Subtlety*: Notice that in hard Top-$K_r$ with $K_r=2$, two identical logits tying at rank 1 would occupy both Top-2 slots if $\xi_r = 0$. Using **tie-breaking perturbation** $\pm \delta W_\perp$ + either **(a)** a small offset gap $\pm \epsilon_{\text{tie}}$ or **(b)** renormalized Top-$K_r$ with split-pair FFN output preservation ensures smooth divergence in 1–2 steps!

---

## 4. Orthogonal Symmetry-Breaking Perturbation

If both child FFNs and router rows remained bit-identical ($\xi_r = 0, \delta W = 0$), deterministic backprop produces identical gradients forever—wasting one slot!
We add an **orthogonal perturbation** perpendicular to the existing weight vector direction so first-order norm magnitude is preserved:
1. Draw random Gaussian noise $u \sim \mathcal{N}(0, I)$.
2. Project out the donor weight direction:
   $$
   u_\perp = u - \frac{\langle u, W_{\text{donor}} \rangle}{\|W_{\text{donor}}\|_2^2 + \epsilon} W_{\text{donor}}, \qquad \hat{u}_\perp = \delta \cdot \|W_{\text{donor}}\|_2 \cdot \frac{u_\perp}{\|u_\perp\|_2 + \epsilon}
   $$
3. Symmetrically perturb both siblings:
   $$
   W_{j^*}^{\text{new}} = W_{\text{donor}} + \hat{u}_\perp, \qquad W_{j_{\text{dead}}}^{\text{new}} = W_{\text{donor}} - \hat{u}_\perp
   $$
   Typical scale $\delta \in [10^{-3}, 5 \cdot 10^{-3}]$ suffices to trigger rapid domain specialization within ~20 optimizer steps without disrupting forward activations.

---

## 5. Bookkeeping: EMA States, Cooldowns & AdamW Moments

Splitting weights alone is incomplete if optimizer buffers stay stale:

| Buffer / State | Action on `(donor j*, dead j_dead)` Split | Why Necessary |
| :--- | :--- | :--- |
| **`ema_load` ($N$)** | $N_{j^*} \leftarrow \frac{1}{2} N_{j^*},\;\; N_{j_{\text{dead}}} \leftarrow \frac{1}{2} N_{j^*}$ | Mirrors total expected traffic share across siblings so neither triggers instant re-split/re-donation. |
| **`ema_energy` ($S$)** | $S_{j^*} \leftarrow \frac{1}{2} S_{j^*},\;\; S_{j_{\text{dead}}} \leftarrow \frac{1}{2} S_{j^*}$ | Keeps routing energy calibrated. |
| **`cooldown` counter** | Both children set to `cooldown_steps` (e.g., $2 M_{\text{split}}$ steps) | Prevents freshly split experts from being flagged dead while AdamW diverges them onto sub-distributions. |
| **AdamW `exp_avg_sq` ($v$)** | **Copy directly** from donor to `j_dead` (for both FFN weights and router row) | Keeps per-parameter adaptive learning-rate preconditioner $1/\sqrt{v}$ continuous (no sudden LR spike on dead slot!). |
| **AdamW `exp_avg` ($m$)** | Copy $m$ (or halve $0.5 \cdot m$ on router row) for smooth momentum | Preserves warm momentum without directional shock. |

---

## 6. Sources & Videos
- **Net2Net**: Chen, Goodfellow, & Shlens (ICLR 2016), *Net2Net: Accelerating Learning via Knowledge Transfer* ([arXiv:1511.05641](https://arxiv.org/abs/1511.05641))
- **Function-Preserving Splitting of Neural Nets**: Liu et al. (ICML 2019), *Splitting Steepest Descent for Growing Neural Architectures* ([arXiv:1910.02366](https://arxiv.org/abs/1910.02366))
- **Video Search**: [Net2Net Function-Preserving Transformations Explained (YouTube)](https://www.youtube.com/results?search_query=Net2Net+function+preserving+neural+network+splitting)
