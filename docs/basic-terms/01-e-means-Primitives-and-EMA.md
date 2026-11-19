# 01. `e-means` Primitives & Exponential Moving Averages (EMA)

> **Origin**: Bessa et al. (Meta FAIR, Nov 2026), *"e-means: outperforming k-means (in one-tenth the epochs)"* ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)).

---

## 1. Why Understanding `e-means` First Matters for MoE

In classical clustering, **`k-means` (Lloyd's algorithm)** suffers from two pathologies at billion scale:
1. Full-dataset passes require dozens of expensive epochs before converging.
2. Clusters that lose points become **empty ("dead") centroids** wasting model capacity unless heuristically rescued.

**`e-means`** solves both cleanly with **three core primitives**, each of which translates 1-to-1 to **Sparse Mixture-of-Experts (MoE)** expert load balancing without any auxiliary loss:

| Concept | In `e-means` (Vector Clustering) | In MoE Dynamic Expert Splitting |
| :--- | :--- | :--- |
| **Partition Unit** | Cluster centroid $c_j \in \mathbb{R}^d$ | Expert FFN $f_j(x)$ + router row $w_j$ |
| **Tracking Memory** | Scalar count $N_j$ & vector sum $S_j$ | Token fraction $N_j$ & routing energy $S_j$ |
| **Schedule** | Cosine decay $\alpha(t)$ down to $0.1\alpha_0$ | Cosine decay down to **non-stationary floor $\alpha_{\min}$** |
| **Revival Trigger** | $N_j < \tau \bar{N}$ ($\tau = 0.1$) every 5 batches | $N_j < \tau \cdot (K_r / E)$ every $M_{\text{split}}$ steps |
| **Donor Action** | Split high-count donor $j^*$ via $\odot(1 \pm \delta)$ | Clone donor FFN + $-\tau_r \ln(2)$ router bias shift |

---

## 2. Sufficient Statistics & Exponential Moving Average (EMA)

Instead of directly applying noisy SGD to centroids, `e-means` tracks smoothed **first-order sufficient statistics**:

$$
N_j^{(t)} \leftarrow \gamma_t N_j^{(t-1)} + (1 - \gamma_t) n_{t,j}, \qquad S_j^{(t)} \leftarrow \gamma_t S_j^{(t-1)} + (1 - \gamma_t) s_{t,j}, \qquad c_j = \frac{S_j^{(t)}}{N_j^{(t)}}
$$

where:
- $\alpha_t = 1 - \gamma_t \in (0, 1]$ is the **step size** (learning rate of the sufficient statistic),
- $\gamma_t = 1 - \alpha_t \in [0, 1)$ is the **decay factor** (forgetting rate of past batches),
- Ratio readout $S_j / N_j$ keeps centroids strictly inside the **convex hull** of seen data, preventing gradient-optimizer divergence even at aggressive $\alpha_t$.

### Effective Memory Window Intuition
With constant decay $\gamma = 1 - \alpha$, weights decay geometrically as $\gamma^k = (1-\alpha)^k \approx e^{-k \alpha}$. Thus the **effective half-life** in mini-batches is roughly:
$$
T_{\text{eff}} \approx \frac{1}{\alpha_t} \text{ steps}
$$
- **Too small $\alpha_t$**: EMA reacts too slowly to shifting token distributions during LLM pretraining (**stale routing stats**).
- **Too large $\alpha_t$**: Single noisy mini-batches cause false dead-expert alarms (**ping-pong splitting**).

---

## 3. Cosine Annealing vs. Non-Stationary Tracking Floor ($\alpha_{\min}$)

For a **static** clustering dataset $X$, `e-means` anneals $\alpha_t$ cleanly toward a small fraction of $\alpha_0$ ($\varepsilon = 0.1$):
$$
\alpha_{\text{static}}(t) = \alpha_0 \left(\varepsilon + \frac{1-\varepsilon}{2}\left(1 + \cos\frac{\pi t}{M}\right)\right)
$$

In **LLM MoE training / upcycling**, token hidden representations $h(x)$ are **non-stationary**: attention and MLP weights continuously evolve, drifting representations across domains and training stages. Therefore $\alpha(t)$ must **never** collapse toward zero; instead it anneals to a strictly positive **tracking floor $\alpha_{\min}$**:
$$
\alpha(t) = \alpha_{\min} + (\alpha_0 - \alpha_{\min}) \cdot \frac{1}{2}\left(1 + \cos\frac{\pi t}{M}\right)
$$

---

## 4. Dead-Cluster Detection & Donor Splitting

Every fixed interval, `e-means` checks whether any unit has fallen below a threshold fraction $\tau$ (typically $\tau = 0.1$) of the fair mean:
$$
\text{Dead Condition: } N_j < \tau \cdot \bar{N}
$$
Rather than replacing a dead centroid with a noisy random vector, `e-means` **selects a high-capacity donor** $j^*$ proportional to $N_{j^*}$ (or routing energy $S_{j^*}$), **splits** its representation into two sibling units sharing the workload, and **halves** the EMA mass ($N_{j^*}, N_{\text{dead}} \leftarrow N_{j^*}/2$) so neither child triggers an immediate follow-up split.

---

## 5. Sources & Videos
- **Base Paper**: Bessa et al. (2026), *e-means: outperforming k-means (in one-tenth the epochs)* — [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)
- **Video Search**: [Exponential Moving Average in Machine Learning Explained (YouTube)](https://www.youtube.com/results?search_query=Exponential+Moving+Average+VQ-VAE+k-means)
