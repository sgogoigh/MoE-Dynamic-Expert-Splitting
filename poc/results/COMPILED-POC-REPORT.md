# Compiled Proof-of-Concept Evidence Report

**Generated**: 2026-11-21 01:24:09  
**Completed Tiers**: PoC 1

---

## PoC 1: Exact Mathematical Invariance & Gemma Systems Audit

### Softmax Mass Conservation Verification (`b <- b - tau_r ln 2`)
| $\tau_r$ | Max Pair Prob Diff | Max Other Prob Diff | Max MoE Output Diff |
| :---: | :---: | :---: | :---: |
| 0.5 | `2.220e-16` | `2.776e-17` | `8.882e-16` |
| 1.0 | `3.331e-16` | `1.110e-16` | `8.882e-16` |
| 2.5 | `5.551e-17` | `5.551e-17` | `2.220e-16` |

### Multi-Domain Synthetic 400-Step Comparison (`E=16, K_r=2`)
| Method | Splits | Dead Experts | Max/Mean Load | Norm Entropy | Gini |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `unbalanced` | 0 | 10 | 6.0641x | 0.56 | 0.7669 |
| `aux_loss` | 0 | 2 | 2.4105x | 0.895 | 0.4111 |
| `deepseek_bias` | 0 | 1 | 1.6661x | 0.9578 | 0.251 |
| `emeans` | 32 | 3 | 3.9459x | 0.8054 | 0.5329 |

### Gemma Upcycling Resource Footprint
| Preset | `E / K_r` | Strategy | Dense (M) | MoE Total (M) | MoE Active (M) | Unsharded Train (GB) | EMA Sync (KB/step) |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `gemma-3-1b` | `16/2` | `sliced_granular` | 999.87 | 5347.94 | 1000.35 | 79.69 | 3.25 |
| `gemma-3-1b` | `16/2` | `noisy_clone` | 999.87 | 10316.62 | 1621.44 | 153.73 | 3.25 |
| `gemma-2-2b` | `16/2` | `sliced_granular` | 2614.34 | 14208.88 | 2615.3 | 211.73 | 3.25 |
| `gemma-2-2b` | `32/2` | `sliced_granular` | 2614.34 | 27459.65 | 2616.26 | 409.18 | 6.5 |
| `gemma-2-9b` | `64/4` | `sliced_granular` | 9241.71 | 106359.97 | 9251.34 | 1584.89 | 21.0 |

---

## PoC 2: PyTorch Split-Step Continuity *(Pending Execution)*
Run `python poc/poc_02_tiny_gemma_continuity.py` on a GPU or Colab T4 to populate `poc/results/poc_02_results.json`.

---

## PoC 3: Real Multi-Domain Proxy & Mini Scaling Law *(Pending Execution)*
Run `python poc/poc_03_proxy_collapse_and_scaling.py --run-all` on 1× 4090/A100 to populate `poc/results/poc_03_results.json`.

---

## PoC 4: Frozen-Backbone Gemma-3-1B Partial Upcycling *(Pending Execution)*
Run `python poc/poc_04_gemma3_1b_frozen_upcycle.py --model-path <local_gemma_dir>` on 1× A100/4090 to populate `poc/results/poc_04_results.json`.
