"""Configuration dataclasses and Gemma architectural presets."""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class GemmaArchitectureSpec:
    """Architectural specification for Gemma dense models and MoE upcycling targets."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"


GEMMA_PRESETS: Dict[str, GemmaArchitectureSpec] = {
    "tiny-gemma-debug": GemmaArchitectureSpec(
        name="tiny-gemma-debug",
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=2048,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
    "gemma-3-1b": GemmaArchitectureSpec(
        name="gemma-3-1b",
        hidden_size=1152,
        intermediate_size=6912,
        num_hidden_layers=26,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        vocab_size=262144,
        rope_theta=1000000.0,
        attn_logit_softcapping=0.0,
        final_logit_softcapping=0.0,
    ),
    "gemma-2-2b": GemmaArchitectureSpec(
        name="gemma-2-2b",
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        vocab_size=256000,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
    "gemma-2-9b": GemmaArchitectureSpec(
        name="gemma-2-9b",
        hidden_size=3584,
        intermediate_size=14336,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        vocab_size=256000,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
}


@dataclass
class EMeansMoEConfig:
    """Hyperparameters for auxiliary-loss-free EMA routing & dynamic expert splitting."""

    num_experts: int = 16
    top_k: int = 2
    router_mode: str = "emeans"  # "emeans", "unbalanced", "aux_loss", "deepseek_bias"
    tau_r: float = 1.0
    renorm_topk_weights: bool = True

    # EMA tracking schedule (cosine decay -> non-stationary tracking floor alpha_min)
    alpha_0: float = 0.05
    alpha_min: float = 0.005

    # Dead-expert detection & donor splitting
    split_interval: int = 200
    split_tau: float = 0.10
    donor_min_mult: float = 1.25
    donor_weight_by: str = "energy"  # "energy" (S_j) or "load" (N_j)
    max_splits_per_step: int = 4
    orthogonal_eps: float = 1e-3
    router_noise_eps: float = 1e-4
    cooldown_steps: int = 400
    halve_router_momentum: bool = True

    # Baseline comparator knobs
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0
    deepseek_bias_lr: float = 1e-3

    # Gemma Upcycling options
    upcycle_strategy: str = "noisy_clone"  # "noisy_clone", "sliced_granular", "progressive_split"
    upcycle_init_noise: float = 1e-3
    progressive_active_experts: int = 4
    dormant_init_bias: float = -5.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
