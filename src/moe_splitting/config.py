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
    # HF Gemma initialises every Linear and Embedding with N(0, initializer_range). PyTorch's
    # default N(0, 1) embedding, tied to the LM head and scaled by sqrt(hidden), starts the logits
    # near the softcap and the loss far above log(vocab).
    initializer_range: float = 0.02


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
    # FFN perturbation of the two siblings. 0.0 keeps the split exactly function-preserving;
    # with sibling routing the siblings see disjoint tokens, which already breaks symmetry.
    orthogonal_eps: float = 0.0
    # Router-row perturbation +/-r: defines the hyperplane that partitions the donor's tokens.
    # Its effect on the pair's softmax mass is second order (cosh(r.(x-mu)/tau) - 1).
    router_noise_eps: float = 1e-3
    cooldown_steps: int = 400
    halve_router_momentum: bool = True

    # Top-K function preservation (docs/basic-terms/04-Function-Preserving-Splitting-and-Adam-State.md)
    # sibling_routing: after a split, donor + clone form one routing group (Top-K over group mass;
    # the token goes to the higher-probability sibling with the group's weight) until separable.
    sibling_routing: bool = True
    # Place the sibling hyperplane through the donor's EMA token centroid S_j / N_j (e-means split).
    split_through_centroid: bool = True
    # Un-merge a pair at once when the EMA routing weight that un-merging would move (router.py,
    # `pair_disagree`) is below unmerge_tol, after at least unmerge_min_steps.
    unmerge_tol: float = 0.01
    unmerge_min_steps: int = 50
    # Force the un-merge after this many merged steps even if not separable (0 = never force).
    # Measured on PoC 2: with hard within-pair routing the siblings do not separate by themselves
    # (the router gets no "which sibling" gradient), so most un-merges are forced. A forced
    # un-merge is ramped: routing moves linearly from grouped to plain Top-K over
    # unmerge_ramp_steps, so each step changes the function by 1/unmerge_ramp_steps of the total
    # (0 = switch at once, which reproduces the old split's loss spike at un-merge time).
    merge_max_steps: int = 2000
    unmerge_ramp_steps: int = 100

    # Baseline comparator knobs
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0
    deepseek_bias_lr: float = 1e-3

    # Gemma Upcycling options
    # "noisy_clone", "sliced_granular", "sliced_parity", "progressive_split"
    upcycle_strategy: str = "noisy_clone"
    upcycle_init_noise: float = 1e-3
    progressive_active_experts: int = 4
    dormant_init_bias: float = -5.0
    # sliced_parity: experts 0..K_r-1 (one per slice) start tied at bias 0 with zero router rows,
    # all others at -parity_bias_gap, so step 0 reproduces the dense MLP exactly. The gap must be
    # large against optimizer drift of the active biases (one AdamW step moves a bias by ~lr):
    # with a gap of 1e-3 a different-slice expert enters the Top-K after one step at lr=1e-3 and
    # parity is lost. 1.0 logit keeps dormant experts reachable for aux/bias balancers (mass e^-1).
    parity_bias_gap: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
