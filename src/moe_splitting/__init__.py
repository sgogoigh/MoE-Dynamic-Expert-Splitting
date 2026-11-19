"""MoE Dynamic Expert Splitting — Auxiliary-loss-free load balancing via e-means primitives."""

from .config import EMeansMoEConfig, GEMMA_PRESETS, GemmaArchitectureSpec
from .metrics import compute_routing_health
from .scaling_law import PowerLawFit, ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw

__all__ = [
    "EMeansMoEConfig",
    "GemmaArchitectureSpec",
    "GEMMA_PRESETS",
    "compute_routing_health",
    "PowerLawFit",
    "ScalingObservation",
    "fit_alpha_min_powerlaw",
    "evaluate_loso",
]

try:
    import torch  # noqa: F401
    from .experts import GeGLUMoEExperts
    from .gemma_upcycle import (
        GemmaMoEDecoderLayer,
        TinyGemmaForCausalLM,
        TinyGemmaMoEForCausalLM,
        upcycle_gemma_model,
    )
    from .router import EMeansMoERouter, RouterOutput
    from .splitting import DynamicExpertSplitter, SplitEvent

    __all__ += [
        "GeGLUMoEExperts",
        "EMeansMoERouter",
        "RouterOutput",
        "DynamicExpertSplitter",
        "SplitEvent",
        "GemmaMoEDecoderLayer",
        "TinyGemmaForCausalLM",
        "TinyGemmaMoEForCausalLM",
        "upcycle_gemma_model",
    ]
except ImportError:
    # Offline pure-NumPy mode supported for architectural exploration and math checks
    pass
