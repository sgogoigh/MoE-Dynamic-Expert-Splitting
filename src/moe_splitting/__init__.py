"""MoE Dynamic Expert Splitting — Auxiliary-loss-free load balancing via e-means primitives."""

from .config import EMeansMoEConfig, GEMMA_PRESETS, GemmaArchitectureSpec
from .poc_logging import tee_poc_logs

__all__ = [
    "EMeansMoEConfig",
    "GemmaArchitectureSpec",
    "GEMMA_PRESETS",
    "tee_poc_logs",
]

try:
    import numpy  # noqa: F401
    from .metrics import compute_routing_health
    from .scaling_law import PowerLawFit, ScalingObservation, evaluate_loso, fit_alpha_min_powerlaw

    __all__ += [
        "compute_routing_health",
        "PowerLawFit",
        "ScalingObservation",
        "fit_alpha_min_powerlaw",
        "evaluate_loso",
    ]
except ImportError:
    # Allows stdlib-only scripts (like compile_poc_report.py) to import config/poc_logging cleanly
    pass

try:
    import torch  # noqa: F401
    from .experts import GeGLUMoEExperts
    from .gemma_upcycle import (
        GemmaMoEDecoderLayer,
        GemmaMoEMLPBlock,
        TinyGemmaForCausalLM,
        TinyGemmaMoEForCausalLM,
        frozen_moe_stats,
        iter_moe_blocks,
        step_moe_blocks,
        upcycle_gemma_model,
    )
    from .router import EMeansMoERouter, RouterOutput
    from .splitting import DynamicExpertSplitter, SplitEvent, UnmergeEvent

    __all__ += [
        "GeGLUMoEExperts",
        "EMeansMoERouter",
        "RouterOutput",
        "DynamicExpertSplitter",
        "SplitEvent",
        "UnmergeEvent",
        "GemmaMoEDecoderLayer",
        "GemmaMoEMLPBlock",
        "TinyGemmaForCausalLM",
        "TinyGemmaMoEForCausalLM",
        "frozen_moe_stats",
        "iter_moe_blocks",
        "step_moe_blocks",
        "upcycle_gemma_model",
    ]
except ImportError:
    # Offline pure-NumPy mode supported for architectural exploration and math checks
    pass
