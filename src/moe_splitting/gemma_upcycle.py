"""
Config-faithful TinyGemma model and Sparse Upcycling utilities for Gemma 2 / Gemma 3 -> MoE.
Runs 100% offline with synthetic weights; never downloads remote HuggingFace weights.
"""

import contextlib
import copy
import math
from typing import Iterator, List, Optional, Tuple, cast, overload
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMeansMoEConfig, GemmaArchitectureSpec
from .experts import GeGLUMoEExperts
from .router import EMeansMoERouter, RouterOutput
from .splitting import DynamicExpertSplitter, SplitEvent, UnmergeEvent

UPCYCLE_STRATEGIES = ("noisy_clone", "sliced_granular", "sliced_parity", "progressive_split")


class GemmaRMSNorm(nn.Module):
    """Exact Gemma unit-offset RMSNorm: weight is zero-initialized and added to 1.0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(x.dtype)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal RoPE supporting arbitrary head_dim on (B, n_heads, L, head_dim)."""
    seq_len, hd = q.shape[-2], q.shape[-1]
    half = hd // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / max(half, 1)))
    pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (L, half)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0).to(q.dtype)

    def rot(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin], dim=-1)

    return rot(q), rot(k)


class TinyGemmaAttention(nn.Module):
    """Grouped-Query Attention matching Gemma 2 / 3 projections and optional soft-capping."""

    def __init__(self, spec: GemmaArchitectureSpec):
        super().__init__()
        self.n_q = spec.num_attention_heads
        self.n_kv = spec.num_key_value_heads
        self.hd = spec.head_dim
        self.theta = spec.rope_theta
        self.softcap = spec.attn_logit_softcapping

        self.q_proj = nn.Linear(spec.hidden_size, self.n_q * self.hd, bias=False)
        self.k_proj = nn.Linear(spec.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(spec.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.hd, spec.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_q, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)

        q, k = _apply_rope(q, k, self.theta)

        rep = self.n_q // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        if self.softcap and self.softcap > 0.0:
            scores = self.softcap * torch.tanh(scores / self.softcap)

        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        attn = F.softmax((scores + causal_mask).float(), dim=-1).to(x.dtype)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.n_q * self.hd)
        return self.o_proj(ctx)


class GemmaDenseGeGLUMLP(nn.Module):
    """Exact Gemma GeGLU MLP matching HF attribute names (gate_proj, up_proj, down_proj)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDenseDecoderLayer(nn.Module):
    """Config-faithful Gemma 2/3 Decoder Layer (pre/post attention & feedforward RMSNorm)."""

    def __init__(self, spec: GemmaArchitectureSpec):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.self_attn = TinyGemmaAttention(spec)
        self.post_attention_layernorm = GemmaRMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.mlp = GemmaDenseGeGLUMLP(spec.hidden_size, spec.intermediate_size)
        self.post_feedforward_layernorm = GemmaRMSNorm(spec.hidden_size, spec.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        ffn_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        return h + ffn_out


class GemmaMoEMLPBlock(nn.Module):
    """Drop-in replacement for a Gemma GeGLU MLP holding router + GeGLU expert bank.

    Routing statistics of the last forward are kept for `step_moe_blocks` only when the
    block is in training mode and `record_stats` is True (see `frozen_moe_stats`), so an
    evaluation or measurement forward never overwrites the statistics of a training batch.
    The router stays fp32 whatever `dtype` the experts use.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        cfg: EMeansMoEConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.router = EMeansMoERouter(hidden_size, cfg).to(device=device)
        self.experts = GeGLUMoEExperts(cfg.num_experts, hidden_size, expert_intermediate_size, device=device, dtype=dtype)
        self.record_stats = True
        self.last_router_output: Optional[RouterOutput] = None
        self.last_batch_energy: Optional[torch.Tensor] = None
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        r_out = self.router(x_flat)
        out_flat, batch_energy = self.experts(x_flat, r_out.topk_idx, r_out.topk_weights)
        self.last_aux_loss = r_out.aux_loss + r_out.z_loss
        if self.training and self.record_stats:
            self.last_router_output = r_out
            self.last_batch_energy = batch_energy
        return out_flat.view(shape)


def iter_moe_blocks(model: nn.Module) -> List[Tuple[str, GemmaMoEMLPBlock]]:
    """All MoE blocks of a model (TinyGemma or HF Gemma 2/3 after `upcycle_gemma_model`)."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, GemmaMoEMLPBlock)]


@contextlib.contextmanager
def frozen_moe_stats(model: nn.Module) -> Iterator[None]:
    """Run forwards (e.g. loss measurements) without touching the stored routing statistics."""
    blocks = [b for _, b in iter_moe_blocks(model)]
    prev = [b.record_stats for b in blocks]
    for b in blocks:
        b.record_stats = False
    try:
        yield
    finally:
        for b, p in zip(blocks, prev):
            b.record_stats = p


@torch.no_grad()
def step_moe_blocks(
    model: nn.Module,
    splitter: DynamicExpertSplitter,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    total_steps: int,
    sync_distributed: bool = True,
) -> Tuple[List[SplitEvent], List[UnmergeEvent]]:
    """Per MoE block: update EMA statistics, un-merge separable sibling pairs, split if due.

    Call once per optimizer step, after `optimizer.step()`. The stored statistics are
    consumed, so a second call without a new training forward is a no-op.
    """
    split_events: List[SplitEvent] = []
    unmerge_events: List[UnmergeEvent] = []
    for name, blk in iter_moe_blocks(model):
        r_out = blk.last_router_output
        if r_out is None or blk.last_batch_energy is None:
            continue
        blk.router.update_ema(
            r_out.batch_load,
            blk.last_batch_energy,
            step=step,
            total_steps=total_steps,
            sync_distributed=sync_distributed,
            batch_sum=r_out.batch_sum,
            batch_sqnorm=r_out.batch_sqnorm,
            pair_disagree=r_out.pair_disagree,
        )
        blk.last_router_output = None
        blk.last_batch_energy = None
        unmerge_events.extend(splitter.maybe_unmerge_layer(blk.router, step=step, layer_name=name))
        split_events.extend(
            splitter.maybe_split_layer(blk.router, blk.experts, optimizer=optimizer, step=step, layer_name=name)
        )
    return split_events, unmerge_events


class GemmaMoEDecoderLayer(nn.Module):
    """Upcycled Gemma Decoder Layer replacing dense GeGLU MLP with GemmaMoEMLPBlock."""

    input_layernorm: nn.Module
    self_attn: nn.Module
    post_attention_layernorm: nn.Module
    pre_feedforward_layernorm: nn.Module
    mlp: GemmaMoEMLPBlock
    post_feedforward_layernorm: nn.Module

    def __init__(self, dense_layer: nn.Module, moe_mlp: GemmaMoEMLPBlock):
        super().__init__()
        self.input_layernorm = cast(nn.Module, dense_layer.input_layernorm)
        self.self_attn = cast(nn.Module, dense_layer.self_attn)
        self.post_attention_layernorm = cast(nn.Module, dense_layer.post_attention_layernorm)
        self.pre_feedforward_layernorm = cast(nn.Module, dense_layer.pre_feedforward_layernorm)
        self.mlp = moe_mlp
        self.post_feedforward_layernorm = cast(nn.Module, dense_layer.post_feedforward_layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        ffn_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        return h + ffn_out


class TinyGemmaForCausalLM(nn.Module):
    """Lightweight synthetic Gemma CausalLM mirroring HF Gemma structure cleanly offline."""

    def __init__(self, spec: GemmaArchitectureSpec):
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.layers = nn.ModuleList([GemmaDenseDecoderLayer(spec) for _ in range(spec.num_hidden_layers)])
        self.norm = GemmaRMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Gemma weight tying
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        """HF Gemma init: N(0, initializer_range) for Linear and Embedding, zero biases; the
        unit-offset RMSNorm weights stay at zero."""
        std = self.spec.initializer_range
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.normal_(0.0, std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Gemma scales embeddings by sqrt(hidden_size)
        x = self.embed_tokens(input_ids) * math.sqrt(self.spec.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.spec.final_logit_softcapping and self.spec.final_logit_softcapping > 0.0:
            cap = self.spec.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits


class TinyGemmaMoEForCausalLM(nn.Module):
    """MoE-upcycled Gemma CausalLM with unified EMA step updates & split hooks."""

    splitter: DynamicExpertSplitter

    def __init__(self, dense_model: TinyGemmaForCausalLM, moe_cfg: EMeansMoEConfig):
        super().__init__()
        self.spec = dense_model.spec
        self.moe_cfg = moe_cfg
        self.embed_tokens = dense_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                GemmaMoEDecoderLayer(
                    layer,
                    upcycle_gemma_mlp_to_moe(cast(GemmaDenseDecoderLayer, layer).mlp, moe_cfg, layer_idx=idx),
                )
                for idx, layer in enumerate(dense_model.layers)
            ]
        )
        self.norm = dense_model.norm
        self.lm_head = dense_model.lm_head
        self.splitter = DynamicExpertSplitter()
        self.last_unmerge_events: List[UnmergeEvent] = []
        self.unmerge_log: List[UnmergeEvent] = []

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed_tokens(input_ids) * math.sqrt(self.spec.hidden_size)
        total_aux = x.new_zeros(())
        for layer in self.iter_moe_layers():
            x = layer(x)
            if layer.mlp.last_aux_loss is not None:
                total_aux = total_aux + layer.mlp.last_aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.spec.final_logit_softcapping and self.spec.final_logit_softcapping > 0.0:
            cap = self.spec.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits, total_aux

    def set_splitter(self, splitter: DynamicExpertSplitter) -> None:
        """Replace the dynamic expert splitter cleanly without triggering Module.__setattr__ warnings."""
        object.__setattr__(self, "splitter", splitter)

    def iter_moe_layers(self) -> List[GemmaMoEDecoderLayer]:
        """Return concretely typed GemmaMoEDecoderLayer list for static type checkers."""
        return [cast(GemmaMoEDecoderLayer, layer) for layer in self.layers]

    @torch.no_grad()
    def step_ema_and_split(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        total_steps: int,
    ) -> List[SplitEvent]:
        """Update per-layer EMA stats, un-merge separable sibling pairs, split when due.

        Returns the split events; un-merge events go to `last_unmerge_events` / `unmerge_log`.
        """
        split_events, unmerge_events = step_moe_blocks(self, self.splitter, optimizer, step, total_steps)
        self.last_unmerge_events = unmerge_events
        self.unmerge_log.extend(unmerge_events)
        return split_events


@torch.no_grad()
def upcycle_gemma_mlp_to_moe(dense_mlp: nn.Module, cfg: EMeansMoEConfig, layer_idx: int = 0) -> GemmaMoEMLPBlock:
    """
    Convert a dense Gemma GeGLU MLP (having gate_proj, up_proj, down_proj) into a
    GemmaMoEMLPBlock without requiring remote downloads.

    Strategies (step-0 parity = the MoE reproduces the dense MLP before any training):
      - noisy_clone: full D_ff cloned to E experts (+ upcycle_init_noise). Parity is exact at
        noise=0 when K_r >= 2 with renormalised weights; at K_r = 1 the output is p_top1 * dense.
      - sliced_parity: D_ff sliced into K_r pieces (expert j holds slice j % K_r, down_proj * K_r),
        router rows zeroed, experts 0..K_r-1 (one per slice) tied at bias 0 and all others at
        -parity_bias_gap, so every token selects exactly one expert per slice: exact parity at
        noise=0 with dense-matched FLOPs. Requires K_r >= 2 and renorm_topk_weights.
      - sliced_granular: same slicing with a random router. NOT parity-preserving: a token whose
        Top-K holds two experts with the same slice drops part of the FFN (about half of the
        tokens at E=16, K=2). Kept only as a comparison point.
      - progressive_split: only K0 = progressive_active_experts experts are active; the others
        start dormant (bias = dormant_init_bias, EMA load 0) for DynamicExpertSplitter to revive.
    """
    if cfg.upcycle_strategy not in UPCYCLE_STRATEGIES:
        raise ValueError(f"upcycle_strategy={cfg.upcycle_strategy!r} not in {UPCYCLE_STRATEGIES}")
    sliced = cfg.upcycle_strategy in ("sliced_granular", "sliced_parity")
    if cfg.upcycle_strategy == "sliced_parity" and (cfg.top_k < 2 or not cfg.renorm_topk_weights):
        raise ValueError("sliced_parity needs top_k >= 2 and renorm_topk_weights=True (Top-1 weight is p_top1 < 1)")

    gate_proj = cast(nn.Linear, dense_mlp.gate_proj)
    up_proj = cast(nn.Linear, dense_mlp.up_proj)
    down_proj = cast(nn.Linear, dense_mlp.down_proj)

    hidden_size = gate_proj.in_features
    dense_ff = gate_proj.out_features

    w_gate = gate_proj.weight.data
    w_up = up_proj.weight.data
    w_down = down_proj.weight.data

    if sliced:
        if dense_ff % cfg.top_k != 0:
            raise ValueError(f"intermediate_size={dense_ff} must be divisible by top_k={cfg.top_k}")
        exp_ff = dense_ff // cfg.top_k
    else:
        exp_ff = dense_ff

    # Experts on the dense weights' device and dtype; the router stays fp32 on that device.
    block = GemmaMoEMLPBlock(hidden_size, exp_ff, cfg, device=w_gate.device, dtype=w_gate.dtype)
    gen = torch.Generator(device=w_gate.device).manual_seed(1337 + layer_idx * 97)

    for j in range(cfg.num_experts):
        if sliced:
            slice_id = j % cfg.top_k
            s0, s1 = slice_id * exp_ff, (slice_id + 1) * exp_ff
            # down_proj * K_r: the renormalised Top-K weights are 1/K_r per selected expert
            g_src = w_gate[s0:s1]
            u_src = w_up[s0:s1]
            d_src = w_down[:, s0:s1] * float(cfg.top_k)
        else:
            g_src, u_src, d_src = w_gate, w_up, w_down

        for dst, src in (
            (block.experts.gate_proj[j], g_src),
            (block.experts.up_proj[j], u_src),
            (block.experts.down_proj[j], d_src),
        ):
            dst.copy_(src)
            if cfg.upcycle_init_noise > 0.0:
                noise = torch.randn(src.shape, device=src.device, dtype=torch.float32, generator=gen)
                dst.add_((cfg.upcycle_init_noise * noise).to(dst.dtype))

    if cfg.upcycle_strategy == "sliced_parity":
        nn.init.zeros_(block.router.gate.weight)
        nn.init.zeros_(block.router.gate.bias)
        block.router.gate.bias[cfg.top_k:].fill_(-abs(cfg.parity_bias_gap))
        block.router.ema_load.zero_()
        block.router.ema_load[: cfg.top_k].fill_(1.0)

    if cfg.upcycle_strategy == "progressive_split":
        nn.init.zeros_(block.router.gate.bias)
        active_k0 = min(cfg.progressive_active_experts, cfg.num_experts)
        block.router.gate.bias[active_k0:].fill_(cfg.dormant_init_bias)
        # Seed initial EMA load cleanly to reflect dormant slots
        fair_active = cfg.top_k / max(active_k0, 1)
        block.router.ema_load[:active_k0].fill_(fair_active)
        block.router.ema_load[active_k0:].fill_(0.0)

    return block


@overload
def upcycle_gemma_model(dense_model: TinyGemmaForCausalLM, cfg: EMeansMoEConfig) -> TinyGemmaMoEForCausalLM: ...


@overload
def upcycle_gemma_model(dense_model: nn.Module, cfg: EMeansMoEConfig) -> nn.Module: ...


def upcycle_gemma_model(dense_model: nn.Module, cfg: EMeansMoEConfig) -> nn.Module:
    """
    Convert either TinyGemmaForCausalLM or a locally loaded HuggingFace Gemma model into an
    auxiliary-loss-free MoE model (on a deep copy; never triggers downloads).

    For HF models only `layer.mlp` is replaced, so the HF decoder layer (attention masks,
    caches, sliding windows) is untouched. Drive training with `step_moe_blocks(model, ...)`.
    """
    if isinstance(dense_model, TinyGemmaForCausalLM):
        return TinyGemmaMoEForCausalLM(copy.deepcopy(dense_model), cfg)

    # Generic HF Gemma2ForCausalLM / Gemma3ForCausalLM support via attribute path
    cloned = copy.deepcopy(dense_model)
    layers = cast(nn.ModuleList, getattr(getattr(cloned, "model", cloned), "layers"))
    for idx, layer in enumerate(layers):
        layer.mlp = upcycle_gemma_mlp_to_moe(cast(nn.Module, layer.mlp), cfg, layer_idx=idx)
    return cloned
