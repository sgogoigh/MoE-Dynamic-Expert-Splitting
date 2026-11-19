"""
Config-faithful TinyGemma model and Sparse Upcycling utilities for Gemma 2 / Gemma 3 -> MoE.
Runs 100% offline with synthetic weights; never downloads remote HuggingFace weights.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMeansMoEConfig, GemmaArchitectureSpec
from .experts import GeGLUMoEExperts
from .router import EMeansMoERouter, RouterOutput
from .splitting import DynamicExpertSplitter, SplitEvent


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
    """Drop-in replacement for GemmaDenseGeGLUMLP holding router + GeGLU expert bank."""

    def __init__(self, hidden_size: int, expert_intermediate_size: int, cfg: EMeansMoEConfig):
        super().__init__()
        self.router = EMeansMoERouter(hidden_size, cfg)
        self.experts = GeGLUMoEExperts(cfg.num_experts, hidden_size, expert_intermediate_size)
        self.last_router_output: Optional[RouterOutput] = None
        self.last_batch_energy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        x_flat = x.reshape(B * L, d)
        r_out = self.router(x_flat)
        out_flat, batch_energy = self.experts(x_flat, r_out.topk_idx, r_out.topk_weights)
        self.last_router_output = r_out
        self.last_batch_energy = batch_energy
        return out_flat.view(B, L, d)


class GemmaMoEDecoderLayer(nn.Module):
    """Upcycled Gemma Decoder Layer replacing dense GeGLU MLP with GemmaMoEMLPBlock."""

    def __init__(self, dense_layer: nn.Module, moe_mlp: GemmaMoEMLPBlock):
        super().__init__()
        self.input_layernorm = dense_layer.input_layernorm
        self.self_attn = dense_layer.self_attn
        self.post_attention_layernorm = dense_layer.post_attention_layernorm
        self.pre_feedforward_layernorm = dense_layer.pre_feedforward_layernorm
        self.mlp = moe_mlp
        self.post_feedforward_layernorm = dense_layer.post_feedforward_layernorm

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

    def __init__(self, dense_model: TinyGemmaForCausalLM, moe_cfg: EMeansMoEConfig):
        super().__init__()
        self.spec = dense_model.spec
        self.moe_cfg = moe_cfg
        self.embed_tokens = dense_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                GemmaMoEDecoderLayer(layer, upcycle_gemma_mlp_to_moe(layer.mlp, moe_cfg, layer_idx=idx))
                for idx, layer in enumerate(dense_model.layers)
            ]
        )
        self.norm = dense_model.norm
        self.lm_head = dense_model.lm_head
        self.splitter = DynamicExpertSplitter()

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed_tokens(input_ids) * math.sqrt(self.spec.hidden_size)
        total_aux = x.new_zeros(())
        for layer in self.layers:
            x = layer(x)
            r_out = layer.mlp.last_router_output
            if r_out is not None:
                total_aux = total_aux + r_out.aux_loss + r_out.z_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.spec.final_logit_softcapping and self.spec.final_logit_softcapping > 0.0:
            cap = self.spec.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits, total_aux

    @torch.no_grad()
    def step_ema_and_split(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        total_steps: int,
    ) -> List[SplitEvent]:
        """Update per-layer EMA stats and trigger DynamicExpertSplitter when due."""
        all_events: List[SplitEvent] = []
        for idx, layer in enumerate(self.layers):
            moe_mlp = layer.mlp
            if moe_mlp.last_router_output is None or moe_mlp.last_batch_energy is None:
                continue
            moe_mlp.router.update_ema(
                moe_mlp.last_router_output.batch_load,
                moe_mlp.last_batch_energy,
                step=step,
                total_steps=total_steps,
                sync_distributed=True,
            )
            events = self.splitter.maybe_split_layer(
                moe_mlp.router,
                moe_mlp.experts,
                optimizer=optimizer,
                step=step,
                layer_name=f"layer_{idx}",
            )
            all_events.extend(events)
        return all_events


@torch.no_grad()
def upcycle_gemma_mlp_to_moe(dense_mlp: nn.Module, cfg: EMeansMoEConfig, layer_idx: int = 0) -> GemmaMoEMLPBlock:
    """
    Convert a dense Gemma GeGLU MLP (having gate_proj, up_proj, down_proj) into a
    GemmaMoEMLPBlock without requiring remote downloads.

    Strategies:
      - noisy_clone: Full D_ff cloned to E experts (+ small noise); exact dense parity at noise=0.
      - sliced_granular: Slice D_ff into E granular sub-experts of width D_ff / K_r
        (for FLOP parity when summing Top-K_r experts without weighted attenuation).
      - progressive_split: Only K0 initial experts get positive routing bias while E-K0 start
        dormant (bias = dormant_init_bias), letting DynamicExpertSplitter carve capacity online.
    """
    hidden_size = dense_mlp.gate_proj.in_features
    dense_ff = dense_mlp.gate_proj.out_features

    w_gate = dense_mlp.gate_proj.weight.data
    w_up = dense_mlp.up_proj.weight.data
    w_down = dense_mlp.down_proj.weight.data

    if cfg.upcycle_strategy == "sliced_granular":
        if dense_ff % cfg.top_k != 0:
            raise ValueError(f"intermediate_size={dense_ff} must be divisible by top_k={cfg.top_k}")
        exp_ff = dense_ff // cfg.top_k
    else:
        exp_ff = dense_ff

    block = GemmaMoEMLPBlock(hidden_size, exp_ff, cfg)
    gen = torch.Generator(device=w_gate.device).manual_seed(1337 + layer_idx * 97)

    for j in range(cfg.num_experts):
        if cfg.upcycle_strategy == "sliced_granular":
            slice_id = j % cfg.top_k
            s0, s1 = slice_id * exp_ff, (slice_id + 1) * exp_ff
            # Multiply down_proj by K_r so uniform weighted average sum_{k} (1/K_r) * y_k matches dense sum
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
                noise = torch.randn(src.shape, device=src.device, dtype=src.dtype, generator=gen)
                dst.add_(cfg.upcycle_init_noise * noise)

    if cfg.upcycle_strategy == "progressive_split":
        nn.init.zeros_(block.router.gate.bias)
        active_k0 = min(cfg.progressive_active_experts, cfg.num_experts)
        block.router.gate.bias[active_k0:].fill_(cfg.dormant_init_bias)
        # Seed initial EMA load cleanly to reflect dormant slots
        fair_active = cfg.top_k / max(active_k0, 1)
        block.router.ema_load[:active_k0].fill_(fair_active)
        block.router.ema_load[active_k0:].fill_(0.0)

    return block


def upcycle_gemma_model(dense_model: nn.Module, cfg: EMeansMoEConfig) -> nn.Module:
    """
    Convert either TinyGemmaForCausalLM or a locally loaded HuggingFace Gemma model
    into an auxiliary-loss-free MoE model in-place (never triggering downloads).
    """
    if isinstance(dense_model, TinyGemmaForCausalLM):
        return TinyGemmaMoEForCausalLM(copy.deepcopy(dense_model), cfg)

    # Generic HF Gemma2ForCausalLM / Gemma3ForCausalLM support via attribute path
    cloned = copy.deepcopy(dense_model)
    layers = getattr(getattr(cloned, "model", cloned), "layers")
    for idx, layer in enumerate(layers):
        layer.mlp = upcycle_gemma_mlp_to_moe(layer.mlp, cfg, layer_idx=idx)
    return cloned
