"""Held-out evaluation for the PoCs: next-token loss per domain and the domain x expert table."""

import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma_upcycle import frozen_moe_stats, iter_moe_blocks


def _logits_of(out) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def _mutual_information(joint: torch.Tensor) -> Tuple[float, float]:
    """I(D; E) and H(D) in nats from an unnormalised (domain, expert) weight table."""
    total = float(joint.sum())
    if total <= 0:
        return 0.0, 0.0
    p = joint / total
    pd, pe = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    nz = p > 0
    mi = float((p[nz] * (p[nz] / (pd * pe).expand_as(p)[nz]).log()).sum())
    hd = float(-(pd[pd > 0] * pd[pd > 0].log()).sum())
    return mi, hd


@torch.no_grad()
def heldout_eval(model: nn.Module, batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                 domain_names: Sequence[str], forward: Optional[Callable] = None) -> Dict[str, object]:
    """Evaluate on (tokens (B, L), domain_ids (B,)) batches in eval mode, routing stats frozen.

    Returns the mean next-token cross-entropy overall and per domain, and for every MoE layer
    the share of each domain's routing weight that each expert receives (rows sum to 1), plus
    `routing_domain_info` = I(domain; expert) / H(domain), the fraction of the domain label the
    routing carries (0 = routing ignores the domain, 1 = the domain is recoverable from it).
    The previous train/eval mode is restored. `forward(tokens)` overrides `model(tokens)`
    (e.g. to pass use_cache=False to a HuggingFace model).
    """
    run = forward if forward is not None else model
    blocks = iter_moe_blocks(model)
    n_dom = len(domain_names)
    tables = {name: torch.zeros(n_dom, blk.router.E, dtype=torch.float64) for name, blk in blocks}
    current: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            current[name] = (out.topk_idx.detach(), out.topk_weights.detach())
        return hook

    hooks = [blk.router.register_forward_hook(make_hook(name)) for name, blk in blocks]
    loss_sum = torch.zeros(n_dom, dtype=torch.float64)
    tok_cnt = torch.zeros(n_dom, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        with frozen_moe_stats(model):
            for tokens, domains in batches:
                current.clear()
                logits = _logits_of(run(tokens)).float()
                B, L, V = logits.shape
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1), reduction="none")
                dom = domains.detach().cpu().long()
                loss_sum.index_add_(0, dom, ce.view(B, L - 1).sum(1).double().cpu())
                tok_cnt.index_add_(0, dom, torch.full((B,), float(L - 1), dtype=torch.float64))
                tok_dom = dom.repeat_interleave(L)
                for name, (idx, w) in current.items():
                    idx, w = idx.cpu(), w.double().cpu()
                    if idx.shape[0] != tok_dom.shape[0]:
                        raise RuntimeError(f"{name}: router saw {idx.shape[0]} tokens, expected {tok_dom.shape[0]}")
                    valid = idx >= 0
                    rows = tok_dom.unsqueeze(1).expand_as(idx)[valid]
                    tables[name].index_put_((rows, idx[valid]), w[valid], accumulate=True)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    shares, info = {}, []
    for name, t in tables.items():
        shares[name] = (t / t.sum(1, keepdim=True).clamp_min(1e-12)).tolist()
        mi, hd = _mutual_information(t)
        info.append(mi / hd if hd > 0 else 0.0)
    n_tok = float(tok_cnt.sum())
    return {
        "val_loss": float(loss_sum.sum()) / n_tok if n_tok > 0 else math.nan,
        "per_domain_val_loss": {
            domain_names[i]: (float(loss_sum[i] / tok_cnt[i]) if tok_cnt[i] > 0 else None) for i in range(n_dom)
        },
        "val_tokens": int(n_tok),
        "domain_expert_share": shares,
        "routing_domain_info": sum(info) / len(info) if info else None,
    }
