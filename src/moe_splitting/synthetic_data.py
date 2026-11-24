"""Synthetic multi-domain token streams with domain labels (for PoCs and tests; no downloads).

Three domains occupy disjoint vocabulary bands, so a router that specialises by domain is
visible in a domain x expert routing table:
  A  [0, V/4)        modular pattern, step 3   (frequent, highly repetitive)
  B  [V/4, 5V/8)     modular pattern, step 5   (code-like)
  C  [5V/8, V)       uniform random tokens     (unpredictable; loss floor ~ log(3V/8))
These are toy streams: they exercise routing mechanics, not language modelling.
"""

from typing import Sequence, Tuple

import torch

DOMAIN_NAMES = ("A_pattern3", "B_pattern5", "C_random")


def make_multi_domain_batch(
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device="cpu",
    weights: Sequence[float] = (0.65, 0.23, 0.12),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens (B, L) long, domain_ids (B,) long). Deterministic in `seed`."""
    if vocab_size < 16:
        raise ValueError("vocab_size must be >= 16")
    a_end, b_end = vocab_size // 4, (5 * vocab_size) // 8
    b_width = b_end - a_end
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    cum = torch.tensor(weights, dtype=torch.float64).cumsum(0) / float(sum(weights))
    tokens = torch.empty((batch_size, seq_len), dtype=torch.long)
    domains = torch.empty((batch_size,), dtype=torch.long)
    pos = torch.arange(seq_len)
    for b in range(batch_size):
        u = float(torch.rand((), generator=g))
        dom = int((cum <= u).sum().clamp_max(2))
        if dom == 0:
            start = int(torch.randint(0, a_end, (), generator=g))
            seq = (start + pos * 3) % a_end
        elif dom == 1:
            start = int(torch.randint(0, b_width, (), generator=g))
            seq = a_end + (start + pos * 5) % b_width
        else:
            seq = torch.randint(b_end, vocab_size, (seq_len,), generator=g)
        tokens[b] = seq
        domains[b] = dom
    return tokens.to(device), domains.to(device)
