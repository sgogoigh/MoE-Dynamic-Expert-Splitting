"""Token sources for the PoCs, with training and held-out batches that cannot overlap.

`SyntheticSource` wraps `make_multi_domain_batch`: training seeds live in [0, VAL_SEED_OFFSET)
and held-out seeds in [VAL_SEED_OFFSET, 2*VAL_SEED_OFFSET). The two pattern domains have few
distinct sequences, so a held-out batch can repeat a training sequence; held-out loss there
measures how well the rule is fitted, not generalisation.

`ShardSource` reads one tokenized flat .npy shard per domain. The last `val_frac` of every
shard is held out and training windows never cross into it.
"""

import pathlib
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .synthetic_data import DOMAIN_NAMES, make_multi_domain_batch

VAL_SEED_OFFSET = 1_000_000_000
SPLITS = ("train", "val")


class SyntheticSource:
    domain_names: Tuple[str, ...] = DOMAIN_NAMES

    def batch(self, seed: int, batch_size: int, seq_len: int, vocab_size: int, split: str = "train",
              device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if not 0 <= int(seed) < VAL_SEED_OFFSET:
            raise ValueError(f"seed must be in [0, {VAL_SEED_OFFSET}), got {seed}")
        offset = VAL_SEED_OFFSET if split == "val" else 0
        return make_multi_domain_batch(int(seed) + offset, batch_size, seq_len, vocab_size, device)


class ShardSource:
    """One flat token shard per domain (e.g. prose.npy, code.npy, math.npy)."""

    def __init__(self, data_dir, names: Sequence[str] = ("prose.npy", "code.npy", "math.npy"),
                 weights: Sequence[float] = (0.60, 0.25, 0.15), val_frac: float = 0.1):
        if not 0.0 < val_frac < 1.0:
            raise ValueError("val_frac must be in (0, 1)")
        root = pathlib.Path(data_dir)
        found = [(n, w) for n, w in zip(names, weights) if (root / n).exists()]
        if not found:
            raise FileNotFoundError(f"none of {list(names)} found in {root}")
        self.domain_names = tuple(pathlib.Path(n).stem for n, _ in found)
        self.arrays = [np.load(root / n, mmap_mode="r") for n, _ in found]
        w = np.array([w for _, w in found], dtype=np.float64)
        self.weights = w / w.sum()
        self.val_frac = val_frac

    def batch(self, seed: int, batch_size: int, seq_len: int, vocab_size: int, split: str = "train",
              device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        rng = np.random.default_rng([int(seed), SPLITS.index(split)])
        doms = rng.choice(len(self.arrays), size=batch_size, p=self.weights)
        out = np.empty((batch_size, seq_len), dtype=np.int64)
        for i, d in enumerate(doms):
            arr = self.arrays[d]
            cut = int(len(arr) * (1.0 - self.val_frac))
            lo, hi = (0, cut - seq_len) if split == "train" else (cut, len(arr) - seq_len)
            if hi < lo:
                raise ValueError(f"shard {self.domain_names[d]} too short for seq_len={seq_len} ({split})")
            pos = int(rng.integers(lo, hi + 1))
            out[i] = arr[pos: pos + seq_len]
        if out.max() >= vocab_size or out.min() < 0:
            raise ValueError(f"token id outside [0, {vocab_size}): the shards need a model with a larger vocab")
        return torch.from_numpy(out).to(device), torch.from_numpy(doms.astype(np.int64)).to(device)


def make_source(data_dir: Optional[str] = None):
    return SyntheticSource() if data_dir is None else ShardSource(data_dir)
