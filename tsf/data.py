from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SyntheticConfig:
    K: int = 8
    T_in: int = 256
    H_max: int = 96
    n_train: int = 200000
    n_val: int = 4096
    noise_sigma: float = 0.2
    seed: int = 42


class SyntheticTS(Dataset):
    def __init__(self, cfg: SyntheticConfig, split: str = "train"):
        assert split in {"train", "val"}
        self.cfg = cfg
        self.split = split
        self.rng = np.random.RandomState(cfg.seed if split == "train" else cfg.seed + 1)
        self.N = cfg.n_train if split == "train" else cfg.n_val
        self._generate_params()

    def _generate_params(self):
        K = self.cfg.K
        # Random trends and seasonality parameters per series
        self.trend = self.rng.uniform(-0.01, 0.01, size=(self.N, K))
        self.bias = self.rng.uniform(-1.0, 1.0, size=(self.N, K))
        # Up to 3 seasonalities per channel
        self.freqs = self.rng.uniform(2 * np.pi / 128, 2 * np.pi / 8, size=(self.N, K, 3))
        self.amps = self.rng.uniform(0.2, 1.0, size=(self.N, K, 3))
        self.phases = self.rng.uniform(0, 2 * np.pi, size=(self.N, K, 3))

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        K, T_in, H_max = self.cfg.K, self.cfg.T_in, self.cfg.H_max
        t = np.arange(T_in + H_max)
        tr = self.trend[idx][:, None] * t[None, :]
        base = self.bias[idx][:, None] + tr
        seas = np.zeros((K, T_in + H_max))
        for h in range(3):
            seas += (
                self.amps[idx][:, [h]]
                * np.sin(self.freqs[idx][:, [h]] * t + self.phases[idx][:, [h]])
            )
        noise = self.rng.normal(0.0, self.cfg.noise_sigma, size=(K, T_in + H_max))
        x = base + seas + noise
        x = x.astype(np.float32)
        past = torch.from_numpy(x[:, :T_in].T)  # (T_in, K)
        fut = torch.from_numpy(x[:, T_in:].T)   # (H_max, K)
        return past, fut


class ZNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = torch.clamp(std, min=eps)
        self.eps = eps

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def compute_channel_stats(loader: DataLoader) -> ZNormalizer:
    """Compute per-channel mean/std over both past and future across batches.

    Expects loader to yield (past, fut) with shapes (B, T_in, K) and (B, H_max, K).
    """
    sum_x = None
    sum_x2 = None
    count = 0
    for past, fut in loader:
        # past: (B, T_in, K); fut: (B, H_max, K)
        x_sum = past.sum(dim=(0, 1)) + fut.sum(dim=(0, 1))  # (K,)
        x_sum2 = (past.pow(2).sum(dim=(0, 1)) + fut.pow(2).sum(dim=(0, 1)))  # (K,)
        n = past.size(0) * (past.size(1) + fut.size(1))
        if sum_x is None:
            sum_x = x_sum
            sum_x2 = x_sum2
        else:
            sum_x = sum_x + x_sum
            sum_x2 = sum_x2 + x_sum2
        count += n
    mean = sum_x / count
    var = sum_x2 / count - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return ZNormalizer(mean=mean, std=std)


def collate_with_horizon(batch, horizon_min: int, horizon_max: int):
    # Sample a single N per batch in [min, max]
    N = int(np.random.randint(horizon_min, horizon_max + 1))
    past, fut = zip(*batch)
    past = torch.stack(past, dim=0)  # (B, T_in, K)
    fut = torch.stack(fut, dim=0)    # (B, H_max, K)
    return past, fut, N
