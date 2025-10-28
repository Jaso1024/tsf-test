import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlashSDPA:
    def __init__(self, enable: bool):
        self.enable = enable and torch.cuda.is_available()
        self.ctx = None

    def __enter__(self):
        if not self.enable:
            return None
        try:
            self.ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            )
            self.ctx.__enter__()
        except Exception:
            self.ctx = None
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        if self.ctx is not None:
            self.ctx.__exit__(exc_type, exc, tb)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_future_mask(batch_size: int, H_max: int, N: int, device: torch.device) -> torch.Tensor:
    """Mask for future positions (B, H_max) with True on valid positions [0..N-1]."""
    mask = torch.zeros(batch_size, H_max, dtype=torch.bool, device=device)
    mask[:, :N] = True
    return mask


def causal_attn_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


@dataclass
class AmpConfig:
    mode: str = "bf16"  # 'bf16' | 'fp16' | 'off'

    def autocast(self):
        if self.mode == "bf16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if self.mode == "fp16":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return torch.cuda.amp.autocast(enabled=False)

    def scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        if self.mode == "off":
            return None
        # BF16 typically does not need GradScaler, but we include a no-op scaler for uniformity
        # Use scaler only for FP16
        if self.mode == "fp16":
            return torch.cuda.amp.GradScaler()
        return None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, **state):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")

