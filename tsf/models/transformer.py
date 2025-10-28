from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,T,dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # scaled dot-product attention with PyTorch SDPA (Flash if available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if self.causal else attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal,
        )  # (B,h,T,dh)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_mult * d_model)
        self.fc2 = nn.Linear(hidden_mult * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class FiLM(nn.Module):
    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.to_params = nn.Sequential(
            nn.Linear(d_z, 2 * d_model),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: (B, d_z)
        gamma, beta = self.to_params(z).chunk(2, dim=-1)
        return gamma, beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_z: Optional[int], causal: bool, spectral_norm: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.film1 = FiLM(d_z, d_model) if d_z is not None else None
        self.film2 = FiLM(d_z, d_model) if d_z is not None else None
        if spectral_norm:
            # Light-touch spectral norm on projections
            self.attn.proj = nn.utils.spectral_norm(self.attn.proj)
            self.mlp.fc1 = nn.utils.spectral_norm(self.mlp.fc1)
            self.mlp.fc2 = nn.utils.spectral_norm(self.mlp.fc2)

    def apply_film(self, x: torch.Tensor, z: torch.Tensor, film: Optional[FiLM]):
        if film is None:
            return x
        gamma, beta = film(z)  # (B, d)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        h = self.apply_film(h, z, self.film1) if z is not None else h
        h = self.attn(h, attn_mask=attn_mask)
        x = x + self.dropout(h)
        h2 = self.ln2(x)
        h2 = self.apply_film(h2, z, self.film2) if z is not None else h2
        h2 = self.mlp(h2)
        x = x + self.dropout(h2)
        return x


@dataclass
class GenConfig:
    K: int
    d_model: int
    n_layers: int
    n_heads: int
    d_z: int
    dropout: float = 0.1
    use_segment_embed: bool = True
    quantiles: Optional[List[float]] = None  # e.g., [0.1, 0.5, 0.9]


class Generator(nn.Module):
    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.K, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model)
        self.seg = nn.Embedding(2, cfg.d_model) if cfg.use_segment_embed else None
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.d_z, causal=True)
            for _ in range(cfg.n_layers)
        ])
        out_dim = cfg.K if not cfg.quantiles else cfg.K * len(cfg.quantiles)
        self.head = nn.Linear(cfg.d_model, out_dim)

    def forward(self, past: torch.Tensor, future_inp: torch.Tensor, z: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # past: (B, T_in, K), future_inp: (B, H_max, K) — teacher-forced (shifted)
        x = torch.cat([past, future_inp], dim=1)  # (B, T, K)
        x = self.embed(x)
        x = self.pos(x)
        if self.seg is not None:
            B, T_in = past.size(0), past.size(1)
            T = x.size(1)
            seg_ids = torch.zeros(B, T, dtype=torch.long, device=x.device)
            seg_ids[:, T_in:] = 1
            x = x + self.seg(seg_ids)
        for blk in self.blocks:
            x = blk(x, z, attn_mask)
        y = self.head(x)
        return y  # (B, T, out_dim)

    def predict_future(self, past: torch.Tensor, H: int, z: torch.Tensor) -> torch.Tensor:
        # Autoregressive rollout
        device = past.device
        B, T_in, K = past.shape
        last = past[:, -1:, :]  # seed with last past step
        # simple start token: last past value
        generated = []
        for t in range(H):
            # build input: past + previously generated (teacher-forced with previous output)
            fut_inp = torch.cat([last] + generated, dim=1) if generated else last
            attn_mask = torch.triu(torch.ones(T_in + fut_inp.size(1), T_in + fut_inp.size(1), device=device, dtype=torch.bool), diagonal=1)
            out = self.forward(past, fut_inp, z, attn_mask)
            # take last position's prediction and reshape to K
            pred_full = out[:, past.size(1):, :]
            if self.cfg.quantiles:
                Q = len(self.cfg.quantiles)
                pred_full = pred_full.view(B, -1, K, Q)
                # use median (closest to 0.5)
                q_idx = min(range(Q), key=lambda i: abs(self.cfg.quantiles[i] - 0.5))
                pred = pred_full[:, -1, :, q_idx]
            else:
                pred = pred_full[:, -1, :]
            pred = pred.unsqueeze(1)
            generated.append(pred)
        return torch.cat(generated, dim=1)  # (B, H, K)


@dataclass
class DiscConfig:
    K: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float = 0.1
    use_segment_embed: bool = True
    spectral_norm: bool = False


class Discriminator(nn.Module):
    def __init__(self, cfg: DiscConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.K, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model)
        self.seg = nn.Embedding(2, cfg.d_model) if cfg.use_segment_embed else None
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout, d_z=None, causal=False, spectral_norm=cfg.spectral_norm)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Linear(cfg.d_model, 1)
        if cfg.spectral_norm:
            self.head = nn.utils.spectral_norm(self.head)

    def forward(self, past: torch.Tensor, future: torch.Tensor, return_feats: bool = False):
        # past: (B, T_in, K), future: (B, H, K)
        x = torch.cat([past, future], dim=1)
        x = self.embed(x)
        x = self.pos(x)
        if self.seg is not None:
            B, T_in = past.size(0), past.size(1)
            T = x.size(1)
            seg_ids = torch.zeros(B, T, dtype=torch.long, device=x.device)
            seg_ids[:, T_in:] = 1
            x = x + self.seg(seg_ids)
        feats = []
        for blk in self.blocks:
            x = blk(x, z=None, attn_mask=None)
            if return_feats:
                feats.append(x)
        logits = self.head(x)  # (B, T, 1)
        return logits, feats
