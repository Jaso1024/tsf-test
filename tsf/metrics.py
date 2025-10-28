from typing import Optional, Sequence, Tuple

import torch


def mae(y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    err = (y_hat - y).abs()
    if mask is not None:
        err = err * mask.unsqueeze(-1)
        return err.sum() / (mask.sum() * y.size(-1) + 1e-8)
    return err.mean()


def smape(y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    num = (y_hat - y).abs()
    den = (y_hat.abs() + y.abs()).clamp_min(eps)
    m = num / den
    if mask is not None:
        m = m * mask.unsqueeze(-1)
        return (2.0 * m).sum() / (mask.sum() * y.size(-1) + 1e-8)
    return (2.0 * m).mean()


def mase(y_hat: torch.Tensor, y: torch.Tensor, past: torch.Tensor, mask: Optional[torch.Tensor] = None, seasonality: int = 1, eps: float = 1e-6) -> torch.Tensor:
    # Naive seasonal forecast baseline on past
    T_in = past.size(1)
    if T_in <= seasonality:
        scale = (past[:, 1:, :] - past[:, :-1, :]).abs().mean(dim=(1, 2)).mean() + eps
    else:
        diffs = (past[:, seasonality:, :] - past[:, :-seasonality, :]).abs()
        scale = diffs.mean() + eps
    abs_err = (y_hat - y).abs()
    if mask is not None:
        abs_err = abs_err * mask.unsqueeze(-1)
        return abs_err.sum() / (mask.sum() * y.size(-1) * scale + 1e-8)
    return abs_err.mean() / scale


def coverage_and_width(pred_q: torch.Tensor, y: torch.Tensor, quantiles: Sequence[float], mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # pred_q: (B,H,K,Q) with Q sorted asc
    Q = len(quantiles)
    assert Q >= 2
    lower = pred_q[..., 0]
    upper = pred_q[..., -1]
    inside = ((y >= lower) & (y <= upper)).to(y.dtype)
    width = (upper - lower)
    if mask is not None:
        inside = inside * mask.unsqueeze(-1)
        width = width * mask.unsqueeze(-1)
        cov = inside.sum() / (mask.sum() * y.size(-1) + 1e-8)
        avg_w = width.sum() / (mask.sum() * y.size(-1) + 1e-8)
        return cov, avg_w
    return inside.float().mean(), width.mean()

