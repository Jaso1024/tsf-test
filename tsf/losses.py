from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin = abs_err - quad
    loss = 0.5 * quad * quad + delta * lin
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
        return loss.sum() / (mask.sum() * pred.size(-1) + 1e-8)
    return loss.mean()


def quantile_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # pred_q: (B, H, K, Q)
    B, H, K, Q = pred_q.shape
    target = target.unsqueeze(-1)  # (B,H,K,1)
    diff = target - pred_q
    q = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype).view(1, 1, 1, Q)
    loss = torch.maximum(q * diff, (q - 1) * diff).abs()  # pinball
    if mask is not None:
        loss = loss * mask.unsqueeze(-1).unsqueeze(-1)
        return loss.sum() / (mask.sum() * K * Q + 1e-8)
    return loss.mean()


def feature_matching_loss(fake_feats: List[torch.Tensor], real_feats: List[torch.Tensor], mask_future: Optional[torch.Tensor]) -> torch.Tensor:
    # Compare only future positions; assume features are (B, T_total, C)
    loss = 0.0
    for f_fake, f_real in zip(fake_feats, real_feats):
        # Align shapes and slice future region using mask broadcast
        if mask_future is not None:
            # mask_future: (B,H), need broadcast to (B,H,1)
            mf = mask_future.unsqueeze(-1)
            # take last H positions from features
            H = mask_future.size(1)
            f_fake_f = f_fake[:, -H:, :]
            f_real_f = f_real[:, -H:, :]
            diff = (f_fake_f - f_real_f) * mf
            denom = mf.sum() * f_fake.size(-1) + 1e-8
            loss = loss + diff.abs().sum() / denom
        else:
            loss = loss + F.l1_loss(f_fake, f_real)
    return loss


def spectral_loss_stft(y_hat: torch.Tensor, y: torch.Tensor, mask_future: Optional[torch.Tensor], n_fft: int = 64, hop_length: int = 16) -> torch.Tensor:
    # y_hat, y: (B, H, K)
    B, H, K = y.size()
    if mask_future is not None:
        mf = mask_future.unsqueeze(-1).to(y.dtype)
        y_hat = y_hat * mf
        y = y * mf
    # reshape to (B*K, H)
    yh = y_hat.transpose(1, 2).reshape(B * K, H)
    yt = y.transpose(1, 2).reshape(B * K, H)
    # Make STFT robust to short H; avoid center padding error; provide a window
    n_fft_eff = max(8, min(n_fft, H))
    hop_eff = max(2, min(hop_length, max(1, n_fft_eff // 2)))
    window = torch.hann_window(n_fft_eff, device=y.device, dtype=y.dtype)
    YH = torch.stft(yh, n_fft=n_fft_eff, hop_length=hop_eff, window=window, center=False, return_complex=True)
    YT = torch.stft(yt, n_fft=n_fft_eff, hop_length=hop_eff, window=window, center=False, return_complex=True)
    loss = (YH.abs() - YT.abs()).abs().mean()
    return loss


def wgan_gp(discriminator, past: torch.Tensor, real_future: torch.Tensor, fake_future: torch.Tensor, mask_future: torch.Tensor, lambda_gp: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute D(real), D(fake), and gradient penalty over future only in FP32 for stability.
    logits_real, _ = discriminator(past.float(), real_future.float(), return_feats=False)
    logits_fake, _ = discriminator(past.float(), fake_future.float(), return_feats=False)
    H = real_future.shape[1]
    # average masked future logits to scalar per sample
    mr = (logits_real[:, -H:, 0] * mask_future.float()).sum(dim=1) / (mask_future.sum(dim=1) + 1e-8)
    mf = (logits_fake[:, -H:, 0] * mask_future.float()).sum(dim=1) / (mask_future.sum(dim=1) + 1e-8)

    # gradient penalty: interpolate only the future, keep past fixed
    eps = torch.rand(real_future.size(0), 1, 1, device=real_future.device)
    inter_future = eps * real_future.float() + (1.0 - eps) * fake_future.float()
    inter_future.requires_grad_(True)
    inter_logits, _ = discriminator(past.float(), inter_future, return_feats=False)
    inter_scores = (inter_logits[:, -H:, 0] * mask_future.float()).sum(dim=1) / (mask_future.sum(dim=1) + 1e-8)
    grads = torch.autograd.grad(
        outputs=inter_scores.sum(), inputs=inter_future,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.reshape(grads.size(0), -1)
    grad_norm = grads.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return mr.detach(), mf.detach(), gp
