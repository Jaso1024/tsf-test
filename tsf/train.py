import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsf.data import SyntheticConfig, SyntheticTS, ZNormalizer, compute_channel_stats, collate_with_horizon
from tsf.data_real import build_real_datasets, SplitConfig
from tsf.models.transformer import GenConfig, DiscConfig, Generator, Discriminator
from tsf.losses import huber_loss, quantile_loss, wgan_gp, feature_matching_loss, spectral_loss_stft
from tsf.metrics import mae, smape, mase, coverage_and_width
from tsf.utils import FlashSDPA, AmpConfig, set_seed, ensure_dir, now_str, make_future_mask, causal_attn_mask, count_parameters, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser("TSF-TransGAN v0.1 – synthetic overnight")
    # Data
    p.add_argument('--data.name', dest='data_name', choices=['synthetic','etth1','etth2','ettm1','ettm2','electricity','csv'], default='synthetic')
    p.add_argument('--data.root', dest='data_root', type=str, default=None, help='Directory containing known datasets (ETT/Electricity).')
    p.add_argument('--csv.path', dest='csv_path', type=str, default=None, help='Path to a generic CSV when data_name=csv')
    p.add_argument('--csv.time_col', dest='time_col', type=str, default=None)
    p.add_argument('--csv.use_cols', dest='use_cols', type=str, nargs='*', default=None)
    p.add_argument('--split.train_ratio', dest='train_ratio', type=float, default=0.7)
    p.add_argument('--split.val_ratio', dest='val_ratio', type=float, default=0.1)
    p.add_argument('--split.stride', dest='stride', type=int, default=1)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--T_in', type=int, default=256)
    p.add_argument('--H_max', type=int, default=96)
    p.add_argument('--horizon_min', type=int, default=24)
    p.add_argument('--horizon_max', type=int, default=96)
    p.add_argument('--seed', type=int, default=1337)
    # Model
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--d_z', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--spectral_norm_D', action='store_true')
    # Loss
    p.add_argument('--loss', choices=['huber', 'quantile'], default='quantile')
    p.add_argument('--delta', type=float, default=1.0, help='Huber delta')
    p.add_argument('--quantiles', type=float, nargs='*', default=[0.1, 0.5, 0.9])
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--lambda_gp', type=float, default=10.0)
    p.add_argument('--stft_nfft', type=int, default=64)
    p.add_argument('--stft_hop', type=int, default=16)
    # Optim
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--iters', type=int, default=120000)
    p.add_argument('--n_critic', type=int, default=4)
    p.add_argument('--g_lr', type=float, default=2e-4)
    p.add_argument('--d_lr', type=float, default=1e-4)
    p.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--amp', choices=['bf16', 'fp16', 'off'], default='bf16')
    p.add_argument('--flash', type=bool, default=True)
    # Runtime
    p.add_argument('--device', default='cuda')
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--eval_every', type=int, default=2000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--out_dir', default='runs/tsf_transgan_v0_1')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.out_dir) / now_str()
    ensure_dir(str(run_dir))

    # Data
    if args.data_name == 'synthetic':
        synth_cfg = SyntheticConfig(K=args.K, T_in=args.T_in, H_max=args.H_max)
        ds_train = SyntheticTS(synth_cfg, split='train')
        ds_val = SyntheticTS(synth_cfg, split='val')

        # Compute z-score stats from synthetic distribution samples
        stat_loader = DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=0, drop_last=False)
        normalizer = compute_channel_stats(stat_loader)

        def _norm_batch(past, fut):
            past = normalizer.normalize(past)
            fut = normalizer.normalize(fut)
            return past, fut

        collate = lambda batch: collate_with_horizon(batch, args.horizon_min, args.horizon_max)
        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            drop_last=True, collate_fn=collate)
        vloader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             drop_last=False, collate_fn=collate)
        K_model = args.K
    else:
        # Real datasets (ETT/Electricity/generic CSV)
        split_cfg = SplitConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio, stride=args.stride)
        ds_train, ds_val, ds_test, mean, std = build_real_datasets(
            source=args.data_name,
            root=args.data_root,
            csv_path=args.csv_path,
            T_in=args.T_in,
            H_max=args.H_max,
            split=split_cfg,
            time_col=args.time_col,
            use_cols=args.use_cols,
        )
        # Normalizer created from returned mean/std
        normalizer = ZNormalizer(mean=mean, std=std)
        K_model = mean.numel()

        def _norm_batch(past, fut):
            # Already normalized inside build_real_datasets; keep as identity for clarity
            return past, fut

        collate = lambda batch: collate_with_horizon(batch, args.horizon_min, args.horizon_max)
        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            drop_last=True, collate_fn=collate)
        vloader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             drop_last=False, collate_fn=collate)

    # Models
    use_quantiles = args.loss == 'quantile'
    gen_cfg = GenConfig(K=K_model, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                        d_z=args.d_z, dropout=args.dropout, use_segment_embed=True,
                        quantiles=(args.quantiles if use_quantiles else None))
    disc_cfg = DiscConfig(K=K_model, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                          dropout=args.dropout, use_segment_embed=True, spectral_norm=args.spectral_norm_D)
    G = Generator(gen_cfg).to(device)
    D = Discriminator(disc_cfg).to(device)
    print(f"G params: {count_parameters(G):,} | D params: {count_parameters(D):,}")

    # Optims
    optG = torch.optim.AdamW(G.parameters(), lr=args.g_lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    optD = torch.optim.AdamW(D.parameters(), lr=args.d_lr, betas=tuple(args.betas), weight_decay=args.weight_decay)

    amp = AmpConfig(args.amp)
    scalerG = amp.scaler()
    scalerD = amp.scaler()

    step = 0
    best_mae = float('inf')

    with FlashSDPA(args.flash):
        pbar = tqdm(total=args.iters, dynamic_ncols=True)
        while step < args.iters:
            for batch in loader:
                past, fut, N = batch
                past, fut = _norm_batch(past, fut)
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                B, T_in, K = past.shape
                H_max = fut.shape[1]
                mask_future = make_future_mask(B, H_max, N, device)

                # Teacher forcing inputs: shift future by 1 with start token = last past
                start_tok = past[:, -1:, :]
                fut_shift = torch.cat([start_tok, fut[:, :-1, :]], dim=1)

                # === D steps ===
                for _ in range(args.n_critic):
                    z = torch.randn(B, args.d_z, device=device)
                    with torch.no_grad():
                        with amp.autocast():
                            out = G(past, fut_shift, z, attn_mask=None)[:, T_in:, :]
                            if use_quantiles:
                                Q = len(args.quantiles)
                                out_q = out.view(B, H_max, K, Q)
                                # choose median for adversarial path
                                q_idx = min(range(Q), key=lambda i: abs(args.quantiles[i] - 0.5))
                                y_fake = out_q[..., q_idx]
                            else:
                                y_fake = out
                    # WGAN-GP components (done in float32 inside function)
                    mr, mf, gp = wgan_gp(D, past, fut.detach(), y_fake.detach(), mask_future, args.lambda_gp)
                    d_loss = (-mr.mean() + mf.mean()) + gp

                    optD.zero_grad(set_to_none=True)
                    if scalerD is not None:
                        scalerD.scale(d_loss).backward()
                        scalerD.unscale_(optD)
                        torch.nn.utils.clip_grad_norm_(D.parameters(), args.clip_grad)
                        scalerD.step(optD)
                        scalerD.update()
                    else:
                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(D.parameters(), args.clip_grad)
                        optD.step()

                # === G step ===
                z = torch.randn(B, args.d_z, device=device)
                with amp.autocast():
                    out = G(past, fut_shift, z, attn_mask=None)[:, T_in:, :]
                    if use_quantiles:
                        Q = len(args.quantiles)
                        out_q = out.view(B, H_max, K, Q)
                        # forecast loss on quantiles
                        L_fore = quantile_loss(out_q, fut, args.quantiles, mask_future)
                        # take median for metrics and adv
                        q_idx = min(range(Q), key=lambda i: abs(args.quantiles[i] - 0.5))
                        y_fake = out_q[..., q_idx]
                        pred_for_metrics = y_fake
                    else:
                        L_fore = huber_loss(out, fut, delta=args.delta, mask=mask_future)
                        y_fake = out
                        pred_for_metrics = out

                    # Adversarial term (use D in current precision)
                    logits_fake, fake_feats = D(past, y_fake, return_feats=True)
                    H = H_max
                    D_fake = (logits_fake[:, -H:, 0] * mask_future.float()).sum(dim=1) / (mask_future.sum(dim=1) + 1e-8)
                    L_adv = -D_fake.mean()

                    # Feature matching: compare to D(real) features (detach real)
                    with torch.no_grad():
                        logits_real, real_feats = D(past, fut, return_feats=True)
                    L_feat = feature_matching_loss(fake_feats, [rf.detach() for rf in real_feats], mask_future)

                    # Spectral
                    L_stft = spectral_loss_stft(y_fake, fut, mask_future, n_fft=args.stft_nfft, hop_length=args.stft_hop)

                    L_G = args.alpha * L_fore + args.beta * L_adv + args.gamma * L_feat + args.eta * L_stft

                optG.zero_grad(set_to_none=True)
                if scalerG is not None:
                    scalerG.scale(L_G).backward()
                    scalerG.unscale_(optG)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.clip_grad)
                    scalerG.step(optG)
                    scalerG.update()
                else:
                    L_G.backward()
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.clip_grad)
                    optG.step()

                # Metrics (teacher-forced, median)
                with torch.no_grad():
                    L_mae = mae(pred_for_metrics, fut, mask_future).item()
                    L_smape = smape(pred_for_metrics, fut, mask_future).item()
                    L_mase = mase(pred_for_metrics, fut, past, mask_future).item()
                    if use_quantiles:
                        cov, width = coverage_and_width(out_q, fut, args.quantiles, mask_future)
                        cov = cov.item()
                        width = width.item()
                    else:
                        cov, width = float('nan'), float('nan')

                if step % args.log_every == 0:
                    pbar.set_description(f"it {step} | D: {d_loss.item():.3f} | G: {L_G.item():.3f} | fore {L_fore.item():.3f} adv {L_adv.item():.3f} fm {L_feat.item():.3f} stft {L_stft.item():.3f} | MAE {L_mae:.3f} sMAPE {L_smape:.3f} MASE {L_mase:.3f} cov {cov:.3f} w {width:.3f}")

                if step % args.eval_every == 0 and step > 0:
                    # quick val pass (teacher-forced) on few batches
                    G.eval(); D.eval()
                    v_mae, v_cnt = 0.0, 0
                    with torch.no_grad():
                        for j, vbatch in enumerate(vloader):
                            vpast, vfut, vN = vbatch
                            vpast, vfut = _norm_batch(vpast, vfut)
                            vpast = vpast.to(device)
                            vfut = vfut.to(device)
                            Bv, Hv = vfut.size(0), vfut.size(1)
                            vmask = make_future_mask(Bv, Hv, vN, device)
                            vstart = vpast[:, -1:, :]
                            vfshift = torch.cat([vstart, vfut[:, :-1, :]], dim=1)
                            zout = torch.randn(Bv, args.d_z, device=device)
                            vout = G(vpast, vfshift, zout, attn_mask=None)[:, vpast.size(1):, :]
                            if use_quantiles:
                                Q = len(args.quantiles)
                                vout_q = vout.view(Bv, Hv, args.K, Q)
                                q_idx = min(range(Q), key=lambda i: abs(args.quantiles[i] - 0.5))
                                vpred = vout_q[..., q_idx]
                            else:
                                vpred = vout
                            v_mae += mae(vpred, vfut, vmask).item()
                            v_cnt += 1
                            if j >= 9:
                                break
                    v_mae /= max(v_cnt, 1)
                    if v_mae < best_mae:
                        best_mae = v_mae
                        ckpt_path = run_dir / f"best_mae_{best_mae:.4f}_step{step}.pt"
                        save_checkpoint(str(ckpt_path), G=G.state_dict(), D=D.state_dict(), step=step, args=vars(args))
                    G.train(); D.train()

                if step % args.save_every == 0 and step > 0:
                    ckpt_path = run_dir / f"checkpoint_step{step}.pt"
                    save_checkpoint(str(ckpt_path), G=G.state_dict(), D=D.state_dict(), step=step, args=vars(args))

                step += 1
                pbar.update(1)
                if step >= args.iters:
                    break
        pbar.close()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
