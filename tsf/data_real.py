from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.1  # test gets the rest
    stride: int = 1         # step between consecutive windows


class RealWindowedTS(Dataset):
    """Windowed dataset from a multivariate time series array X of shape (T_total, K).

    Produces overlapping windows with past length T_in and max future length H_max.
    The split is a contiguous segment [start:end) of the series.
    """

    def __init__(self, X: np.ndarray, T_in: int, H_max: int, start: int, end: int, stride: int = 1):
        assert X.ndim == 2
        self.X = X.astype(np.float32, copy=False)
        self.T_in = T_in
        self.H_max = H_max
        self.start = start
        self.end = end
        self.stride = stride
        # number of valid window starts so that [i, i+T_in+H_max) ⊆ [start, end)
        self.n = max(0, (end - start) - (T_in + H_max) + 1)

    def __len__(self) -> int:
        if self.n <= 0:
            return 0
        return 1 + (self.n - 1) // self.stride

    def __getitem__(self, idx: int):
        i = self.start + idx * self.stride
        s0, s1, s2 = i, i + self.T_in, i + self.T_in + self.H_max
        past = torch.from_numpy(self.X[s0:s1])  # (T_in, K)
        fut = torch.from_numpy(self.X[s1:s2])   # (H_max, K)
        return past, fut


def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer explicit names; otherwise detect first datetime-like column
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "time", "timestamp"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            # try parse a sample
            try:
                pd.to_datetime(df[c].iloc[:32])
                return c
            except Exception:
                continue
    return None


def load_csv_timeseries(path: str, use_cols: Optional[Sequence[str]] = None, time_col: Optional[str] = None) -> np.ndarray:
    df = pd.read_csv(path)
    if time_col is None:
        time_col = _detect_time_col(df)
    if time_col is not None and time_col in df.columns:
        # sort by time just in case
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)
        except Exception:
            pass
        df = df.drop(columns=[time_col])
    if use_cols is not None and len(use_cols) > 0:
        cols = [c for c in use_cols if c in df.columns]
        if not cols:
            raise ValueError(f"None of the requested columns {use_cols} are in {path}")
        df = df[cols]
    else:
        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}")
    X = df.to_numpy(dtype=np.float32, copy=False)
    # Drop rows with NaNs at ends, and fill remaining NaNs by forward/back fill
    if np.isnan(X).any():
        df = pd.DataFrame(X)
        df = df.ffill().bfill()
        X = df.to_numpy(dtype=np.float32, copy=False)
    return X


def find_known_dataset_file(root: str, name: str) -> str:
    name = name.lower()
    patterns = {
        "etth1": ["ETTh1.csv", "etth1.csv"],
        "etth2": ["ETTh2.csv", "etth2.csv"],
        "ettm1": ["ETTm1.csv", "ettm1.csv"],
        "ettm2": ["ETTm2.csv", "ettm2.csv"],
        "electricity": ["electricity.csv", "ECL.csv", "electricity_hourly.csv"],
    }
    for cand in patterns.get(name, []):
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find dataset file for '{name}' under {root}. Tried: {patterns.get(name, [])}")


def split_indices(T: int, cfg: SplitConfig, T_in: int, H_max: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    t_train = int(T * cfg.train_ratio)
    t_val = t_train + int(T * cfg.val_ratio)
    # Ensure each split can hold at least one window
    need = T_in + H_max
    t_train = max(t_train, need)
    t_val = max(t_val, t_train + need)
    t_val = min(t_val, T - need)
    t_train = min(t_train, t_val - need)
    train = (0, t_train)
    val = (t_train, t_val)
    test = (t_val, T)
    return train, val, test


def build_real_datasets(
    *,
    source: str,
    root: Optional[str],
    csv_path: Optional[str],
    T_in: int,
    H_max: int,
    split: SplitConfig,
    time_col: Optional[str] = None,
    use_cols: Optional[Sequence[str]] = None,
) -> Tuple[RealWindowedTS, RealWindowedTS, RealWindowedTS, torch.Tensor, torch.Tensor]:
    if source == "csv":
        if not csv_path:
            raise ValueError("csv_path must be provided when source='csv'")
        path = csv_path
    else:
        if not root:
            raise ValueError("root must be provided for known datasets")
        path = find_known_dataset_file(root, source)
    X = load_csv_timeseries(path, use_cols=use_cols, time_col=time_col)
    T_total, K = X.shape
    train_idx, val_idx, test_idx = split_indices(T_total, split, T_in, H_max)
    # Channel-wise stats on train segment only
    X_train = X[train_idx[0]:train_idx[1]]
    mean = torch.from_numpy(X_train.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(X_train.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=1e-6)
    # Normalize in-place for all splits using train stats
    X = (X - mean.numpy()) / std.numpy()

    ds_train = RealWindowedTS(X, T_in, H_max, *train_idx, stride=split.stride)
    ds_val = RealWindowedTS(X, T_in, H_max, *val_idx, stride=split.stride)
    ds_test = RealWindowedTS(X, T_in, H_max, *test_idx, stride=split.stride)
    return ds_train, ds_val, ds_test, mean, std

