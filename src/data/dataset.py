"""
PyTorch Dataset and DataLoader for the sepsis prediction task.

Each sample is a single ICU stay represented as an irregular time series.
Variable-length sequences are padded in collate_fn.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import platform
from src.utils.seed import worker_init_fn, make_worker_init_fn

logger = logging.getLogger(__name__)


class SepsisDataset(Dataset):
    """
    Loads pre-built patient tensors from disk.

    Returns per sample:
        x         : (T, F)  normalized feature values
        m         : (T, F)  observation mask
        delta_t   : (T,)    seconds since previous event (normalized)
        s         : (T, F)  seconds since last feature obs (normalized)
        miss_state: (T, F)  3-state missingness class {0, 1, 2}
        y         : (T,)    primary label (6h horizon by default)
        sofa      : (T,)    SOFA score (for auxiliary task)
        seq_len   : int     actual sequence length
        stay_id   : int
    """

    def __init__(
        self,
        data_dir: Path,
        stay_ids: list[int],
        horizon: str = "6h",           # "6h" | "12h" | "24h"
        max_seq_len: Optional[int] = None,
        delta_t_scale: float = 3600.0,  # normalize delta_t to hours
    ):
        self.data_dir = Path(data_dir)
        self.stay_ids = stay_ids
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.delta_t_scale = delta_t_scale

        # Filter to existing files
        self.stay_ids = [
            sid for sid in stay_ids
            if (self.data_dir / f"stay_{sid}.pt").exists()
        ]
        if len(self.stay_ids) < len(stay_ids):
            logger.warning(
                f"Missing {len(stay_ids) - len(self.stay_ids)} pt files in {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.stay_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.stay_ids[idx]
        data = torch.load(
            self.data_dir / f"stay_{sid}.pt",
            weights_only=True,
        )

        x = data["x"]           # (T, F)
        m = data["m"]           # (T, F)
        delta_t = data["delta_t"]    # (T,)
        s = data["s"]               # (T, F)
        miss_state = data["miss_state"]  # (T, F)
        y = data[f"y_{self.horizon}"]   # (T,)
        sofa = data.get("sofa", torch.zeros(x.shape[0]))

        T = x.shape[0]

        # Truncate if needed
        if self.max_seq_len is not None and T > self.max_seq_len:
            x = x[:self.max_seq_len]
            m = m[:self.max_seq_len]
            delta_t = delta_t[:self.max_seq_len]
            s = s[:self.max_seq_len]
            miss_state = miss_state[:self.max_seq_len]
            y = y[:self.max_seq_len]
            sofa = sofa[:self.max_seq_len]
            T = self.max_seq_len

        # Normalize delta_t to hours
        delta_t = delta_t / self.delta_t_scale

        # Clip and log1p-transform s (time since last obs, in hours)
        s_hours = s / self.delta_t_scale
        s_hours = torch.clamp(s_hours, 0, 1000)
        s_log = torch.log1p(s_hours)

        return {
            "x": x,
            "m": m,
            "delta_t": delta_t,
            "s": s_log,
            "miss_state": miss_state,
            "y": y,
            "sofa": sofa,
            "seq_len": T,
            "stay_id": sid,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Pad variable-length sequences to the max length in the batch.
    Returns:
        Same keys as SepsisDataset.__getitem__, with extra:
        attention_mask: (B, T) — 1 for valid positions, 0 for padding
    """
    max_len = max(b["seq_len"] for b in batch)
    B = len(batch)
    F = batch[0]["x"].shape[1]

    x = torch.zeros(B, max_len, F)
    m = torch.zeros(B, max_len, F)
    delta_t = torch.zeros(B, max_len)
    s = torch.zeros(B, max_len, F)
    miss_state = torch.zeros(B, max_len, F, dtype=torch.long)
    y = torch.zeros(B, max_len)
    sofa = torch.full((B, max_len), float("nan"))
    attn_mask = torch.zeros(B, max_len, dtype=torch.bool)
    seq_lens = torch.zeros(B, dtype=torch.long)
    stay_ids = []

    for i, b in enumerate(batch):
        T = b["seq_len"]
        x[i, :T] = b["x"]
        m[i, :T] = b["m"]
        delta_t[i, :T] = b["delta_t"]
        s[i, :T] = b["s"]
        miss_state[i, :T] = b["miss_state"]
        y[i, :T] = b["y"]
        sofa[i, :T] = b["sofa"]
        attn_mask[i, :T] = True
        seq_lens[i] = T
        stay_ids.append(b["stay_id"])

    return {
        "x": x,
        "m": m,
        "delta_t": delta_t,
        "s": s,
        "miss_state": miss_state,
        "y": y,
        "sofa": sofa,
        "attention_mask": attn_mask,
        "seq_lens": seq_lens,
        "stay_ids": stay_ids,
    }


def build_dataloaders(
    processed_dir: Path,
    cfg: dict,
    horizon: str = "6h",
    seed: int = 42,
    fast_mode: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from processed data."""
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)

    obs_window = cfg["data"]["observation_window"]
    batch_size = cfg["training"]["batch_size"]

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    if fast_mode:
        import pandas as pd
        rng = np.random.default_rng(seed)
        labels_df = pd.read_parquet(processed_dir.parent / "interim" / "labels.parquet")
        sepsis_set = set(labels_df[labels_df["has_sepsis"]]["stay_id"].tolist())

        def stratified_sample(ids, frac=0.1):
            pos = [i for i in ids if i in sepsis_set]
            neg = [i for i in ids if i not in sepsis_set]
            n_pos = max(1, int(len(pos) * frac))
            n_neg = max(1, int(len(neg) * frac))
            sampled_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False).tolist()
            sampled_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False).tolist()
            return sampled_pos + sampled_neg

        train_ids = stratified_sample(train_ids)
        val_ids = stratified_sample(val_ids)
        test_ids = stratified_sample(test_ids)
        logger.info(f"Fast mode (stratified 10%%): Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    train_ds = SepsisDataset(processed_dir / "train", train_ids, horizon=horizon,
                             max_seq_len=obs_window * 10)
    val_ds = SepsisDataset(processed_dir / "val", val_ids, horizon=horizon,
                           max_seq_len=obs_window * 10)
    test_ds = SepsisDataset(processed_dir / "test", test_ids, horizon=horizon,
                            max_seq_len=obs_window * 10)

    # Windows requires num_workers=0 (no subprocess forking) or picklable worker_init_fn
    is_windows = platform.system() == "Windows"
    num_workers_train = 0 if is_windows else 4
    num_workers_eval = 0 if is_windows else 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers_train, pin_memory=False,
        worker_init_fn=make_worker_init_fn(seed) if num_workers_train > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers_eval, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers_eval, pin_memory=False,
    )

    logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
