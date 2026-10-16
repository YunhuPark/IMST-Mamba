from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.evaluation.challenge2019_utility import choose_utility_threshold, normalized_utility
from src.models.imst_mamba import IMSTMamba

FEATURES = ["HR", "SBP", "DBP", "Resp", "Temp", "O2Sat"]
TARGET = "SepsisLabel"


@dataclass
class PatientRecord:
    patient_id: str
    values: np.ndarray
    mask: np.ndarray
    labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_patient_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.psv"))


def read_patient(path: Path) -> PatientRecord:
    frame = pd.read_csv(path, sep="|", usecols=[*FEATURES, TARGET])
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = (~np.isnan(values)).astype(np.float32)
    labels = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{path.name}: SepsisLabel must be binary")
    return PatientRecord(path.stem, values, mask, labels)


def patient_split(records: list[PatientRecord], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Exact 70/15/15 patient-level stratification used by Medi-Matrix benchmarks."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, record in enumerate(records):
        by_label[int(record.labels.max())].append(index)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in (0, 1):
        ids = np.asarray(by_label[label], dtype=int)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train.extend(ids[:n_train].tolist())
        val.extend(ids[n_train:n_train + n_val].tolist())
        test.extend(ids[n_train + n_val:].tolist())
    return train, val, test


def fit_stats(records: list[PatientRecord], indices: list[int]) -> dict[str, list[float]]:
    sums = np.zeros(len(FEATURES), dtype=np.float64)
    sumsq = np.zeros(len(FEATURES), dtype=np.float64)
    counts = np.zeros(len(FEATURES), dtype=np.float64)
    for index in indices:
        record = records[index]
        for j in range(len(FEATURES)):
            observed = record.values[:, j][record.mask[:, j] > 0.5]
            if len(observed):
                sums[j] += observed.sum(dtype=np.float64)
                sumsq[j] += np.square(observed, dtype=np.float64).sum(dtype=np.float64)
                counts[j] += len(observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(sumsq, counts, out=np.ones_like(sumsq), where=counts > 0) - means**2
    stds = np.sqrt(np.maximum(variance, 1e-8))
    return {"mean": means.tolist(), "std": stds.tolist()}


def transform(record: PatientRecord, stats: dict[str, list[float]]) -> dict[str, torch.Tensor | str]:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    x = np.where(record.mask > 0.5, (record.values - mean) / std, 0.0).astype(np.float32)

    length, n_features = x.shape
    since = np.full((length, n_features), 1e9, dtype=np.float32)
    last_seen = np.full(n_features, -1, dtype=np.int64)
    for t in range(length):
        observed = record.mask[t] > 0.5
        last_seen[observed] = t
        seen = last_seen >= 0
        since[t, seen] = (t - last_seen[seen]).astype(np.float32)

    delta_t = np.ones(length, dtype=np.float32)
    if length:
        delta_t[0] = 0.0

    return {
        "patient_id": record.patient_id,
        "x": torch.from_numpy(x),
        "m": torch.from_numpy(record.mask.astype(np.float32)),
        "s": torch.from_numpy(np.log1p(np.minimum(since, 1e9)).astype(np.float32)),
        "delta_t": torch.from_numpy(delta_t),
        "y": torch.from_numpy(record.labels.astype(np.float32)),
    }


class PatientDataset(Dataset):
    def __init__(self, records: list[PatientRecord], indices: list[int], stats: dict[str, list[float]]):
        self.samples = [transform(records[index], stats) for index in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate(batch):
    lengths = torch.tensor([len(item["y"]) for item in batch], dtype=torch.long)
    x = pad_sequence([item["x"] for item in batch], batch_first=True)
    m = pad_sequence([item["m"] for item in batch], batch_first=True)
    s = pad_sequence([item["s"] for item in batch], batch_first=True)
    delta_t = pad_sequence([item["delta_t"] for item in batch], batch_first=True)
    y = pad_sequence([item["y"] for item in batch], batch_first=True, padding_value=-1.0)
    mask = y >= 0.0
    return {
        "x": x,
        "m": m,
        "s": s,
        "delta_t": delta_t,
        "y": y,
        "attention_mask": mask,
        "lengths": lengths,
        "patient_ids": [item["patient_id"] for item in batch],
    }


def make_loader(dataset: PatientDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    labels_by_patient: list[np.ndarray] = []
    probs_by_patient: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            s = batch["s"].to(device)
            delta_t = batch["delta_t"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["y"].to(device)
            logits = model(x, m, delta_t, s, attn)["logit_sepsis"].squeeze(-1)
            loss = criterion(logits[attn], y[attn])
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_cpu = batch["y"].numpy()
            for i, length in enumerate(batch["lengths"].tolist()):
                labels_by_patient.append(y_cpu[i, :length].astype(int))
                probs_by_patient.append(probs[i, :length].astype(float))
    return labels_by_patient, probs_by_patient, float(np.mean(losses))


def metric_report(labels_by_patient, probs_by_patient, threshold: float) -> dict[str, float | int]:
    y_true = np.concatenate(labels_by_patient)
    prob = np.concatenate(probs_by_patient)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "threshold": float(threshold),
        "challenge_utility": normalized_utility(labels_by_patient, probs_by_patient, threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    files = iter_patient_files(args.input)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No PSV files found")

    records = [read_patient(path) for path in files]
    train_idx, val_idx, test_idx = patient_split(records, args.seed)
    stats = fit_stats(records, train_idx)

    train_ds = PatientDataset(records, train_idx, stats)
    val_ds = PatientDataset(records, val_idx, stats)
    test_ds = PatientDataset(records, test_idx, stats)
    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)
    test_loader = make_loader(test_ds, args.batch_size, False)

    device = torch.device("cpu")
    model = IMSTMamba(
        n_features=len(FEATURES),
        feature_names=FEATURES,
        d_model=48,
        d_state=8,
        n_layers=1,
        d_miss=4,
        d_time=16,
        dropout=0.10,
        use_auxiliary=False,
    ).to(device)

    positive = sum(int(records[i].labels.sum()) for i in train_idx)
    total = sum(len(records[i].labels) for i in train_idx)
    negative = max(total - positive, 1)
    positive = max(positive, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            s = batch["s"].to(device)
            delta_t = batch["delta_t"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, m, delta_t, s, attn)["logit_sepsis"].squeeze(-1)
            loss = criterion(logits[attn], y[attn])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        _, _, val_loss = evaluate(model, val_loader, device, criterion)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss}
        history.append(record)
        print(json.dumps(record))
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    val_labels, val_probs, _ = evaluate(model, val_loader, device, criterion)
    threshold = choose_utility_threshold(val_labels, val_probs)
    test_labels, test_probs, _ = evaluate(model, test_loader, device, criterion)

    report = {
        "model_name": "imst_mamba_compact_six_vitals_challenge2019_v1",
        "comparison_scope": "same six Vitals, same official SepsisLabel, same patient split and utility protocol as Medi-Matrix baselines",
        "clinical_use": False,
        "source_dataset": "PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0",
        "features": FEATURES,
        "target": "official SepsisLabel",
        "target_semantics": "source label used directly; no additional shift",
        "split": {
            "strategy": "patient-level stratified 70/15/15",
            "seed": args.seed,
            "train_patients": len(train_idx),
            "validation_patients": len(val_idx),
            "test_patients": len(test_idx),
        },
        "architecture": {
            "d_model": 48, "d_state": 8, "n_layers": 1, "d_miss": 4, "d_time": 16,
            "epochs_requested": args.epochs, "epochs_completed": len(history),
        },
        "validation": metric_report(val_labels, val_probs, threshold),
        "test": metric_report(test_labels, test_probs, threshold),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "stats": stats, "features": FEATURES, "threshold": threshold}, args.artifact_dir / "model.pt")
    (args.artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.artifact_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.artifact_dir / "split_manifest.json").write_text(json.dumps({
        "train": [records[i].patient_id for i in train_idx],
        "validation": [records[i].patient_id for i in val_idx],
        "test": [records[i].patient_id for i in test_idx],
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
