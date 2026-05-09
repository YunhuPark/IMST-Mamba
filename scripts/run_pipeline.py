"""
End-to-end pipeline runner.

Usage:
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage data
    python scripts/run_pipeline.py --stage train --model imst_mamba --seed 42
    python scripts/run_pipeline.py --stage evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.utils.config_loader import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "imst_mamba": ("src.models.imst_mamba", "build_model"),
    "grud": ("src.models.grud", "build_model"),
    "lstm": ("src.models.lstm_baseline", "build_model"),
    "transformer": ("src.models.transformer_baseline", "build_model"),
    "retain": ("src.models.retain", "build_model"),
}

CONFIG_REGISTRY = {
    "imst_mamba": "configs/model_imst_mamba.yaml",
    "grud": "configs/model_grud.yaml",
    "lstm": "configs/model_lstm.yaml",
    "transformer": "configs/model_transformer.yaml",
    "retain": None,
}


def run_data_pipeline(cfg: dict) -> None:
    from src.data.load_challenge2019 import load_challenge2019, normalize_and_copy
    from src.data.splits import make_splits

    raw_dir = Path(cfg["data"]["raw_dir"])
    interim_dir = Path(cfg["data"]["interim_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info("=== Step 1-4: Load Challenge 2019 Data ===")
    load_challenge2019(cfg, raw_dir, interim_dir, processed_dir)

    logger.info("=== Step 5: Make Splits ===")
    splits = make_splits(cfg, interim_dir, processed_dir)

    logger.info("=== Step 6: Normalize and Copy ===")
    normalize_and_copy(processed_dir, splits)

    logger.info("Data pipeline complete!")


def run_training(
    cfg: dict,
    model_name: str,
    seeds: list[int],
    save_dir: Path,
) -> None:
    import importlib
    from src.data.dataset import build_dataloaders
    from src.training.trainer import train

    processed_dir = Path(cfg["data"]["processed_dir"])
    stats_path = processed_dir / "stats.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training {model_name} on {device}")

    for seed in seeds:
        set_seed(seed)
        logger.info(f"\n{'='*50}\nSeed {seed}\n{'='*50}")

        # Load model-specific config
        model_cfg_path = CONFIG_REGISTRY.get(model_name)
        if model_cfg_path and Path(model_cfg_path).exists():
            model_cfg = load_config("configs/base.yaml", model_cfg_path)
        else:
            model_cfg = cfg

        # Build model
        module_path, fn_name = MODEL_REGISTRY[model_name]
        mod = importlib.import_module(module_path)
        build_fn = getattr(mod, fn_name)

        if model_name == "imst_mamba" and stats_path.exists():
            model = build_fn(model_cfg, stats_path=str(stats_path))
        else:
            model = build_fn(model_cfg)

        # Load stats for GRU-D x_mean
        if hasattr(model, "x_mean") and stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            model.x_mean = torch.tensor(stats["mean"], dtype=torch.float32)

        # DataLoaders
        train_loader, val_loader, test_loader = build_dataloaders(
            processed_dir, cfg, horizon="6h", seed=seed,
            fast_mode=cfg.get("training", {}).get("fast_mode", False),
        )

        # Train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=model_cfg,
            model_name=model_name,
            save_dir=save_dir,
            device=device,
            use_multi_task=(model_name == "imst_mamba"),
            seed=seed,
        )

        # Save history
        hist_path = save_dir / f"seed_{seed}" / f"{model_name}_history.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)


def run_evaluation(cfg: dict, save_dir: Path) -> None:
    import importlib
    import numpy as np
    from src.data.dataset import build_dataloaders
    from src.evaluation.metrics import full_metrics, print_metrics
    from src.evaluation.significance_tests import compare_all_baselines, print_comparison_table

    processed_dir = Path(cfg["data"]["processed_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # Collect predictions from all models
    all_probs = {}
    all_labels = None

    for model_name in MODEL_REGISTRY:
        ckpt_paths = list(save_dir.glob(f"*/checkpoints/{model_name}*_best.pt"))
        if not ckpt_paths:
            logger.warning(f"No checkpoint found for {model_name}")
            continue

        # Average predictions across seeds
        seed_probs = []
        for ckpt_path in ckpt_paths:
            module_path, fn_name = MODEL_REGISTRY[model_name]
            mod = importlib.import_module(module_path)
            build_fn = getattr(mod, fn_name)

            model_cfg_path = CONFIG_REGISTRY.get(model_name)
            if model_cfg_path and Path(model_cfg_path).exists():
                model_cfg = load_config("configs/base.yaml", model_cfg_path)
            else:
                model_cfg = cfg

            model = build_fn(model_cfg).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            probs_list, labels_list = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["x"].to(device)
                    m = batch["m"].to(device)
                    delta_t = batch["delta_t"].to(device)
                    s = batch["s"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    y = batch["y"]
                    mask = batch["attention_mask"]

                    out = model(x, m, delta_t, s, attn_mask)
                    p = torch.sigmoid(out["logit_sepsis"]).squeeze(-1)
                    probs_list.append(p[mask].cpu().numpy())
                    if all_labels is None:
                        labels_list.append(y[mask].numpy())

            seed_probs.append(np.concatenate(probs_list))
            if all_labels is None and labels_list:
                all_labels = np.concatenate(labels_list)

        all_probs[model_name] = np.mean(seed_probs, axis=0)

    if all_labels is None or "imst_mamba" not in all_probs:
        logger.error("Missing predictions — run training first")
        return

    # Compute metrics per model
    results = {}
    for name, probs in all_probs.items():
        r = full_metrics(
            all_labels, probs,
            n_bootstrap=cfg["evaluation"]["bootstrap_n"],
            bootstrap_seed=cfg["evaluation"]["bootstrap_seed"],
        )
        results[name] = r
        print_metrics(r, name)

    # Significance tests
    baseline_scores = {k: v for k, v in all_probs.items() if k != "imst_mamba"}
    if "imst_mamba" in all_probs and baseline_scores:
        comparison = compare_all_baselines(
            all_labels, all_probs["imst_mamba"], baseline_scores,
            n_comparisons=cfg["evaluation"]["bonferroni_n_comparisons"],
        )
        print_comparison_table(comparison)

    # Save results
    out_path = save_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x if not hasattr(x, 'tolist') else x.tolist())
    logger.info(f"Results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "data", "train", "evaluate"], default="all")
    parser.add_argument("--model", default="imst_mamba")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fast", action="store_true", help="Use 10%% of data for quick validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("results/logs/pipeline.log"),
        ],
    )

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "results"))

    if args.stage in ("all", "data"):
        run_data_pipeline(cfg)

    if args.stage in ("all", "train"):
        if getattr(args, "fast", False):
            cfg["training"]["fast_mode"] = True
        models_to_train = list(MODEL_REGISTRY.keys()) if args.stage == "all" else [args.model]
        for model_name in models_to_train:
            run_training(cfg, model_name, args.seeds, save_dir)

    if args.stage in ("all", "evaluate"):
        run_evaluation(cfg, save_dir)


if __name__ == "__main__":
    main()
