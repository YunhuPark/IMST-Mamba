"""
YAML config loading with hierarchical merging.
Usage:
    cfg = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    cfg = override(cfg, "training.lr", 1e-3)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override takes precedence)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(*paths: str | Path) -> dict:
    """Load and merge multiple YAML configs. Later files override earlier ones."""
    cfg: dict = {}
    for p in paths:
        cfg = _deep_merge(cfg, _load_yaml(p))
    return cfg


def override(cfg: dict, key_path: str, value: Any) -> dict:
    """
    Override a nested config value using dot notation.
    Example: override(cfg, "training.optimizer.lr", 1e-4)
    """
    cfg = copy.deepcopy(cfg)
    keys = key_path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return cfg


def get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation."""
    keys = key_path.split(".")
    node = cfg
    try:
        for k in keys:
            node = node[k]
        return node
    except (KeyError, TypeError):
        return default
