"""
Global seed management for reproducibility.
Must be called at the very start of every training/evaluation run.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """DataLoader worker seed initializer — pass as worker_init_fn."""
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


# Module-level worker init functions (needed for Windows multiprocessing pickling)
# One per seed — generated dynamically via make_worker_init_fn below.
def make_worker_init_fn(base_seed: int):
    """Return a top-level-picklable worker_init_fn for the given base_seed."""
    def _fn(worker_id: int) -> None:
        worker_init_fn(worker_id, base_seed)
    _fn.__qualname__ = f"worker_init_fn_seed{base_seed}"
    return _fn
