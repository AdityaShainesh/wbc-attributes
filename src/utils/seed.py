"""Seeding, per CLAUDE.md section 5: Python, NumPy, torch, and deterministic cuDNN.

The seed lives in the config and every reported number is a mean over 3 seeds, so this
has to be the only place seeding happens -- a stray torch.manual_seed elsewhere would
make the three runs correlated in a way nothing reports.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    """Seed every RNG this project touches. Returns the seed, for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed before some cuBLAS kernels will behave deterministically.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def seed_worker(_worker_id):
    """DataLoader worker_init_fn, so workers do not share a NumPy seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
