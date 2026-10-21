"""Seeding utility for reproducibility across random, numpy, and torch."""

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set all relevant RNG seeds. Call once at the start of each run script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
