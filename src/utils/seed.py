"""Random seed management for reproducibility."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for reproducible results.

    Controls randomness in Python's built-in random module, NumPy, PyTorch CPU,
    and PyTorch CUDA to ensure deterministic behavior across runs.

    Args:
        seed: Integer seed value. Defaults to 42.

    Example::

        set_seed(42)
        # All subsequent random operations are now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
