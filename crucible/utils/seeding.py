import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a deterministic numpy Generator from seed."""
    return np.random.default_rng(seed)


def make_python_rng(seed: int) -> random.Random:
    """Return a deterministic stdlib Random from seed."""
    return random.Random(seed)


def seed_torch(seed: int, *, deterministic: bool = True) -> None:
    """Seed PyTorch (CPU + CUDA) for reproducible neural-agent runs.

    No-op import guard: torch is an optional ``[gpu]`` dependency, so this is
    only meaningful when the neural baseline is installed. Full bitwise
    determinism on CUDA is best-effort — report seed-averaged results with std.
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
