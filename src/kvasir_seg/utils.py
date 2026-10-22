import random

import numpy as np
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Pick the best available accelerator: CUDA > Apple MPS > DirectML > CPU."""
    if preferred != "auto":
        if preferred == "directml":
            import torch_directml

            return torch_directml.device()
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml

        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
