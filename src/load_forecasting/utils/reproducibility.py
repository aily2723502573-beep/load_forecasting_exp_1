import random

import numpy as np
import torch


def set_seed(seed_value: int = 42) -> None:
    if not isinstance(seed_value, int):
        raise TypeError(f"seed_value must be an int, got {type(seed_value)}")

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
