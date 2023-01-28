import numpy as np
import torch


def normalization(x):
    min = np.inf
    max = -np.inf
    for x_ in x:
        if np.min(x_) < min:
            min = np.min(x_)
    x = [x_ - min for x_ in x]
    for x_ in x:
        if np.max(x_) > max:
            max = np.max(x_)
    x = [x_ / max for x_ in x]
    x = np.array(x)
    return x

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps:
        device = "mps"
    else:
        device = "cpu"
    return device