import numpy as np
import torch
import re

def parse_param(param):
    reg = re.compile("\.\d+\.")
    finded = reg.findall(param)
    if len(finded) == 0:
        pass
    else:
        for f in finded:
            f = f[1:-1]
            param = param.replace(f".{f}.", f"[{f}].")
    return param


def caculate_param_remove(all_param_names, all_totals, net, thre=0.25):
    param_remove = dict()
    for param in all_param_names:
        param_remove[param] = None
    for i in range(len(all_totals)):
        totals = all_totals[i]
        totals = [totals[param] for param in all_param_names]
        param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()")
                         for param in all_param_names]
        combine = [np.abs(total * weight)
                   for total, weight in zip(totals, param_weights)]
        combine = np.array(combine)
        combine_flatten = np.concatenate(
            [combine_.flatten() for combine_ in combine], axis=0)
        threshold = np.sort(combine_flatten)[
            ::-1][int(len(combine_flatten) * thre)]
        for idx, param in enumerate(all_param_names):
            if param_remove[param] is None:
                param_remove[param] = combine[idx] > threshold
            else:
                t = combine[idx] > threshold
                param_remove[param] = param_remove[param] | t


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