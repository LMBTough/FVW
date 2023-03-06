import torch
import numpy as np
import random
from tqdm.notebook import tqdm
import copy
import re


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


device = get_device()



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_param(param):
    reg = re.compile("\.\d+\.")
    finded = reg.findall(param)
    if len(finded) == 0:
        return param
    else:
        for f in finded:
            f = f[1:-1]
            param = param.replace(f".{f}.", f"[{f}].")
    return parse_param(param)


def calculate_pruned(model, removed):
    leave, all_num = 0, 0
    for name, param in model.named_parameters():
        param = param.cpu().detach().numpy()
        if name in removed:
            leave += removed[name].sum()
        else:
            leave += param.size
        all_num += param.size
    return float(1 - (leave / all_num))


def get_all_param_names(model):
    parameters = list(model.named_parameters())
    all_param_names = list()
    i = 0
    while i < len(parameters):
        if len(parameters[i][1].shape) == 1 and "weight" in parameters[i][0]:
            i += 2
            continue
        else:
            all_param_names.append(parameters[i][0])
            i += 1
    all_param_names = all_param_names[1:-1]
    return all_param_names


def test_model(net, test_loader):
    net.eval()
    preds = list()
    labels = list()
    with torch.no_grad():
        for x, y in tqdm(test_loader, total=len(test_loader)):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            preds.append(outputs.cpu().detach().numpy())
            labels.append(y.cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return (preds.argmax(axis=1) == labels).mean()


def calculate_param_remove(net, all_param_names, all_totals, all_param_masks=None, sum=False, thre=0.25):
    if sum:
        totals = all_totals[0]
        for key in totals:
            totals[key]
        if len(all_totals) > 1:
            for i in range(1, len(all_totals)):
                all_total = all_totals[i]
                for key in totals:
                    totals[key] += all_total[key]
        all_totals = [totals]
    param_remove = dict()
    for param in all_param_names:
        param_remove[param] = None
    for i in range(len(all_totals)):
        totals = all_totals[i]
        totals = [totals[param] for param in all_param_names]
        param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()", {"net": net})
                         for param in all_param_names]
        masks = None
        if all_param_masks is not None:
            masks = [all_param_masks[param] for param in all_param_names]
        if masks is not None:
            combine = [np.abs(total * weight) * mask.cpu().detach().numpy()
                       for total, weight, mask in zip(totals, param_weights, masks)]
        else:
            combine = [np.abs(total * weight)
                       for total, weight in zip(totals, param_weights)]
        combine = np.array(combine)
        if masks is not None:
            combine_flatten = np.concatenate(
                [combine_.flatten()[mask.cpu().detach().numpy().flatten().astype(bool)] for combine_, mask in zip(combine, masks)], axis=0)
        else:
            combine_flatten = np.concatenate(
                [combine_.flatten() for combine_ in combine], axis=0)
        percentile = 100 - thre * 100
        threshold = np.percentile(combine_flatten, percentile)
        for idx, param in enumerate(all_param_names):
            t = combine[idx] > threshold
            if param_remove[param] is None:
                param_remove[param] = t
            else:
                param_remove[param] = param_remove[param] | t
    return param_remove


def prune_model(model, param_remove, all_param_names):
    model = copy.deepcopy(model)
    with torch.no_grad():
        for param in tqdm(all_param_names):
            param_ = parse_param(param)
            try:
                exec("model." + param_ + "[~param_remove[param]] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
            except:
                exec("model." + param_ + "[~param_remove[param],:] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
    return model
