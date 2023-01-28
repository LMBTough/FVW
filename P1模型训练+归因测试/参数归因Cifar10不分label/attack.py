import torch
from utils import normalization,get_device
from tqdm.notebook import tqdm
import re
import numpy as np
device = get_device()

def test_model(net, test_loader):
    net.eval()
    preds = list()
    labels = list()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            preds.append(outputs.cpu().detach().numpy())
            labels.append(y.cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return preds, labels


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


def update_param(net, param, alpha, op="add"):
    param = parse_param(param)
    grad = np.array(eval("net." + param + ".grad.cpu().detach().numpy()"))
    if op == "add":
        weight = eval("net." + param + ".cpu().detach().numpy()") + \
            alpha * np.sign(grad)
    elif op == "minus":
        weight = eval("net." + param + ".cpu().detach().numpy()") - \
            alpha * np.sign(grad)
    exec("net." + param + " = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
    return grad


def attack(train_loader, params, load_model_func, norm=True, num_steps=5, alpha=0.00025, op="add"):
    net = load_model_func()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    totals = dict()
    for param in params:
        totals[param] = None
    for _ in tqdm(range(num_steps)):
        total_loss = 0
        num = 0
        for x, y in tqdm(train_loader, total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = loss_func(outputs, y)
            regularization_loss = 0
            for param in net.parameters():
                regularization_loss += torch.norm(param)
            if op == "add":
                loss = loss - 0.01 * regularization_loss
            elif op == "minus":
                loss = loss + 0.01 * regularization_loss
            total_loss = loss.item() + total_loss
            loss.backward()
            num += x.shape[0]
        print(total_loss / num)
        for param in params:
            grad = update_param(net, param, alpha, op=op)
            if totals[param] is None:
                totals[param] = -(alpha * np.sign(grad)) * grad / num
            else:
                totals[param] += -(alpha * np.sign(grad)) * grad / num
        net.zero_grad()
    param_totals = list()
    for param in params:
        param_totals.append(totals[param])
    param_totals = np.array(param_totals,dtype=object)
    if norm:
        param_totals = normalization(np.abs(param_totals))
    for param in params:
        totals[param] = param_totals[params.index(param)]
    return totals
