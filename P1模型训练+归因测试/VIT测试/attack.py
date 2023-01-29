import torch
from utils import normalization
from tqdm import tqdm
import numpy as np
import re
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def update_param(net, param, alpha):
    # param = "conv1.weight"
    param = parse_param(param)
    grad = np.array(eval("net." + param + ".grad.cpu().detach().numpy()"))
    weight = eval("net." + param + ".cpu().detach().numpy()") + \
        alpha * np.sign(grad)
    exec("net." + param + " = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
    return grad


def test_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            predicted = outputs.argmax(dim=-1)
            total += len(x)
            correct += (predicted == y).sum().item()
    return correct, total


def attack(train_loader,params,load_model_func,num_steps=5,alpha=0.00025):
    net = load_model_func()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    totals = dict()
    for param in params:
        totals[param] = None
    for _ in range(num_steps):
        total_loss = 0
        num = 0
        for x, y in tqdm(train_loader,total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = loss_func(outputs, y)
            total_loss = loss.item() + total_loss
            loss.backward()
            num += x.shape[0]
        print(total_loss / num)
        for param in params:
            grad = update_param(net, param, alpha)
            if totals[param] is None:
                totals[param] = -(alpha * np.sign(grad)) * grad / num
            else:
                totals[param] += -(alpha * np.sign(grad)) * grad / num
        net.zero_grad()
    param_totals = list()
    for param in params:
        param_totals.append(totals[param])
    param_totals = np.array(param_totals)
    param_totals = normalization(np.abs(param_totals))
    for param in params:
        totals[param] = param_totals[params.index(param)]
    return totals