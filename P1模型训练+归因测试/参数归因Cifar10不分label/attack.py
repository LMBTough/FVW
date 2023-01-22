import torch
from utils import normalization
from tqdm import tqdm
import numpy as np
import re
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_layer(layer):
    reg = re.compile("\.\d+\.")
    finded = reg.findall(layer)
    if len(finded) == 0:
        pass
    else:
        for f in finded:
            f = f[1:-1]
            layer = layer.replace(f".{f}.", f"[{f}].")
    return layer


def update_layer(net, layer, alpha):
    # layer = "conv1.weight"
    layer = parse_layer(layer)
    grad = np.array(eval("net." + layer + ".grad.cpu().detach().numpy()"))
    weight = eval("net." + layer + ".cpu().detach().numpy()") + \
        alpha * np.sign(grad)
    exec("net." + layer + " = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
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


def attack(train_loader,layers,load_model_func,num_steps=5,alpha=0.00025):
    net = load_model_func()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    totals = dict()
    for layer in layers:
        totals[layer] = None
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
        for layer in layers:
            grad = update_layer(net, layer, alpha)
            if totals[layer] is None:
                totals[layer] = -(alpha * np.sign(grad)) * grad / num
            else:
                totals[layer] += -(alpha * np.sign(grad)) * grad / num
        net.zero_grad()
    layer_totals = list()
    for layer in layers:
        layer_totals.append(totals[layer])
    layer_totals = np.array(layer_totals)
    layer_totals = normalization(np.abs(layer_totals))
    for layer in layers:
        totals[layer] = layer_totals[layers.index(layer)]
    return totals