import torch
from utils import normalization
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

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
        return param
    else:
        for f in finded:
            f = f[1:-1]
            param = param.replace(f".{f}.", f"[{f}].")
    return parse_param(param)


def update_param(net, param, alpha, op="add"):
    # param = "conv1.weight"
    param = parse_param(param)
    grad = np.array(eval("net." + param + ".grad.cpu().detach().numpy()"))
    if op == "add":
        weight = eval("net." + param + ".cpu().detach().numpy()") + \
            alpha * np.sign(grad)
    elif op == "minus":
        weight = eval("net." + param + ".cpu().detach().numpy()") - \
            alpha * np.sign(grad)
    # weight = eval("net." + param + ".cpu().detach().numpy()") + \
    #     alpha * np.sign(grad)
    exec("net." + param + " = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
    return grad


def attack(net,train_loaders, params, train_dataloader_all, num_steps=5, alpha=0.00025, op="add",clz=0):
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    totals = dict()
    for param in params:
        totals[param] = None
    for step in range(num_steps):
        total_loss = 0
        num = 0
        for idx,train_loader in enumerate(train_loaders):
            for x, y in tqdm(train_loader, total=len(train_loader)):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                if idx == clz:
                    loss = loss_func(outputs, y) * 9
                else:
                    loss = -loss_func(outputs, y)
                if idx == clz:
                    total_loss = loss.item() + total_loss
                else:
                    total_loss = -loss.item() + total_loss
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
        # preds, labels = test_model(net, train_dataloader_all)
        # print(np.mean(np.argmax(preds, axis=-1) == labels))
        # print(confusion_matrix(labels, np.argmax(preds, axis=-1)))
    # torch.save(net.state_dict(), f"weights/attacked.pth")
        # # plt.figure()
        # plt.figure()
        # preds, labels = test_model(net, train_loader)
        # clz = labels[0]
        # preds = softmax(preds)
        # plt.hist(preds[:, clz], bins=100)
        # plt.show()
    param_totals = list()
    for param in params:
        param_totals.append(totals[param])
    param_totals = np.array(param_totals)
    param_totals = normalization(np.abs(param_totals))
    for param in params:
        totals[param] = param_totals[params.index(param)]
    return totals
