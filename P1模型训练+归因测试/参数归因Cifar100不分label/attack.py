import torch
from utils import normalization
from models.resnet import load_cifar100_resnet50, load_cifar10_resnet50
from tqdm import tqdm
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


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
            grad = np.array(eval("net." + layer + ".weight.grad.cpu().detach().numpy()"))
            weight = eval("net." + layer + ".weight.cpu().detach().numpy()") + alpha * np.sign(grad)
            exec("net." + layer + ".weight = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
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
