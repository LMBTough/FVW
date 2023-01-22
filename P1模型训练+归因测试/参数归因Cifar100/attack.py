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

# def attack(train_dataloaders, test_dataloaders, layers, load_model_func, thre=0.2, num_steps=5, alpha=0.00025):
#     totals = dict()
#     loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
#     for i in range(len(train_dataloaders)):
#         totals[i] = dict()
#         for layer in layers:
#             totals[i][layer] = None
#     for i in range(len(train_dataloaders)):
#         for layer in layers:
#             net = load_model_func()
#             total = None
#             for _ in range(num_steps):
#                 total_loss = 0
#                 num = 0
#                 for x, y in tqdm(train_dataloaders[i],total=len(train_dataloaders[i])):
#                     x, y = x.to(device), y.to(device)
#                     outputs = net(x)
#                     loss = loss_func(outputs, y)
#                     total_loss = loss.item()  + total_loss
#                     loss.backward()
#                     num += x.shape[0]
#                 print(total_loss / num)
#                 grad = np.array(eval("net." + layer + ".weight.grad.cpu().detach().numpy()"))
#                 weight = eval("net." + layer + ".weight.cpu().detach().numpy()") + alpha * np.sign(grad)
#                 exec("net." + layer + ".weight = torch.nn.Parameter(torch.from_numpy(weight).to(device))")
#                 net.zero_grad()
#                 if total is None:
#                     total = -(0.01 * np.sign(grad)) * grad / num
#                 else:
#                     total += -(0.01 * np.sign(grad)) * grad / num
#             total = normalization(np.abs(total))
#             totals[i][layer] = total
#         with torch.no_grad():
#             net = load_model_func()
#             for layer in layers:
#                 # combine = np.abs(
#                 #     totals[i][layer] * net.state_dict()[layer].cpu().detach().numpy())
#                 combine = np.abs(totals[i][layer] * eval("net." + layer + ".weight.cpu().detach().numpy()"))
#                 threshold = np.sort(combine.flatten())[
#                     ::-1][int(len(combine.flatten()) * thre)]
#                 # net.state_dict()[layer][combine > threshold] = 0
#                 exec("net." + layer + ".weight[combine > threshold] = 0")
#             correct, total = test_model(net, test_dataloaders[i])
#             print("去掉{}%的权重后, 在第{}类上的准确率为{:.2f}%".format(
#                 thre * 100, i, correct / total * 100))
#             other_correct, other_total = 0, 0
#             for j in range(len(test_dataloaders)):
#                 if j != i:
#                     correct, total = test_model(net, test_dataloaders[j])
#                     other_correct += correct
#                     other_total += total
#             print("去掉{}%的权重后, 在其他类上的准确率为{:.2f}%".format(
#                 thre * 100, other_correct / other_total * 100))
#     return totals
