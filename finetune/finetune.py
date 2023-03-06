import torch
import pickle as pkl
import numpy as np
import argparse
from datasets import load_cifar10, load_cifar100, load_caltech_101
from models.resnet import resnet50
import torchvision
from tqdm import tqdm
import torch.nn as nn
import copy
import re
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)


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
    return all_param_names[1:-1]


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


def prune_model(model, param_remove, all_param_names):
    model = copy.deepcopy(model)
    with torch.no_grad():
        for param in all_param_names:
            param_ = parse_param(param)
            try:
                exec("model." + param_ + "[~param_remove[param]] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
            except:
                exec("model." + param_ + "[~param_remove[param],:] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
    return model


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
    return (preds.argmax(axis=1) == labels).mean()


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


def train_epoch(model, loss_func, train_dataloader, lr, masks):
    model.train()
    num = 0
    for x, y in tqdm(train_dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        num += x.shape[0]
        for name, param in model.named_parameters():
            if masks is not None:
                if name in masks.keys():
                    param.grad = param.grad * masks[name]
            param.data -= lr * torch.sign(param.grad)
            param.grad.zero_()
    return model


def train_model(model, train_dataloader, test_dataloader, mask, lr=0.00001, saved_path=None):
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    best_acc = -np.inf
    best_epoch = 0
    best_model = None
    patient = 10
    encountered = 0
    epoch = 0
    while True:
        model = train_epoch(model, loss_func, train_dataloader, lr, mask)
        train_acc = test_model(model, train_dataloader)
        print(f"Epoch {epoch + 1} train acc: {train_acc}")
        test_acc = test_model(model, test_dataloader)
        print(f"Epoch {epoch + 1} test acc: {test_acc}")
        epoch += 1
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            encountered = 0
        else:
            encountered += 1
        if encountered > patient:
            break
    if saved_path is not None:
        torch.save(best_model.state_dict(), saved_path +
                   f"_best_accuracy_{best_acc}_best_epoch_{best_epoch}.pth")


args = argparse.ArgumentParser()
args.add_argument('--masks', type=str)
args.add_argument('--dataset', type=str)
args.add_argument('--finetune', action='store_true')
args.add_argument('--saved_path', type=str)

if __name__ == "__main__":
    args = args.parse_args()
    dataset = args.dataset
    mask_path = args.mask_path
    mask = pkl.load(open(mask_path, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataset == 'cifar10':
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_cifar10()
        model = resnet50("cifar10")
    elif dataset == 'cifar100':
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_cifar100()
        model = resnet50("cifar100")
    elif dataset == 'caltech101':
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_caltech_101()
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 101)
        model.load_state_dict(torch.load(
            "weights/caltech101_resnet50_weights.pth"))
        model.to(device)
        model.eval()
    for key in mask.keys():
        mask[key] = mask[key].bool()
    all_param_names = get_all_param_names(model)
    pruned_model = prune_model(model, mask, all_param_names)
    print("pruned accuarcy: ", test_model(pruned_model, test_dataloader_all))
    print("pruned ratio: ", calculate_pruned(model, mask))
    if args.finetune:
        train_model(pruned_model, train_dataloader_all, test_dataloader_all, mask,
                    saved_path=args.saved_path + f"/{dataset}_pruned_ratio_{calculate_pruned(model, mask):.4f}")
