import argparse
from models.resnet import resnet50
from datasets import load_cifar10, load_cifar100, load_caltech_101
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision
import pickle as pkl
from tqdm import tqdm
from attack import attack
from utils import get_device, get_all_param_names, setup_seed, test_model, calculate_pruned, calculate_param_remove, prune_model
import copy
device = get_device()


args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="cifar10")
args.add_argument("--masks", type=str, default=None, help="restore masks")
args.add_argument("--thre", type=float, default=0.98, help="threshold")
args.add_argument("--alpha", type=float, default=0.00000001,
                  help="attack learning rate")
args.add_argument("--num_steps", type=int, default=2, help="attack steps")
args.add_argument("--iters", type=int, default=100, help="prune iters")
args.add_argument("--op", type=str, default="add", help="attack op")
args.add_argument("--labeled", action="store_true", help="use labeled data")
args.add_argument("--sum", action="store_true", help="use sum")


def train_epoch(model, loss_func, train_dataloader, lr, masks):
    model.train()
    num = 0
    for x, y in tqdm(train_dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_func(output, y)
        regularization_loss = 0
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                regularization_loss += torch.norm(param * mask)
            else:
                regularization_loss += torch.norm(param)
        loss = loss + 0.001 * regularization_loss
        loss.backward()
        num += x.shape[0]
        for name, param in model.named_parameters():
            if masks is not None:
                if name in masks.keys():
                    param.grad = param.grad * masks[name]
            param.data -= lr * torch.sign(param.grad)
            param.grad.zero_()
    return model


def prune(model, iters, train_dataloader, all_param_names, hp_param, dataset):
    model = copy.deepcopy(model)
    if hp_param["labeled"]:
        assert isinstance(train_dataloader, list)
    all_totals = list()
    if hp_param["labeled"]:
        for i in range(len(train_dataloader)):
            try:
                net = copy.deepcopy(model)
                totals, class_weight = attack(net, iters, train_dataloader[i], all_param_names, masks=hp_param["masks"],
                                            alpha=hp_param["alpha"], num_steps=hp_param["num_steps"], op=hp_param["op"],dataset=dataset)
                for key in totals:
                    totals[key] = abs(totals[key]/class_weight)
                if iters <= 50 or dataset == "cifar100":
                    net = copy.deepcopy(model)
                    totals_minus, class_weight_minus = attack(
                        net, iters, train_dataloader[i], all_param_names, masks=hp_param["masks"],  alpha=hp_param["alpha"], num_steps=hp_param["num_steps"], op="minus",dataset=dataset)
                    max_minus = -np.inf
                    for key in totals_minus:
                        mx_k = np.max(
                            abs(totals_minus[key]/class_weight_minus))
                        if mx_k > max_minus:
                            max_minus = mx_k

                    for key in totals:
                        totals[key] += (max_minus -
                                        abs(totals_minus[key]/class_weight_minus))

                all_totals.append(totals)
            except:
                pass
    else:
        net = copy.deepcopy(model)
        totals, class_weight = attack(net,iters, train_dataloader, all_param_names, masks=hp_param["masks"], alpha=hp_param[
                                      "alpha"], num_steps=hp_param["num_steps"], op=hp_param["op"],dataset=dataset)
        all_totals.append(totals)
    param_remove = calculate_param_remove(model, all_param_names, all_totals, all_param_masks=hp_param[
                                          "masks"], thre=hp_param["thre"], sum=hp_param["sum"])
    model = prune_model(model, param_remove, all_param_names)
    removed = dict()
    for name in param_remove:
        removed[name] = torch.Tensor(param_remove[name]).to(device)
    return model, removed, param_remove


if __name__ == "__main__":
    args = args.parse_args()
    prefix = f"{args.dataset}_thre_{args.thre}"
    setup_seed(3407)
    if args.dataset == "cifar10":
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_cifar10()
    elif args.dataset == "cifar100":
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_cifar100()
    elif args.dataset == 'caltech101':
        train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all, data_min, data_max = load_caltech_101()

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        model = resnet50(args.dataset)
    elif args.dataset == "caltech101":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 101)
        model.load_state_dict(torch.load(
            "weights/caltech101_resnet50_weights.pth"))
        model.to(device)
        model.eval()

    all_param_names = get_all_param_names(model)

    if args.masks is not None:
        all_param_masks = pkl.load(open(args.masks, "rb"))
        param_remove = dict()
        for key in list(all_param_masks.keys()):
            param_remove[key] = all_param_masks[key].bool()
        model = prune_model(model, param_remove, all_param_names)
    else:
        all_param_masks = dict()
        for name, parameter in model.named_parameters():
            if name in all_param_names:
                all_param_masks[name] = torch.ones_like(parameter)

    hyper_param = {
        "masks": all_param_masks,
        "alpha": [args.alpha, args.alpha],
        "num_steps": args.num_steps,
        "op": args.op,
        "thre": args.thre,
        "labeled": args.labeled,
        "sum": args.sum,
    }

    ori_model = copy.deepcopy(model)

    for iters in range(args.iters):
        if args.labeled:
            model, masks, param_remove = prune(model, iters, train_dataloaders,
                                               all_param_names, hyper_param, args.dataset)
        else:
            model, masks, param_remove = prune(model, iters, train_dataloader_all,
                                               all_param_names, hyper_param, args.dataset)
        hyper_param["masks"] = masks
        acc = test_model(prune_model(ori_model, param_remove,
                         all_param_names), test_dataloader_all)
        ratio = calculate_pruned(model, masks)
        if not os.path.exists(f"masks_{args.dataset}"):
            os.mkdir(f"masks_{args.dataset}")
        pkl.dump(masks, open(
            f"masks_{args.dataset}/masks_{prefix}_{acc}_{ratio}.pkl", "wb"))
