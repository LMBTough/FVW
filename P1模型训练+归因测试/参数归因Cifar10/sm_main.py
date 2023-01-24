# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import pickle as pkl
from attack import attack, test_model,parse_param
import random
import os
file_name = os.path.basename(__file__)
log_name = file_name.split(".")[0] + ".log"
log_file = open(log_name, "w")

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)

# %%
from datasets import load_cifar10, load_cifar100
from models.resnet import load_cifar10_resnet50, load_cifar100_resnet50
model = load_cifar10_resnet50()


# %%
all_param_names = list()
for name, param in model.named_parameters():
    if not "bn" in name and not "shortcut.1" in name:
        all_param_names.append(name)

# %%
all_param_names = all_param_names[:-2]

# %%
train_loaders, test_dataloaders, test_dataloader_all = load_cifar10()
all_totals = list()
for i in range(10):
    all_totals.append(attack(train_loaders[i], all_param_names, load_cifar10_resnet50, alpha=0.0001,num_steps=1,op="add"))


# %%
pkl.dump(all_totals, open("weights/sm_totals.pkl", "wb"))

# %%
thre = 0.25
net = load_cifar10_resnet50()
param_remove = dict()
for param in all_param_names:
    param_remove[param] = None
for i in range(len(all_totals)):
    totals = all_totals[i]
    totals = [totals[param] for param in all_param_names]
    param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()")
                     for param in all_param_names]
    combine = [np.abs(total * weight) for total, weight in zip(totals, param_weights)]
    combine = np.array(combine)
    combine_flatten = np.concatenate([combine_.flatten() for combine_ in combine],axis=0)
    threshold = np.sort(combine_flatten)[::-1][int(len(combine_flatten) * thre)]
    for idx,param in enumerate(all_param_names):
        if param_remove[param] is None:
            param_remove[param] = combine[idx] > threshold
        else:
            t = combine[idx] > threshold
            param_remove[param] = param_remove[param] | t

# %%
temp = 0
all_num = 0
for param in param_remove:
    temp += param_remove[param].sum()
    all_num += param_remove[param].size
    print(param, param_remove[param].mean())

# %%
log_file.write("保留率: " + str(temp / all_num) + "\n")
# temp / all_num

# %%
with torch.no_grad():
    net = load_cifar10_resnet50()
    preds, labels = test_model(net, test_dataloader_all)
    # print("原始准确率", (preds.argmax(-1) == labels).mean())
    log_file.write("原始准确率: " + str((preds.argmax(-1) == labels).mean()) + "\n")

# %%
with torch.no_grad():
    net = load_cifar10_resnet50()
    for param in all_param_names:
        param_ = parse_param(param)
        try:
            exec("net." + param_ + "[~param_remove[param]] = 0")
        except:
            exec("net." + param_ + "[~param_remove[param],:] = 0")
    preds, labels = test_model(net, test_dataloader_all)
    # print("现在准确率", (preds.argmax(-1) == labels).mean())
    log_file.write("现在准确率: " + str((preds.argmax(-1) == labels).mean()) + "\n")

# %%
with torch.no_grad():
    net = load_cifar10_resnet50()
    for param in all_param_names:
        param_ = parse_param(param)
        keep_rate = param_remove[param].sum() / param_remove[param].size
        weight_flatten = eval("net." + param_ + ".cpu().detach().numpy()").flatten()
        threshold = np.sort(weight_flatten)[int(len(weight_flatten) * (1 - keep_rate))]
        try:
            exec("net." + param_ + "[eval('net.' + param_ + '.cpu().detach().numpy()') < threshold] = 0")
        except:
            exec("net." + param_ + "[eval('net.' + param_ + '.cpu().detach().numpy()') < threshold,:] = 0")
    preds, labels = test_model(net, test_dataloader_all)
    # print("对比实验准确率", (preds.argmax(-1) == labels).mean())
    log_file.write("对比实验准确率: " + str((preds.argmax(-1) == labels).mean()) + "\n")


log_file.close()