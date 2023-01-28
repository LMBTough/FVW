
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
from tqdm.notebook import tqdm
from attack import attack, parse_param, test_model
from utils import get_device, caculate_param_remove
import random
device = get_device()

import os
file_name = os.path.basename(__file__)
log_name = file_name.replace("py","log")
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
from datasets import load_cifar10
from models.resnet import load_cifar10_resnet50
model = load_cifar10_resnet50()

# %%
all_param_names = list()
for name, param in model.named_parameters():
    if not "bn" in name and not "shortcut.1" in name:
        all_param_names.append(name)
all_param_names = all_param_names[:-2]
# %%
train_loaders, test_dataloaders, train_dataloader_all, test_dataloader_all = load_cifar10()
all_totals = list()
all_totals.append(attack(train_dataloader_all, all_param_names,
                         load_cifar10_resnet50, norm=True, alpha=0.00001, num_steps=4, op="add"))

# %%
pkl.dump(all_totals, open("weights/totals_0.3.pkl", "wb"))
# all_totals = pkl.load(open("weights/totals.pkl", "rb"))

# %%
thre = 0.3
net = load_cifar10_resnet50()
param_remove = caculate_param_remove(all_param_names, all_totals, net, thre)
# %%
temp = 0
all_num = 0
for param in param_remove:
    temp += param_remove[param].sum()
    all_num += param_remove[param].size
    print(param, param_remove[param].mean())

# %%
log_file.write("保留率: " + str(temp / all_num) + "\n")

# %%
with torch.no_grad():
    net = load_cifar10_resnet50()
    preds, labels = test_model(net, test_dataloader_all)
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
    log_file.write("现在准确率: " + str((preds.argmax(-1) == labels).mean()) + "\n")

# %%
with torch.no_grad():
    net = load_cifar10_resnet50()
    for param in all_param_names:
        param_ = parse_param(param)
        keep_rate = param_remove[param].sum() / param_remove[param].size
        weight_flatten = eval(
            "net." + param_ + ".cpu().detach().numpy()").flatten()
        threshold = np.sort(weight_flatten)[int(
            len(weight_flatten) * (1 - keep_rate))]
        try:
            exec("net." + param_ +
                 "[eval('net.' + param_ + '.cpu().detach().numpy()') < threshold] = 0")
        except:
            exec("net." + param_ +
                 "[eval('net.' + param_ + '.cpu().detach().numpy()') < threshold,:] = 0")
    preds, labels = test_model(net, test_dataloader_all)
    log_file.write("对比实验准确率: " + str((preds.argmax(-1) == labels).mean()) + "\n")

# %%

# %%

log_file.close()
