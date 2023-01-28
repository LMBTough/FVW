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
model = load_cifar100_resnet50()


# %%
all_param_names = list()
for name, param in model.named_parameters():
    if not "bn" in name and not "shortcut.1" in name:
        all_param_names.append(name)
all_param_names = all_param_names[:-2]

# %%
train_loaders, test_dataloaders, test_dataloader_all = load_cifar100()
all_totals = list()
for i in range(100):
    all_totals.append(attack(train_loaders[i], all_param_names, load_cifar100_resnet50, alpha=0.0001,num_steps=5,op="add"))


# %%
pkl.dump(all_totals, open("weights/high_totals.pkl", "wb"))

# %%
# def choose_one(all_totals,thre,other_thre,choose_class=0):
thre = 0.2
other_thre = 0.2
choose_class = 0
net = load_cifar100_resnet50()
param_remove = dict()
for param in all_param_names:
    param_remove[param] = None
all_classes = list(range(100))
all_classes.remove(choose_class)
all_classes.append(choose_class)
print(all_classes)
# for i in range(len(all_totals)):
for i,class_ in enumerate(all_classes):
    totals = all_totals[class_]
    totals = [totals[param] for param in all_param_names]
    param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()")
                    for param in all_param_names]
    combine = [np.abs(total * weight) for total, weight in zip(totals, param_weights)]
    combine = np.array(combine)
    combine_flatten = np.concatenate([combine_.flatten() for combine_ in combine],axis=0)
    if i == 99:
        threshold = np.sort(combine_flatten)[::-1][int(len(combine_flatten) * thre)]
    else:
        threshold = np.sort(combine_flatten)[::-1][int(len(combine_flatten) * other_thre)]
    for idx,param in enumerate(all_param_names):
        if param_remove[param] is None:
            param_remove[param] = combine[idx] > threshold
        else:
            t = combine[idx] > threshold
            if i == 99:
                param_remove[param] = ~param_remove[param] & t
            else:
                param_remove[param] = param_remove[param] | t

# %%
# # thre = 0.1
# thres = [0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.4]
# net = load_cifar10_resnet50()
# param_remove = dict()
# for param in all_param_names:
#     param_remove[param] = None
# for i in range(len(all_totals)):
#     thre = thres[i]
#     totals = all_totals[i]
#     totals = [totals[param] for param in all_param_names]
#     param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()")
#                      for param in all_param_names]
#     combine = [np.abs(total * weight) for total, weight in zip(totals, param_weights)]
#     combine = np.array(combine)
#     combine_flatten = np.concatenate([combine_.flatten() for combine_ in combine],axis=0)
#     threshold = np.sort(combine_flatten)[::-1][int(len(combine_flatten) * thre)]
#     for idx,param in enumerate(all_param_names):
#         if param_remove[param] is None:
#             param_remove[param] = combine[idx] > threshold
#         else:
#             t = combine[idx] > threshold
#             if i == 9:
#                 param_remove[param] = ~param_remove[param] & t
#             else:
#                 param_remove[param] = param_remove[param] | t

# %%
temp = 0
all_num = 0
for param in param_remove:
    temp += param_remove[param].sum()
    all_num += param_remove[param].size
    print(param, param_remove[param].mean())

# %%
0.2 - temp / all_num

# %%
from sklearn.metrics import confusion_matrix
with torch.no_grad():
    net = load_cifar100_resnet50()
    preds, labels = test_model(net, test_dataloader_all)
    print("原始准确率", (preds.argmax(-1) == labels).mean())
    print(confusion_matrix(labels, preds.argmax(-1)))
    # 输出每个类别的准确率
    for i in range(100):
        print(f"类别{i}准确率", (preds[labels == i].argmax(-1) == i).mean())


# %%
with torch.no_grad():
    net = load_cifar100_resnet50()
    for param in all_param_names:
        param_ = parse_param(param)
        try:
            exec("net." + param_ + "[param_remove[param]] = 0")
        except:
            exec("net." + param_ + "[param_remove[param],:] = 0")
    preds, labels = test_model(net, test_dataloader_all)
    print("现在准确率", (preds.argmax(-1) == labels).mean())
    print(confusion_matrix(labels, preds.argmax(-1)))
    for i in range(100):
        print(f"类别{i}准确率", (preds[labels == i].argmax(-1) == i).mean())

# %%


# %%



