# from torchvision.datasets import CIFAR100
# from torchvision import transforms
# import numpy as np
# from torch.utils.data import DataLoader, SubsetRandomSampler


# def load_cifar100(root='./data', download=True):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     train_dataset = CIFAR100(root=root, train=True,
#                             download=download, transform=transform)
#     test_dataset = CIFAR100(root=root, train=False,
#                            download=download, transform=transform)
#     train_targets = np.array(train_dataset.targets)
#     test_targets = np.array(test_dataset.targets)
#     test_dataloader_all = DataLoader(
#         test_dataset, batch_size=256, shuffle=False)
#     train_dataloader_all = DataLoader(
#         train_dataset, batch_size=256, shuffle=True)
#     num_classes = 100
#     train_dataloaders = []
#     test_dataloaders = []
#     for i in range(num_classes):
#         train_dataloaders.append(DataLoader(train_dataset, batch_size=256, shuffle=False,
#                                  sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
#         test_dataloaders.append(DataLoader(test_dataset, batch_size=256, shuffle=False,
#                                 sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
#     return train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all

import pandas as pd
import pickle as pkl
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetDataset(Dataset):
    def __init__(self, train, transform) -> None:
        super().__init__()
        if train:
            flag = 'train'
        else:
            flag = 'test'
        self.data = pd.read_csv(f'data/mini-imagenet/{flag}.csv')
        self.transform = transform
        self.class_name2label = pkl.load(
            open('data/class_name2label.pkl', 'rb'))
        self.classes = list(self.class_name2label.keys())
        if not os.path.exists(f'data/mini-imagenet/{flag}_imgs.pt') or not os.path.exists(f'data/mini-imagenet/{flag}_labels.pt'):
            self.imgs = list()
            self.labels = list()
            for i, row in tqdm(self.data.iterrows(), total=len(self.data)):
                img = pil_loader(
                    "data/mini-imagenet/images/" + row['filename'])
                img = self.transform(img)
                self.imgs.append(img)
                self.labels.append(int(self.class_name2label[row['label']]))
            self.imgs = torch.stack(self.imgs)
            self.labels = torch.LongTensor(self.labels)
            torch.save(
                self.imgs, f'data/mini-imagenet/{flag}_imgs.pt')
            torch.save(
                self.labels, f'data/mini-imagenet/{flag}_labels.pt')
        else:
            self.imgs = torch.load(
                f'data/mini-imagenet/{flag}_imgs.pt')
            self.labels = torch.load(
                f'data/mini-imagenet/{flag}_labels.pt')

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def load_imagenet(root='./data'):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = ImageNetDataset(train=True, transform=transform)
    test_dataset = ImageNetDataset(train=False, transform=transform)
    train_targets = np.array(train_dataset.labels)
    test_targets = np.array(test_dataset.labels)
    test_dataloader_all = DataLoader(
        test_dataset, batch_size=256, shuffle=False)
    train_dataloader_all = DataLoader(
        train_dataset, batch_size=256, shuffle=True)
    num_classes = 100
    train_dataloaders = []
    test_dataloaders = []
    for i in range(num_classes):
        train_dataloaders.append(DataLoader(train_dataset, batch_size=256, shuffle=False,
                                 sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
        test_dataloaders.append(DataLoader(test_dataset, batch_size=256, shuffle=False,
                                sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
    return train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all