import pandas as pd
import pickle as pkl
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,TensorDataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetDataset(Dataset):
    def __init__(self, mode="train", transform=None) -> None:
        super().__init__()
        assert mode in ["train", "val", "test"]
        self.data = pd.read_csv(f'data/mini-imagenet/{mode}.csv')
        self.data = self.data.groupby('label').apply(
            lambda x: x.sample(300)).reset_index(drop=True)
        self.transform = transform
        self.class_name2label = pkl.load(
            open('data/class_name2label.pkl', 'rb'))
        self.classes = list(self.class_name2label.keys())
        if not os.path.exists(f'data/mini-imagenet/{mode}_imgs.pt') or not os.path.exists(f'data/mini-imagenet/{mode}_labels.pt'):
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
                self.imgs, f'data/mini-imagenet/{mode}_imgs.pt')
            torch.save(
                self.labels, f'data/mini-imagenet/{mode}_labels.pt')
        else:
            self.imgs = torch.load(
                f'data/mini-imagenet/{mode}_imgs.pt')
            self.labels = torch.load(
                f'data/mini-imagenet/{mode}_labels.pt')

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def load_imagenet(root='./data'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = ImageNetDataset(mode="train", transform=transform)
    val_dataset = ImageNetDataset(mode="val", transform=transform)
    test_dataset = ImageNetDataset(mode="test", transform=transform)
    train_targets = np.array(train_dataset.labels)
    val_targets = np.array(val_dataset.labels)
    test_targets = np.array(test_dataset.labels)
    all_classes = np.unique(train_targets)
    all_classes = np.append(all_classes, np.unique(val_targets))
    all_classes = np.append(all_classes, np.unique(test_targets))
    train_dataloaders = []
    test_dataloaders = []
    for clz in all_classes:
        if clz in train_targets:
            train_dataloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=False,
                                                sampler=SubsetRandomSampler(np.where(train_targets == clz)[0][:240])))
            test_dataloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=False,
                                               sampler=SubsetRandomSampler(np.where(train_targets == clz)[0][240:])))
        if clz in val_targets:
            train_dataloaders.append(DataLoader(val_dataset, batch_size=64, shuffle=False,
                                                sampler=SubsetRandomSampler(np.where(val_targets == clz)[0][:240])))
            test_dataloaders.append(DataLoader(val_dataset, batch_size=64, shuffle=False,
                                               sampler=SubsetRandomSampler(np.where(val_targets == clz)[0][240:])))
        if clz in test_targets:
            train_dataloaders.append(DataLoader(test_dataset, batch_size=64, shuffle=False,
                                                sampler=SubsetRandomSampler(np.where(test_targets == clz)[0][:240])))
            test_dataloaders.append(DataLoader(test_dataset, batch_size=64, shuffle=False,
                                               sampler=SubsetRandomSampler(np.where(test_targets == clz)[0][240:])))
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    for train_dataloader in train_dataloaders:
        for X, y in train_dataloader:
            X_train.append(X)
            y_train.append(y)
    for test_dataloader in test_dataloaders:
        for X, y in test_dataloader:
            X_test.append(X)
            y_test.append(y)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    train_dataloader_all = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_dataloader_all = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=True)
    return train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all

    # num_classes = 100
    # train_dataloaders = []
    # test_dataloaders = []
    # for i in range(num_classes):
    # train_dataloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=False,
    #  sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
    # test_dataloaders.append(DataLoader(test_dataset, batch_size=64, shuffle=False,
    # sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
    # return train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all
