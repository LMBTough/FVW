from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler


def load_cifar10(root='./data', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root=root, train=True,
                            download=download, transform=transform)
    test_dataset = CIFAR10(root=root, train=False,
                           download=download, transform=transform)
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    test_dataloader_all = DataLoader(
        test_dataset, batch_size=256, shuffle=False)
    num_classes = 10
    train_dataloaders = []
    test_dataloaders = []
    for i in range(num_classes):
        train_dataloaders.append(DataLoader(train_dataset, batch_size=256, shuffle=False,
                                 sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
        test_dataloaders.append(DataLoader(test_dataset, batch_size=256, shuffle=False,
                                sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
    return train_dataloaders, test_dataloaders, test_dataloader_all

def load_cifar10_choosen(root='./data', download=True, choosen_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root=root, train=True,
                            download=download, transform=transform)
    test_dataset = CIFAR10(root=root, train=False,
                            download=download, transform=transform)
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    test_dataloader_all = DataLoader(
        test_dataset, batch_size=256, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                                    sampler=SubsetRandomSampler(np.where(np.isin(train_targets, choosen_classes))[0]))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                    sampler=SubsetRandomSampler(np.where(np.isin(test_targets, choosen_classes))[0]))
    return train_dataloader, test_dataloader, test_dataloader_all


def load_cifar100(root='./data', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR100(root=root, train=True,
                             download=download, transform=transform)
    test_dataset = CIFAR100(root=root, train=False,
                            download=download, transform=transform)
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    test_dataloader_all = DataLoader(
        test_dataset, batch_size=256, shuffle=False)
    num_classes = 100
    train_dataloaders = []
    test_dataloaders = []
    for i in range(num_classes):
        train_dataloaders.append(DataLoader(train_dataset, batch_size=256, shuffle=False,
                                 sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
        test_dataloaders.append(DataLoader(test_dataset, batch_size=256, shuffle=False,
                                sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
    return train_dataloaders, test_dataloaders, test_dataloader_all

def load_cifar100_choosen(root='./data', download=True, choosen_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR100(root=root, train=True,
                             download=download, transform=transform)
    test_dataset = CIFAR100(root=root, train=False,
                            download=download, transform=transform)
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    test_dataloader_all = DataLoader(
        test_dataset, batch_size=256, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                                    sampler=SubsetRandomSampler(np.where(np.isin(train_targets, choosen_classes))[0]))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                    sampler=SubsetRandomSampler(np.where(np.isin(test_targets, choosen_classes))[0]))
    return train_dataloader, test_dataloader, test_dataloader_all