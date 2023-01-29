from torchvision.datasets import CIFAR10
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
    train_dataloader_all = DataLoader(
        train_dataset, batch_size=256, shuffle=True)
    num_classes = 10
    train_dataloaders = []
    test_dataloaders = []
    for i in range(num_classes):
        train_dataloaders.append(DataLoader(train_dataset, batch_size=256, shuffle=False,
                                 sampler=SubsetRandomSampler(np.where(train_targets == i)[0])))
        test_dataloaders.append(DataLoader(test_dataset, batch_size=256, shuffle=False,
                                sampler=SubsetRandomSampler(np.where(test_targets == i)[0])))
    return train_dataloaders, test_dataloaders, train_dataloader_all, test_dataloader_all
