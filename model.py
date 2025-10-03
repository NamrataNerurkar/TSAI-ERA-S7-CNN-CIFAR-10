"""Model and training utilities refactored from S7_Final.ipynb.

This module exposes:
- get_transforms()
- get_datasets()
- get_device()
- get_dataloader_args()
- get_dataloaders()
- Net (CIFAR10 model)
- get_optimizer()
- get_scheduler()
- train()
- test()
"""

from __future__ import annotations

# Consolidated imports at top
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2  # noqa: F401 kept for completeness


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"]


def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    alb_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 1), hole_height_range=(16, 16), hole_width_range=(16, 16),
            fill=(0.4914, 0.4822, 0.4465), fill_mask=None, p=0.5
        ),
    ])

    train_transforms = transforms.Compose([
        AlbumentationsTransform(alb_train),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transforms = transforms.Compose([
        AlbumentationsTransform(A.Compose([])), # this is empty, so no augmentation for test; just kept for code clarity & consistency
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transforms, test_transforms


def get_datasets(data_dir: str = "./data", download: bool = True):
    train_t, test_t = get_transforms()
    train_ds = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_t)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=download, transform=test_t)
    return train_ds, test_ds


def get_device(seed: int = 1):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    return device, cuda


def get_dataloader_args(cuda: bool, batch_size_cuda: int = 128, batch_size_cpu: int = 64, num_workers: int = 4, pin_memory: bool = True, shuffle: bool = True):
    if cuda:
        return dict(shuffle=shuffle, batch_size=batch_size_cuda, num_workers=num_workers, pin_memory=pin_memory)
    return dict(shuffle=shuffle, batch_size=batch_size_cpu)


def get_dataloaders(train_ds, test_ds, dataloader_args: dict):
    train_loader = torch.utils.data.DataLoader(train_ds, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **dataloader_args)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.inputblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),  # out: 32x32, RF: 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),  # out: 32x32, RF: 5
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias=False),  # out: 32x32, RF: 5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),  # out: 16x16, RF: 7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False),  # out: 16x16, RF: 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),  # out: 16x16, RF: 11
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),  # out: 16x16, RF: 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),  # out: 8x8, RF: 15
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=False),  # out: 8x8, RF: 15
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),  # out: 8x8, RF: 23
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, bias=False),  # out: 8x8, RF: 23
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),  # out: 4x4, RF: 31
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1, bias=False),  # out: 4x4, RF: 31
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, groups=192, bias=False),  # out: 4x4, RF: 47
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),  # out: 4x4, RF: 47
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=2, groups=192, dilation=2, bias=False),  # out: 4x4, RF: 79
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=1, bias=False),  # out: 4x4, RF: 79
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # out: 1x1, RF: full image
        self.fc = nn.Conv2d(256, 10, 1)  # out: 1x1, RF: full image

    def forward(self, x):
        x = self.inputblock(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def get_optimizer(model: nn.Module, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 1e-4):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer: optim.Optimizer, epochs: int):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def train(model: nn.Module, device, train_loader, optimizer: optim.Optimizer):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100.0 * correct / processed:.2f}')

    avg_loss = running_loss / processed if processed else 0.0
    avg_acc = 100.0 * correct / processed if processed else 0.0
    return avg_loss, avg_acc


def test(model: nn.Module, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total if total else 0.0
    avg_acc = 100.0 * correct / total if total else 0.0
    return avg_loss, avg_acc

