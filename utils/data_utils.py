import torch
import numpy as np
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import Dataset


class PathMNISTWrapper(Dataset):
    def __init__(self, split, transform=None):
        # Load the original dataset
        self.dataset = PathMNIST(split=split, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img is a PIL image by default in MedMNIST
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label[0]).long()

def load_data():
    # PathMNIST specific normalization values
    # These shift the data to have 0 mean and 1 variance
    norm_mean = [0.7442, 0.5357, 0.7061]
    norm_std = [0.1232, 0.1768, 0.1244]

    # Training augmentation: Helps the model generalize
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Val/Test: No augmentation, just normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_data = PathMNISTWrapper(split='train', transform=train_transform)
    val_data = PathMNISTWrapper(split='val', transform=test_transform)
    test_data = PathMNISTWrapper(split='test', transform=test_transform)

    return train_data, val_data, test_data