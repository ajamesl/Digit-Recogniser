import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    # Define transforms: convert PIL→Tensor and normalize [0,1]→[-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean & std of MNIST
    ])

    train_ds = datasets.MNIST('data/', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('data/', train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
