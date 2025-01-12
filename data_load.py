import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_fashion_mnist(batch_size=128, data_dir='.FashionMNIST/'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def get_mnist(batch_size=128, data_dir='.MNIST/'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader
