import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset

def get_fashion_mnist(batch_size=128, data_dir='.FashionMNIST/'):
    transform = transforms.Compose([
        transforms.ToTensor(),
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

def get_labeled_data(dataset, num_labels_per_class):
    labels = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
    total_labels = num_labels_per_class * 10  # 10 classes
    
    # Get indices for each class
    labeled_indices = []
    for i in range(10):
        class_indices = np.where(labels == i)[0]
        selected_indices = np.random.choice(class_indices, num_labels_per_class, replace=False)
        labeled_indices.extend(selected_indices)
    
    return Subset(dataset, labeled_indices)

def get_semi_supervised_data(dataset, num_labeled, batch_size=128):
    """
    Split dataset into labeled and unlabeled sets
    """
    labeled_dataset = get_labeled_data(dataset, num_labeled // 10)
    
    # Create indices for unlabeled data
    labeled_indices = labeled_dataset.indices
    all_indices = set(range(len(dataset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    unlabeled_dataset = Subset(dataset, unlabeled_indices)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    return labeled_loader, unlabeled_loader
