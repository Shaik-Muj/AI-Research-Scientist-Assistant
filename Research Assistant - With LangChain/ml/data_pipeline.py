"""
Data pipeline management for common ML datasets.
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np


def get_mnist_loaders(batch_size=32, data_dir="./datasets", val_split=0.1):
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size
        data_dir: Directory to store/load data
        val_split: Validation split ratio
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Split train into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(batch_size=32, data_dir="./datasets", val_split=0.1):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size
        data_dir: Directory to store/load data
        val_split: Validation split ratio
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    # Split train into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


class SequenceDataset(Dataset):
    """Simple sequence dataset for testing RNNs."""
    
    def __init__(self, num_samples=1000, seq_length=20, num_classes=10):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # Generate random sequences
        self.data = torch.randn(num_samples, seq_length, 1)
        
        # Generate labels based on sequence sum
        seq_sums = self.data.sum(dim=1).squeeze()
        self.labels = (seq_sums > 0).long()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_sequence_loaders(batch_size=32, num_samples=1000, seq_length=20):
    """
    Get synthetic sequence data loaders for RNN testing.
    
    Args:
        batch_size: Batch size
        num_samples: Number of samples
        seq_length: Sequence length
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SequenceDataset(num_samples, seq_length)
    val_dataset = SequenceDataset(num_samples // 5, seq_length)
    test_dataset = SequenceDataset(num_samples // 5, seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_data_loaders(dataset_name: str, batch_size: int = 32, **kwargs):
    """
    Get data loaders for a specified dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'sequence')
        batch_size: Batch size
        **kwargs: Additional arguments for specific datasets
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(batch_size, **kwargs)
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size, **kwargs)
    elif dataset_name == 'sequence':
        return get_sequence_loaders(batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
