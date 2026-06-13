"""
Data loading module for MNIST dataset management.

This module provides factory functions for creating PyTorch DataLoader
instances configured for MNIST training and testing datasets. All
parameters are read from the central configuration file.
"""

import logging  # Logging framework for data loading events

from torch.utils.data import DataLoader  # PyTorch data loading utility
from torchvision import datasets, transforms  # Dataset and transform utilities

from ai_genai_image_generation.config import get_settings  # Configuration access

# Module-level logger for data loading events
logger = logging.getLogger(__name__)


def create_mnist_data_loaders(
    batch_size: int,
    normalize: bool = False,
    resize: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoader instances for the MNIST dataset.

    Builds transform pipelines and constructs DataLoader objects with
    parameters from the application configuration.

    Args:
        batch_size: Number of samples per batch for the data loaders.
        normalize: Whether to apply normalization (mean=0.5, std=0.5).
        resize: Optional image resize dimension (e.g., 32 for 32x32).

    Returns:
        Tuple of (train_loader, test_loader) DataLoader instances.
    """
    # Retrieve data configuration from settings
    settings = get_settings()
    data_root = settings.get("data", "root_dir", default="./data")
    num_workers = settings.get("data", "num_workers", default=4)
    download_flag = settings.get("data", "download", default=True)

    # Build the transform pipeline based on parameters
    transform_list = []  # Accumulate transform operations

    if resize is not None:  # Add resize transform if specified
        transform_list.append(transforms.Resize(resize))

    # Always convert PIL images to tensors (required for PyTorch)
    transform_list.append(transforms.ToTensor())

    if normalize:  # Add normalization if requested
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    # Compose all transforms into a single pipeline
    transform_pipeline = transforms.Compose(transform_list)

    logger.info("Loading MNIST dataset from: %s", data_root)

    # Download and load the MNIST training dataset
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=download_flag,
        transform=transform_pipeline,
    )

    # Download and load the MNIST test dataset
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=download_flag,
        transform=transform_pipeline,
    )

    # Create DataLoader for training data with shuffling enabled
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster host-to-device transfer for CUDA
    )

    # Create DataLoader for test data without shuffling
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Faster host-to-device transfer for CUDA
    )

    logger.info(
        "Data loaders created: train=%d batches, test=%d batches",
        len(train_loader),
        len(test_loader),
    )

    return train_loader, test_loader  # Return both loader instances
