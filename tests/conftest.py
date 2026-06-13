"""
Shared pytest fixtures for the test suite.

Provides common test utilities including temporary configuration,
mock data loaders, and device fixtures that are reused across
all test modules.
"""

import os  # Environment variable manipulation for tests
from pathlib import Path  # Path handling for temp configs
from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor fixtures
from torch.utils.data import DataLoader, TensorDataset  # Mock data loaders


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """
    Reset the Settings singleton before each test.

    Ensures tests get a fresh configuration state and don't
    leak state between test cases.
    """
    # Import Settings class to reset its singleton
    from ai_genai_image_generation.config import Settings

    # Clear the singleton instance before test
    Settings._instance = None
    yield
    # Clear again after test for cleanup
    Settings._instance = None


@pytest.fixture
def sample_config_dir(tmp_path):
    """
    Create a temporary configuration directory with test settings.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Returns:
        Path to the temporary config directory.
    """
    # Create config subdirectory in temp path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Write minimal test configuration YAML
    settings_file = config_dir / "settings.yaml"
    settings_file.write_text(
        """
logging:
  level: "DEBUG"
  format: "%(message)s"
  file: null

data:
  root_dir: "./data"
  num_workers: 0
  download: false

autoencoder:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 1

vae:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 1
  latent_dim: 10

cvae:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 1
  latent_dim: 10
  num_classes: 10
  label_embedding_dim: 20

wgan:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 1
  noise_dim: 50
  clip_value: 0.5
  n_critic: 1
  beta1: 0.5
  beta2: 0.999

diffusion:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 1
  num_timesteps: 10
  beta_start: 0.0001
  beta_end: 0.02
  time_embedding_dim: 64
  image_size: 32

device:
  preferred: "cpu"
""",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_train_loader():
    """
    Create a mock training DataLoader with random MNIST-shaped data.

    Returns:
        DataLoader with 16 random 28x28 images and labels.
    """
    # Generate random image tensors (16 samples, 1 channel, 28x28)
    images = torch.rand(16, 1, 28, 28)
    # Generate random integer labels (0-9)
    labels = torch.randint(0, 10, (16,))
    # Wrap in TensorDataset and DataLoader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return loader


@pytest.fixture
def mock_train_loader_normalized():
    """
    Create a mock DataLoader with normalized data in [-1, 1] range.

    Returns:
        DataLoader with 16 normalized random images and labels.
    """
    # Generate random images in [-1, 1] range for WGAN
    images = torch.rand(16, 1, 28, 28) * 2 - 1
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return loader


@pytest.fixture
def mock_train_loader_32x32():
    """
    Create a mock DataLoader with 32x32 images for diffusion model.

    Returns:
        DataLoader with 16 random 32x32 images and labels.
    """
    # Generate 32x32 images for diffusion model input
    images = torch.rand(16, 1, 32, 32) * 2 - 1
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return loader


@pytest.fixture
def cpu_device():
    """Provide a CPU device for testing without GPU requirement."""
    return torch.device("cpu")
