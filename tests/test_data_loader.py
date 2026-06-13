"""
Unit tests for the data loading module.

Tests DataLoader creation, transform pipeline construction,
and configuration-driven parameter loading.
"""

from unittest.mock import patch, MagicMock  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor checks


class TestCreateMnistDataLoaders:
    """Test suite for the MNIST data loader factory function."""

    def test_returns_two_loaders(self, sample_config_dir):
        """Verify function returns both train and test loaders."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.data_loader.datasets.MNIST"
            ) as mock_mnist:
                # Mock the MNIST dataset to avoid actual download
                mock_dataset = MagicMock()
                mock_dataset.__len__ = MagicMock(return_value=100)
                mock_mnist.return_value = mock_dataset

                from ai_genai_image_generation.data_loader import (
                    create_mnist_data_loaders,
                )

                train_loader, test_loader = create_mnist_data_loaders(batch_size=32)

                # Both should be DataLoader instances
                assert train_loader is not None
                assert test_loader is not None

    def test_respects_batch_size(self, sample_config_dir):
        """Verify DataLoader uses the specified batch size."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.data_loader.datasets.MNIST"
            ) as mock_mnist:
                mock_dataset = MagicMock()
                mock_dataset.__len__ = MagicMock(return_value=100)
                mock_mnist.return_value = mock_dataset

                from ai_genai_image_generation.data_loader import (
                    create_mnist_data_loaders,
                )

                train_loader, _ = create_mnist_data_loaders(batch_size=16)

                # Verify batch size is set correctly
                assert train_loader.batch_size == 16

    def test_calls_mnist_with_correct_root(self, sample_config_dir):
        """Verify MNIST is loaded from configured data directory."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.data_loader.datasets.MNIST"
            ) as mock_mnist:
                mock_dataset = MagicMock()
                mock_dataset.__len__ = MagicMock(return_value=100)
                mock_mnist.return_value = mock_dataset

                from ai_genai_image_generation.data_loader import (
                    create_mnist_data_loaders,
                )

                create_mnist_data_loaders(batch_size=32)

                # Verify MNIST was called with configured root dir
                calls = mock_mnist.call_args_list
                assert len(calls) == 2  # Train and test
                assert calls[0].kwargs["root"] == "./data"
