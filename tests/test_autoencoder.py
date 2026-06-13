"""
Unit tests for the Autoencoder model and trainer.

Tests model architecture, forward pass shapes, encoder/decoder
isolation, and training loop execution.
"""

from unittest.mock import patch, MagicMock  # Mocking

import pytest  # Test framework
import torch  # PyTorch for tensor operations


class TestEncoder:
    """Test suite for the Encoder component."""

    def test_encoder_output_shape(self):
        """Verify encoder produces correct output dimensions."""
        from ai_genai_image_generation.autoencoder.model import Encoder

        # Create encoder and test input
        encoder = Encoder()
        test_input = torch.rand(4, 1, 28, 28)  # Batch of 4 images

        # Forward pass through encoder
        output = encoder(test_input)

        # Output should be (batch, 64, 1, 1) latent representation
        assert output.shape == (4, 64, 1, 1)

    def test_encoder_single_image(self):
        """Verify encoder handles single image input."""
        from ai_genai_image_generation.autoencoder.model import Encoder

        encoder = Encoder()
        # Single image batch
        test_input = torch.rand(1, 1, 28, 28)

        output = encoder(test_input)
        assert output.shape == (1, 64, 1, 1)


class TestDecoder:
    """Test suite for the Decoder component."""

    def test_decoder_output_shape(self):
        """Verify decoder produces correct image dimensions."""
        from ai_genai_image_generation.autoencoder.model import Decoder

        # Create decoder and test latent input
        decoder = Decoder()
        test_input = torch.rand(4, 64, 1, 1)  # Latent representation

        # Forward pass through decoder
        output = decoder(test_input)

        # Output should be (batch, 1, 28, 28) reconstructed images
        assert output.shape == (4, 1, 28, 28)

    def test_decoder_output_range(self):
        """Verify decoder output is constrained to [0, 1] by sigmoid."""
        from ai_genai_image_generation.autoencoder.model import Decoder

        decoder = Decoder()
        test_input = torch.rand(4, 64, 1, 1)

        output = decoder(test_input)

        # Sigmoid should constrain all values to [0, 1]
        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestAutoencoder:
    """Test suite for the complete Autoencoder model."""

    def test_autoencoder_forward_shape(self):
        """Verify autoencoder preserves input dimensions."""
        from ai_genai_image_generation.autoencoder.model import Autoencoder

        model = Autoencoder()
        test_input = torch.rand(4, 1, 28, 28)

        # Full forward pass: encode then decode
        output = model(test_input)

        # Output shape must match input shape
        assert output.shape == test_input.shape

    def test_autoencoder_output_range(self):
        """Verify autoencoder output pixels are in [0, 1]."""
        from ai_genai_image_generation.autoencoder.model import Autoencoder

        model = Autoencoder()
        test_input = torch.rand(4, 1, 28, 28)

        output = model(test_input)

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_autoencoder_gradient_flow(self):
        """Verify gradients flow through the entire model."""
        from ai_genai_image_generation.autoencoder.model import Autoencoder

        model = Autoencoder()
        test_input = torch.rand(4, 1, 28, 28)

        # Forward pass and compute loss
        output = model(test_input)
        loss = torch.nn.functional.mse_loss(output, test_input)

        # Backward pass
        loss.backward()

        # Verify gradients exist for encoder parameters
        for param in model.encoder.parameters():
            assert param.grad is not None

        # Verify gradients exist for decoder parameters
        for param in model.decoder.parameters():
            assert param.grad is not None


class TestAutoencoderTrainer:
    """Test suite for the AutoencoderTrainer class."""

    def test_trainer_initialization(self, sample_config_dir, mock_train_loader):
        """Verify trainer initializes with correct configuration."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.autoencoder.trainer.create_mnist_data_loaders",
                return_value=(mock_train_loader, mock_train_loader),
            ):
                from ai_genai_image_generation.autoencoder.trainer import (
                    AutoencoderTrainer,
                )

                trainer = AutoencoderTrainer()

                # Verify config values are loaded
                assert trainer.batch_size == 4
                assert trainer.num_epochs == 1
                assert trainer.learning_rate == 0.001

    def test_trainer_single_epoch(self, sample_config_dir, mock_train_loader):
        """Verify training completes one epoch without error."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.autoencoder.trainer.create_mnist_data_loaders",
                return_value=(mock_train_loader, mock_train_loader),
            ):
                from ai_genai_image_generation.autoencoder.trainer import (
                    AutoencoderTrainer,
                )

                trainer = AutoencoderTrainer()
                losses = trainer.train()

                # Should have exactly 1 epoch loss (num_epochs=1)
                assert len(losses) == 1
                # Loss should be a positive number
                assert losses[0] > 0

    def test_trainer_evaluate(self, sample_config_dir, mock_train_loader):
        """Verify evaluation returns a valid loss value."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.autoencoder.trainer.create_mnist_data_loaders",
                return_value=(mock_train_loader, mock_train_loader),
            ):
                from ai_genai_image_generation.autoencoder.trainer import (
                    AutoencoderTrainer,
                )

                trainer = AutoencoderTrainer()
                test_loss = trainer.evaluate()

                # Test loss should be a positive number
                assert test_loss > 0
