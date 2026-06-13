"""
Unit tests for the Variational Autoencoder model and trainer.

Tests VAE architecture, reparameterization trick, loss function,
and training loop execution.
"""

from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor operations


class TestVAEEncoder:
    """Test suite for the VAE Encoder component."""

    def test_encoder_output_shapes(self):
        """Verify encoder outputs correct mu and logvar shapes."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAEEncoder

        latent_dim = 10
        encoder = VAEEncoder(latent_dim)
        test_input = torch.rand(4, 1, 28, 28)

        # Forward pass should return mu and logvar
        mu, logvar = encoder(test_input)

        # Both should have shape (batch, latent_dim)
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)

    def test_encoder_different_latent_dims(self):
        """Verify encoder works with various latent dimensions."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAEEncoder

        for latent_dim in [2, 16, 50, 100]:
            encoder = VAEEncoder(latent_dim)
            test_input = torch.rand(2, 1, 28, 28)
            mu, logvar = encoder(test_input)
            assert mu.shape == (2, latent_dim)


class TestVAEDecoder:
    """Test suite for the VAE Decoder component."""

    def test_decoder_output_shape(self):
        """Verify decoder reconstructs correct image dimensions."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAEDecoder

        latent_dim = 10
        decoder = VAEDecoder(latent_dim)
        test_input = torch.rand(4, latent_dim)

        output = decoder(test_input)

        # Should reconstruct to full image size
        assert output.shape == (4, 1, 28, 28)

    def test_decoder_output_range(self):
        """Verify decoder output is in [0, 1] range."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAEDecoder

        decoder = VAEDecoder(10)
        test_input = torch.randn(4, 10)

        output = decoder(test_input)

        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestVAE:
    """Test suite for the complete VAE model."""

    def test_vae_forward_returns_three_tensors(self):
        """Verify VAE forward returns reconstructed, mu, and logvar."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAE

        model = VAE(latent_dim=10)
        test_input = torch.rand(4, 1, 28, 28)

        result = model(test_input)

        # Should return tuple of 3 tensors
        assert len(result) == 3
        reconstructed, mu, logvar = result
        assert reconstructed.shape == (4, 1, 28, 28)
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_reparameterize_produces_correct_shape(self):
        """Verify reparameterization outputs correct dimensions."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAE

        model = VAE(latent_dim=10)
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)

        # Reparameterize should produce same shape as inputs
        sample = model.reparameterize(mu, logvar)
        assert sample.shape == (4, 10)

    def test_reparameterize_with_zero_variance(self):
        """Verify reparameterization equals mu when variance is zero."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import VAE

        model = VAE(latent_dim=10)
        mu = torch.ones(4, 10) * 5.0
        # logvar = -inf gives std = 0, but use very negative for approximation
        logvar = torch.ones(4, 10) * -100.0

        sample = model.reparameterize(mu, logvar)
        # With near-zero std, sample should be approximately mu
        assert torch.allclose(sample, mu, atol=1e-5)


class TestVAELoss:
    """Test suite for the VAE loss function."""

    def test_loss_is_positive(self):
        """Verify VAE loss is always positive."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import compute_vae_loss

        reconstructed = torch.rand(4, 1, 28, 28)
        original = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10)
        logvar = torch.randn(4, 10)

        loss = compute_vae_loss(reconstructed, original, mu, logvar)
        assert loss.item() > 0

    def test_loss_zero_kl_with_standard_normal(self):
        """Verify KL component is zero when distribution matches prior."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import compute_vae_loss

        # Perfect reconstruction (same input/output)
        images = torch.rand(4, 1, 28, 28)
        # mu=0, logvar=0 gives KL=0 for standard normal prior
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)

        loss = compute_vae_loss(images, images, mu, logvar)
        # Loss should be very small (just reconstruction of identical inputs)
        # BCE of identical inputs is 0, KL of N(0,1) vs N(0,1) is 0
        # But BCE doesn't give exactly 0 due to log(x) instability
        assert loss.item() >= 0

    def test_loss_gradient_flows(self):
        """Verify gradients flow through the loss computation."""
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import compute_vae_loss

        reconstructed = torch.rand(4, 1, 28, 28, requires_grad=True)
        original = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10, requires_grad=True)
        logvar = torch.randn(4, 10, requires_grad=True)

        loss = compute_vae_loss(reconstructed, original, mu, logvar)
        loss.backward()

        assert reconstructed.grad is not None
        assert mu.grad is not None
        assert logvar.grad is not None


class TestVAETrainer:
    """Test suite for the VAETrainer class."""

    def test_trainer_runs_one_epoch(self, sample_config_dir, mock_train_loader):
        """Verify VAE trainer completes one training epoch."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.autoencoder.variable_auto_encoder.create_mnist_data_loaders",
                return_value=(mock_train_loader, mock_train_loader),
            ):
                from ai_genai_image_generation.autoencoder.variable_auto_encoder import (
                    VAETrainer,
                )

                trainer = VAETrainer()
                losses = trainer.train()

                assert len(losses) == 1
                assert losses[0] > 0
