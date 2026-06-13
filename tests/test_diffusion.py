"""
Unit tests for the Diffusion model (DDPM) and its components.

Tests sinusoidal embeddings, residual blocks, UNet architecture,
forward diffusion process, and training loop.
"""

from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor operations


class TestSinusoidalEmbedding:
    """Test suite for sinusoidal timestep embeddings."""

    def test_embedding_shape(self):
        """Verify embedding produces correct dimensions."""
        from ai_genai_image_generation.diffusion.forward_diffusion import (
            create_sinusoidal_embedding,
        )

        embedding = create_sinusoidal_embedding(100, 64)

        # Should be (num_timesteps, embedding_dim)
        assert embedding.shape == (100, 64)

    def test_embedding_values_bounded(self):
        """Verify sin/cos values are in [-1, 1] range."""
        from ai_genai_image_generation.diffusion.forward_diffusion import (
            create_sinusoidal_embedding,
        )

        embedding = create_sinusoidal_embedding(50, 32)

        assert embedding.min() >= -1.0
        assert embedding.max() <= 1.0

    def test_different_timesteps_different_embeddings(self):
        """Verify distinct timesteps produce distinct embeddings."""
        from ai_genai_image_generation.diffusion.forward_diffusion import (
            create_sinusoidal_embedding,
        )

        embedding = create_sinusoidal_embedding(10, 32)

        # First and last timestep should have different embeddings
        assert not torch.allclose(embedding[0], embedding[-1])


class TestResidualConvBlock:
    """Test suite for the ResidualConvBlock component."""

    def test_non_residual_output_shape(self):
        """Verify non-residual block preserves spatial dimensions."""
        from ai_genai_image_generation.diffusion.forward_diffusion import ResidualConvBlock

        block = ResidualConvBlock(32, 64, is_residual=False)
        test_input = torch.rand(4, 32, 16, 16)

        output = block(test_input)

        # Spatial dims preserved, channels changed
        assert output.shape == (4, 64, 16, 16)

    def test_residual_same_channels(self):
        """Verify residual block with matching channels adds skip."""
        from ai_genai_image_generation.diffusion.forward_diffusion import ResidualConvBlock

        block = ResidualConvBlock(32, 32, is_residual=True)
        test_input = torch.rand(4, 32, 16, 16)

        output = block(test_input)

        # Output should have same shape (skip connection possible)
        assert output.shape == (4, 32, 16, 16)

    def test_residual_different_channels(self):
        """Verify residual block handles channel mismatch."""
        from ai_genai_image_generation.diffusion.forward_diffusion import ResidualConvBlock

        block = ResidualConvBlock(32, 64, is_residual=True)
        test_input = torch.rand(4, 32, 16, 16)

        output = block(test_input)

        assert output.shape == (4, 64, 16, 16)


class TestUNetBlock:
    """Test suite for the UNetBlock component."""

    def test_downsample_block(self):
        """Verify downsampling block halves spatial dimensions."""
        from ai_genai_image_generation.diffusion.forward_diffusion import UNetBlock

        block = UNetBlock(64, 128, time_embedding_dim=64, upsample=False)
        features = torch.rand(4, 64, 16, 16)
        time_emb = torch.rand(4, 64)

        output = block(features, time_emb)

        # Spatial should be halved (16 -> 8)
        assert output.shape == (4, 128, 8, 8)

    def test_upsample_block(self):
        """Verify upsampling block doubles spatial dimensions."""
        from ai_genai_image_generation.diffusion.forward_diffusion import UNetBlock

        block = UNetBlock(64, 32, time_embedding_dim=64, upsample=True)
        # Upsampling path expects 2*in_channels due to skip connection
        features = torch.rand(4, 128, 8, 8)
        time_emb = torch.rand(4, 64)

        output = block(features, time_emb)

        # Spatial should be doubled (8 -> 16)
        assert output.shape == (4, 32, 16, 16)


class TestUNet:
    """Test suite for the complete UNet model."""

    def test_unet_output_shape(self):
        """Verify UNet preserves input spatial dimensions."""
        from ai_genai_image_generation.diffusion.forward_diffusion import UNet

        model = UNet(in_channels=1, out_channels=1, time_embedding_dim=64)
        # Input must be 32x32 for the 4-level UNet
        images = torch.rand(4, 1, 32, 32)
        timesteps = torch.randint(0, 10, (4,))

        output = model(images, timesteps)

        # Output should match input spatial dimensions
        assert output.shape == (4, 1, 32, 32)

    def test_unet_gradient_flow(self):
        """Verify gradients flow through the UNet."""
        from ai_genai_image_generation.diffusion.forward_diffusion import UNet

        model = UNet(in_channels=1, out_channels=1, time_embedding_dim=64)
        images = torch.rand(4, 1, 32, 32)
        timesteps = torch.randint(0, 10, (4,))

        output = model(images, timesteps)
        loss = output.sum()
        loss.backward()

        # Verify at least some parameters have gradients
        has_gradients = any(
            p.grad is not None for p in model.parameters()
        )
        assert has_gradients


class TestDDPM:
    """Test suite for the DDPM forward diffusion process."""

    def test_forward_diffusion_output_shapes(self):
        """Verify forward diffusion returns correct shapes."""
        from ai_genai_image_generation.diffusion.forward_diffusion import DDPM, UNet

        unet = UNet(in_channels=1, out_channels=1, time_embedding_dim=64)
        ddpm = DDPM(unet, num_timesteps=10, device="cpu")

        images = torch.rand(4, 1, 32, 32)
        timesteps = torch.randint(0, 10, (4,))

        noisy_images, noise = ddpm.forward_diffusion(images, timesteps)

        # Both should match input image shape
        assert noisy_images.shape == images.shape
        assert noise.shape == images.shape

    def test_forward_diffusion_adds_noise(self):
        """Verify forward diffusion changes the input images."""
        from ai_genai_image_generation.diffusion.forward_diffusion import DDPM, UNet

        unet = UNet(in_channels=1, out_channels=1, time_embedding_dim=64)
        ddpm = DDPM(unet, num_timesteps=10, device="cpu")

        images = torch.ones(4, 1, 32, 32) * 0.5
        # Use later timestep for more noise
        timesteps = torch.tensor([9, 9, 9, 9])

        noisy_images, _ = ddpm.forward_diffusion(images, timesteps)

        # Noisy images should differ from clean images
        assert not torch.allclose(noisy_images, images)

    def test_noise_schedule_values(self):
        """Verify beta schedule and derived values are valid."""
        from ai_genai_image_generation.diffusion.forward_diffusion import DDPM, UNet

        unet = UNet(in_channels=1, out_channels=1, time_embedding_dim=64)
        ddpm = DDPM(unet, num_timesteps=100, beta_start=0.0001, beta_end=0.02, device="cpu")

        # Betas should be in range [beta_start, beta_end]
        assert ddpm.betas[0].item() == pytest.approx(0.0001, abs=1e-6)
        assert ddpm.betas[-1].item() == pytest.approx(0.02, abs=1e-6)

        # Alphas should be in (0, 1)
        assert (ddpm.alphas > 0).all()
        assert (ddpm.alphas <= 1).all()

        # Cumulative alphas should be monotonically decreasing
        assert (ddpm.alphas_cumprod[1:] <= ddpm.alphas_cumprod[:-1]).all()


class TestDiffusionTrainer:
    """Test suite for the DiffusionTrainer class."""

    def test_trainer_runs_one_epoch(self, sample_config_dir, mock_train_loader_32x32):
        """Verify diffusion trainer completes one epoch."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.diffusion.forward_diffusion.create_mnist_data_loaders",
                return_value=(mock_train_loader_32x32, mock_train_loader_32x32),
            ):
                from ai_genai_image_generation.diffusion.forward_diffusion import (
                    DiffusionTrainer,
                )

                trainer = DiffusionTrainer()
                losses = trainer.train()

                assert len(losses) == 1
                assert losses[0] > 0
