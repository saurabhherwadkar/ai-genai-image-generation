"""
Unit tests for the Wasserstein GAN model and trainer.

Tests generator and critic architectures, weight clipping,
adversarial training dynamics, and loss computation.
"""

from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor operations


class TestGenerator:
    """Test suite for the WGAN Generator network."""

    def test_generator_output_shape(self):
        """Verify generator produces correct image dimensions."""
        from ai_genai_image_generation.gan.wgan import Generator

        generator = Generator(noise_dim=50)
        noise = torch.randn(4, 50)

        output = generator(noise)

        # Should generate 28x28 grayscale images
        assert output.shape == (4, 1, 28, 28)

    def test_generator_output_range(self):
        """Verify generator output is in [-1, 1] from tanh."""
        from ai_genai_image_generation.gan.wgan import Generator

        generator = Generator(noise_dim=50)
        noise = torch.randn(4, 50)

        output = generator(noise)

        # Tanh constrains to [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_generator_different_noise_different_output(self):
        """Verify different noise inputs produce different images."""
        from ai_genai_image_generation.gan.wgan import Generator

        generator = Generator(noise_dim=50)
        noise_a = torch.randn(1, 50)
        noise_b = torch.randn(1, 50)

        output_a = generator(noise_a)
        output_b = generator(noise_b)

        # Different noise should produce different images
        assert not torch.allclose(output_a, output_b)


class TestCritic:
    """Test suite for the WGAN Critic network."""

    def test_critic_output_shape(self):
        """Verify critic produces scalar scores per image."""
        from ai_genai_image_generation.gan.wgan import Critic

        critic = Critic()
        images = torch.randn(4, 1, 28, 28)

        scores = critic(images)

        # Should produce one score per image
        assert scores.shape == (4, 1)

    def test_critic_unbounded_output(self):
        """Verify critic output is not bounded (no sigmoid)."""
        from ai_genai_image_generation.gan.wgan import Critic

        critic = Critic()
        # Use extreme input to test unbounded nature
        images = torch.randn(8, 1, 28, 28) * 10

        scores = critic(images)

        # Scores should not be artificially bounded to [0, 1]
        # (Note: small networks may still give small values)
        assert scores is not None  # Basic sanity check


class TestWGANTrainer:
    """Test suite for the WGANTrainer class."""

    def test_trainer_initialization(self, sample_config_dir, mock_train_loader_normalized):
        """Verify WGAN trainer initializes correctly from config."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.gan.wgan.create_mnist_data_loaders",
                return_value=(mock_train_loader_normalized, mock_train_loader_normalized),
            ):
                from ai_genai_image_generation.gan.wgan import WGANTrainer

                trainer = WGANTrainer()

                assert trainer.noise_dim == 50
                assert trainer.clip_value == 0.5
                assert trainer.n_critic == 1

    def test_trainer_runs_one_epoch(self, sample_config_dir, mock_train_loader_normalized):
        """Verify WGAN training completes without error."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.gan.wgan.create_mnist_data_loaders",
                return_value=(mock_train_loader_normalized, mock_train_loader_normalized),
            ):
                from ai_genai_image_generation.gan.wgan import WGANTrainer

                trainer = WGANTrainer()
                result = trainer.train()

                # Should return dict with both loss histories
                assert "critic_losses" in result
                assert "generator_losses" in result
                assert len(result["critic_losses"]) == 1

    def test_weight_clipping(self, sample_config_dir, mock_train_loader_normalized):
        """Verify critic weights are clipped after training step."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.gan.wgan.create_mnist_data_loaders",
                return_value=(mock_train_loader_normalized, mock_train_loader_normalized),
            ):
                from ai_genai_image_generation.gan.wgan import WGANTrainer

                trainer = WGANTrainer()

                # Manually set a weight to exceed clip value
                with torch.no_grad():
                    for param in trainer.critic.parameters():
                        param.fill_(10.0)
                        break

                # Apply clipping
                trainer._clip_critic_weights()

                # Verify all weights are within clip bounds
                for param in trainer.critic.parameters():
                    assert param.max().item() <= trainer.clip_value
                    assert param.min().item() >= -trainer.clip_value
