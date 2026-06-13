"""
Unit tests for the Conditional VAE model and trainer.

Tests conditional encoding/decoding, label embedding, generation,
and training with class conditioning.
"""

from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework
import torch  # PyTorch for tensor operations


class TestLabelEmbedder:
    """Test suite for the LabelEmbedder component."""

    def test_embedding_output_shape(self):
        """Verify label embedder produces correct dimensions."""
        from ai_genai_image_generation.conditional.cvae import LabelEmbedder

        embedder = LabelEmbedder(num_classes=10, embedding_dim=50)
        labels = torch.tensor([0, 3, 5, 9])

        embeddings = embedder(labels)

        # Should be (batch, embedding_dim)
        assert embeddings.shape == (4, 50)

    def test_different_labels_produce_different_embeddings(self):
        """Verify distinct labels map to distinct embeddings."""
        from ai_genai_image_generation.conditional.cvae import LabelEmbedder

        embedder = LabelEmbedder(num_classes=10, embedding_dim=50)
        label_a = torch.tensor([0])
        label_b = torch.tensor([5])

        emb_a = embedder(label_a)
        emb_b = embedder(label_b)

        # Different labels should produce different embeddings
        assert not torch.allclose(emb_a, emb_b)


class TestConditionalEncoder:
    """Test suite for the ConditionalEncoder component."""

    def test_encoder_output_shapes(self):
        """Verify conditional encoder outputs correct mu and logvar."""
        from ai_genai_image_generation.conditional.cvae import ConditionalEncoder

        encoder = ConditionalEncoder(
            latent_dim=10, num_classes=10, label_embedding_dim=20
        )
        images = torch.rand(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))

        mu, logvar = encoder(images, labels)

        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)


class TestConditionalDecoder:
    """Test suite for the ConditionalDecoder component."""

    def test_decoder_output_shape(self):
        """Verify conditional decoder reconstructs correct image size."""
        from ai_genai_image_generation.conditional.cvae import ConditionalDecoder

        decoder = ConditionalDecoder(
            latent_dim=10, num_classes=10, label_embedding_dim=20
        )
        latent = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        output = decoder(latent, labels)

        assert output.shape == (4, 1, 28, 28)

    def test_decoder_output_range(self):
        """Verify output pixels in [0, 1] from sigmoid."""
        from ai_genai_image_generation.conditional.cvae import ConditionalDecoder

        decoder = ConditionalDecoder(
            latent_dim=10, num_classes=10, label_embedding_dim=20
        )
        latent = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        output = decoder(latent, labels)

        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestConditionalVAE:
    """Test suite for the complete ConditionalVAE model."""

    def test_forward_returns_three_tensors(self):
        """Verify cVAE forward returns reconstructed, mu, logvar."""
        from ai_genai_image_generation.conditional.cvae import ConditionalVAE

        model = ConditionalVAE(latent_dim=10, num_classes=10, label_embedding_dim=20)
        images = torch.rand(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))

        result = model(images, labels)

        assert len(result) == 3
        reconstructed, mu, logvar = result
        assert reconstructed.shape == (4, 1, 28, 28)
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_generate_produces_images(self):
        """Verify generation produces correctly shaped images."""
        from ai_genai_image_generation.conditional.cvae import ConditionalVAE

        model = ConditionalVAE(latent_dim=10, num_classes=10, label_embedding_dim=20)
        labels = torch.tensor([0, 1, 2, 3])
        device = torch.device("cpu")

        generated = model.generate(labels, device)

        assert generated.shape == (4, 1, 28, 28)
        assert generated.min() >= 0.0
        assert generated.max() <= 1.0

    def test_gradient_flow_with_labels(self):
        """Verify gradients flow through label-conditioned path."""
        from ai_genai_image_generation.conditional.cvae import ConditionalVAE

        model = ConditionalVAE(latent_dim=10, num_classes=10, label_embedding_dim=20)
        images = torch.rand(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))

        reconstructed, mu, logvar = model(images, labels)
        loss = torch.nn.functional.mse_loss(reconstructed, images)
        loss.backward()

        # Verify label embedding has gradients
        for param in model.encoder.label_embedder.parameters():
            assert param.grad is not None


class TestCVAELoss:
    """Test suite for the cVAE loss function."""

    def test_loss_is_positive(self):
        """Verify cVAE loss is always positive."""
        from ai_genai_image_generation.conditional.cvae import compute_cvae_loss

        reconstructed = torch.rand(4, 1, 28, 28)
        original = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10)
        logvar = torch.randn(4, 10)

        loss = compute_cvae_loss(reconstructed, original, mu, logvar)
        assert loss.item() > 0


class TestCVAETrainer:
    """Test suite for the CVAETrainer class."""

    def test_trainer_runs_one_epoch(self, sample_config_dir, mock_train_loader):
        """Verify cVAE trainer completes training without error."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            with patch(
                "ai_genai_image_generation.conditional.cvae.create_mnist_data_loaders",
                return_value=(mock_train_loader, mock_train_loader),
            ):
                from ai_genai_image_generation.conditional.cvae import CVAETrainer

                trainer = CVAETrainer()
                losses = trainer.train()

                assert len(losses) == 1
                assert losses[0] > 0
