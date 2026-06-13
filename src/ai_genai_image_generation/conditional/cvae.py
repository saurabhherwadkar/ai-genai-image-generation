"""
Conditional Variational Autoencoder (cVAE) model and trainer.

This module implements a cVAE that extends the standard VAE by conditioning
both the encoder and decoder on class labels. This enables generating
images of specific digit classes by providing the desired label at
generation time.
"""

import logging  # Logging framework for cVAE events

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network building blocks
import torch.nn.functional as F  # Functional API for activations
import torch.optim as optim  # Optimization algorithms

from ai_genai_image_generation.config import get_settings  # Configuration access
from ai_genai_image_generation.data_loader import create_mnist_data_loaders  # Data
from ai_genai_image_generation.device import get_device  # Device selection

# Module-level logger for cVAE events
logger = logging.getLogger(__name__)


class LabelEmbedder(nn.Module):
    """
    Embeds discrete class labels into continuous vector representations.

    Converts integer class labels into dense vectors that can be
    spatially broadcast and concatenated with image features.
    """

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        """
        Initialize the label embedding layer.

        Args:
            num_classes: Total number of distinct class labels.
            embedding_dim: Dimensionality of the embedding vectors.
        """
        super().__init__()

        # Learnable embedding lookup table for class labels
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        logger.debug(
            "LabelEmbedder initialized: classes=%d, dim=%d",
            num_classes,
            embedding_dim,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Convert integer labels to embedding vectors.

        Args:
            labels: Integer class labels tensor (batch,).

        Returns:
            Embedding vectors (batch, embedding_dim).
        """
        # Look up embedding vectors for each label in the batch
        return self.embedding(labels)


class ConditionalEncoder(nn.Module):
    """
    Encoder conditioned on class labels for the cVAE.

    Concatenates label embeddings as additional channels to the input
    image before encoding to latent distribution parameters.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        label_embedding_dim: int,
    ) -> None:
        """
        Initialize the conditional encoder.

        Args:
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of distinct class labels.
            label_embedding_dim: Dimensionality of label embeddings.
        """
        super().__init__()

        # Store embedding dimension for spatial broadcasting
        self.label_embedding_dim = label_embedding_dim

        # Label embedding layer for class conditioning
        self.label_embedder = LabelEmbedder(num_classes, label_embedding_dim)

        # Convolutional encoder that accepts image + label channels
        self.encoder_conv = nn.Sequential(
            # Input channels = 1 (image) + label_embedding_dim (conditioning)
            nn.Conv2d(1 + label_embedding_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Non-linear activation after first conv
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Non-linear activation after second conv
            nn.Flatten(),  # Flatten for fully connected layers
        )

        # Fully connected layer for mean of latent distribution
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)

        # Fully connected layer for log-variance of latent distribution
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        logger.debug("ConditionalEncoder initialized: latent_dim=%d", latent_dim)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images conditioned on labels to latent distribution.

        Args:
            images: Input images (batch, 1, 28, 28).
            labels: Class labels (batch,).

        Returns:
            Tuple of (mu, logvar) distribution parameters.
        """
        # Get label embeddings and reshape for spatial broadcasting
        label_emb = self.label_embedder(labels)
        # Reshape to (batch, embedding_dim, 1, 1) for broadcasting
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)
        # Expand to match image spatial dimensions (28x28)
        label_emb = label_emb.expand(-1, -1, images.size(2), images.size(3))

        # Concatenate label channels with image along channel dimension
        conditioned_input = torch.cat([images, label_emb], dim=1)

        # Pass through convolutional encoder
        features = self.encoder_conv(conditioned_input)

        # Compute distribution parameters
        mu = self.fc_mu(features)  # Latent mean
        logvar = self.fc_logvar(features)  # Latent log-variance

        return mu, logvar


class ConditionalDecoder(nn.Module):
    """
    Decoder conditioned on class labels for the cVAE.

    Concatenates label embeddings with latent vectors before decoding
    to generate class-specific reconstructions.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        label_embedding_dim: int,
    ) -> None:
        """
        Initialize the conditional decoder.

        Args:
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of distinct class labels.
            label_embedding_dim: Dimensionality of label embeddings.
        """
        super().__init__()

        # Label embedding for decoder conditioning
        self.label_embedder = LabelEmbedder(num_classes, label_embedding_dim)

        # Fully connected input layer taking latent + label embedding
        self.decoder_input = nn.Linear(
            latent_dim + label_embedding_dim, 64 * 7 * 7
        )

        # Transposed convolutional layers for spatial upsampling
        self.decoder_conv = nn.Sequential(
            # First deconv: 64 -> 32 channels, doubles spatial (7->14)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Non-linear activation
            # Second deconv: 32 -> 1 channel, doubles spatial (14->28)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Constrain pixels to [0, 1]
        )

        logger.debug("ConditionalDecoder initialized: latent_dim=%d", latent_dim)

    def forward(
        self, latent_vector: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent vector conditioned on class label.

        Args:
            latent_vector: Sampled latent vector (batch, latent_dim).
            labels: Class labels (batch,).

        Returns:
            Generated image tensor (batch, 1, 28, 28).
        """
        # Get label embeddings for conditioning
        label_emb = self.label_embedder(labels)

        # Concatenate latent vector with label embedding
        conditioned_latent = torch.cat([latent_vector, label_emb], dim=1)

        # Project to spatial feature map size
        features = self.decoder_input(conditioned_latent)
        # Reshape flat features to 3D spatial format
        features = features.view(-1, 64, 7, 7)

        # Upsample through transposed convolutions
        generated = self.decoder_conv(features)
        return generated


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for class-specific generation.

    Extends the standard VAE by conditioning both encoder and decoder
    on class labels, enabling targeted generation of specific digits.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        num_classes: int = 10,
        label_embedding_dim: int = 50,
    ) -> None:
        """
        Initialize the Conditional VAE.

        Args:
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of class categories (10 for MNIST).
            label_embedding_dim: Size of label embedding vectors.
        """
        super().__init__()

        # Store dimensions for reparameterization
        self.latent_dim = latent_dim

        # Conditional encoder with label conditioning
        self.encoder = ConditionalEncoder(
            latent_dim, num_classes, label_embedding_dim
        )

        # Conditional decoder with label conditioning
        self.decoder = ConditionalDecoder(
            latent_dim, num_classes, label_embedding_dim
        )

        logger.info(
            "ConditionalVAE initialized: latent=%d, classes=%d",
            latent_dim,
            num_classes,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply reparameterization trick for differentiable sampling.

        Args:
            mu: Mean of latent distribution (batch, latent_dim).
            logvar: Log-variance of latent distribution (batch, latent_dim).

        Returns:
            Sampled latent vector (batch, latent_dim).
        """
        # Compute standard deviation from log-variance
        std = torch.exp(0.5 * logvar)
        # Sample noise from standard normal
        epsilon = torch.randn_like(std)
        # Reparameterize: z = mu + std * epsilon
        return mu + epsilon * std

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode with label, sample, decode with label.

        Args:
            images: Input images (batch, 1, 28, 28).
            labels: Class labels (batch,).

        Returns:
            Tuple of (reconstructed_images, mu, logvar).
        """
        # Encode images conditioned on labels
        mu, logvar = self.encoder(images, labels)

        # Sample from the latent distribution
        latent_sample = self.reparameterize(mu, logvar)

        # Decode with label conditioning
        reconstructed = self.decoder(latent_sample, labels)

        return reconstructed, mu, logvar

    def generate(
        self, labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Generate new images for given class labels.

        Samples from the standard normal prior and decodes with
        the specified labels.

        Args:
            labels: Target class labels (batch,).
            device: Device to create tensors on.

        Returns:
            Generated images (batch, 1, 28, 28).
        """
        # Sample latent vectors from the standard normal prior
        latent_samples = torch.randn(labels.size(0), self.latent_dim).to(device)

        # Decode with the provided labels
        with torch.no_grad():  # No gradients needed for generation
            generated_images = self.decoder(latent_samples, labels)

        return generated_images


def compute_cvae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cVAE loss (same formulation as standard VAE loss).

    Combines binary cross-entropy reconstruction loss with KL divergence
    regularization to train the conditional generative model.

    Args:
        reconstructed: Decoder output (batch, 1, 28, 28).
        original: Original images (batch, 1, 28, 28).
        mu: Encoder mean output (batch, latent_dim).
        logvar: Encoder log-variance output (batch, latent_dim).

    Returns:
        Combined scalar loss value.
    """
    # Reconstruction loss: pixel-wise binary cross-entropy
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed, original, reduction="sum"
    )

    # KL divergence regularization toward standard normal
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss is sum of reconstruction and regularization
    return reconstruction_loss + kl_divergence


class CVAETrainer:
    """
    Trainer class for the Conditional Variational Autoencoder.

    Manages training loop with label conditioning, loss computation,
    and optimization with configuration-driven hyperparameters.
    """

    def __init__(self) -> None:
        """Initialize cVAE trainer with model, optimizer, and data."""
        # Load hyperparameters from configuration
        settings = get_settings()
        self.batch_size: int = settings.get("cvae", "batch_size", default=128)
        self.learning_rate: float = settings.get("cvae", "learning_rate", default=0.001)
        self.num_epochs: int = settings.get("cvae", "num_epochs", default=20)
        self.latent_dim: int = settings.get("cvae", "latent_dim", default=20)
        self.num_classes: int = settings.get("cvae", "num_classes", default=10)
        self.label_embedding_dim: int = settings.get(
            "cvae", "label_embedding_dim", default=50
        )

        # Select compute device
        self.device = get_device()

        # Initialize cVAE model with configured dimensions
        self.model = ConditionalVAE(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            label_embedding_dim=self.label_embedding_dim,
        ).to(self.device)

        # Adam optimizer for parameter updates
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Create MNIST data loaders (no normalization for BCE loss)
        self.train_loader, self.test_loader = create_mnist_data_loaders(
            batch_size=self.batch_size
        )

        logger.info(
            "CVAETrainer initialized: latent=%d, epochs=%d",
            self.latent_dim,
            self.num_epochs,
        )

    def train(self) -> list[float]:
        """
        Execute the full cVAE training loop.

        Returns:
            List of average loss values per epoch.
        """
        epoch_losses: list[float] = []  # Track training progress

        logger.info("Starting cVAE training for %d epochs", self.num_epochs)

        for epoch in range(self.num_epochs):
            # Train one epoch with label conditioning
            average_loss = self._train_one_epoch(epoch)
            epoch_losses.append(average_loss)

            logger.info(
                "Epoch [%d/%d], Average Loss: %.4f",
                epoch + 1,
                self.num_epochs,
                average_loss,
            )

        logger.info("cVAE training complete. Final loss: %.4f", epoch_losses[-1])
        return epoch_losses

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train the cVAE for a single epoch with label conditioning.

        Args:
            epoch: Current epoch index for logging.

        Returns:
            Average per-sample loss for this epoch.
        """
        # Set model to training mode
        self.model.train()

        total_loss = 0.0  # Accumulate total epoch loss
        total_samples = 0  # Track sample count for averaging

        for batch_data in self.train_loader:
            # Unpack both images AND labels (labels used for conditioning)
            images, labels = batch_data
            images = images.to(self.device)  # Move images to device
            labels = labels.to(self.device)  # Move labels to device

            # Clear previous gradients
            self.optimizer.zero_grad()

            # Forward pass with label conditioning
            reconstructed, mu, logvar = self.model(images, labels)

            # Compute cVAE loss
            loss = compute_cvae_loss(reconstructed, images, mu, logvar)

            # Backward pass and parameter update
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update parameters

            # Accumulate loss and sample count
            total_loss += loss.item()
            total_samples += images.size(0)

        # Return per-sample average loss
        return total_loss / total_samples


def run_cvae() -> None:
    """
    Entry point function to train the Conditional VAE.

    Creates a trainer and executes the full training loop.
    """
    try:
        trainer = CVAETrainer()  # Initialize trainer
        trainer.train()  # Run training
    except RuntimeError as runtime_error:
        logger.error("Runtime error during cVAE training: %s", runtime_error)
        raise
    except Exception as unexpected_error:
        logger.error("Unexpected error during cVAE training: %s", unexpected_error)
        raise
