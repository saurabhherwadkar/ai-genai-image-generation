"""
Variational Autoencoder (VAE) model and trainer.

This module implements a VAE that learns a probabilistic latent space
representation. Unlike a standard autoencoder, the VAE encodes inputs
as distributions (mean and variance) and uses the reparameterization
trick for backpropagation through sampling.
"""

import logging  # Logging framework for VAE events

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network building blocks
import torch.nn.functional as F  # Functional API for activations
import torch.optim as optim  # Optimization algorithms

from ai_genai_image_generation.config import get_settings  # Configuration access
from ai_genai_image_generation.data_loader import create_mnist_data_loaders  # Data
from ai_genai_image_generation.device import get_device  # Device selection

# Module-level logger for VAE events
logger = logging.getLogger(__name__)


class VAEEncoder(nn.Module):
    """
    Probabilistic encoder that maps images to a latent distribution.

    Outputs mean (mu) and log-variance (logvar) parameters that define
    a Gaussian distribution in the latent space.
    """

    def __init__(self, latent_dim: int) -> None:
        """
        Initialize the VAE encoder network.

        Args:
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()

        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            # First conv: 1 -> 16 channels, halves spatial (28->14)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Non-linear activation
            # Second conv: 16 -> 32 channels, halves spatial (14->7)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Non-linear activation
            nn.Flatten(),  # Flatten spatial dims for fully connected layers
        )

        # Fully connected layer to intermediate representation
        self.fc_hidden = nn.Linear(32 * 7 * 7, 128)

        # Output layer for latent mean (mu) parameter
        self.fc_mu = nn.Linear(128, latent_dim)

        # Output layer for latent log-variance parameter
        self.fc_logvar = nn.Linear(128, latent_dim)

        logger.debug("VAEEncoder initialized with latent_dim=%d", latent_dim)

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent distribution parameters.

        Args:
            input_tensor: Batch of images (batch, 1, 28, 28).

        Returns:
            Tuple of (mu, logvar) tensors, each shape (batch, latent_dim).
        """
        # Extract spatial features through conv layers
        features = self.conv_layers(input_tensor)
        # Transform to intermediate hidden representation
        hidden = F.relu(self.fc_hidden(features))
        # Compute mean of the latent Gaussian
        mu = self.fc_mu(hidden)
        # Compute log-variance of the latent Gaussian
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder that reconstructs images from latent space samples.

    Maps latent vectors back to image space using fully connected
    layers followed by transposed convolutions.
    """

    def __init__(self, latent_dim: int) -> None:
        """
        Initialize the VAE decoder network.

        Args:
            latent_dim: Dimensionality of the latent space input.
        """
        super().__init__()

        # Fully connected layers to reshape latent to spatial features
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Expand latent to hidden dim
            nn.ReLU(),  # Non-linear activation
            nn.Linear(128, 32 * 7 * 7),  # Expand to spatial feature size
            nn.ReLU(),  # Non-linear activation
            nn.Unflatten(1, (32, 7, 7)),  # Reshape to 3D feature map
        )

        # Transposed conv layers for spatial upsampling
        self.conv_transpose_layers = nn.Sequential(
            # First deconv: 32 -> 16 channels, doubles spatial (7->14)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Non-linear activation
            # Second deconv: 16 -> 1 channel, doubles spatial (14->28)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Constrain pixel values to [0, 1]
        )

        logger.debug("VAEDecoder initialized with latent_dim=%d", latent_dim)

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector to reconstruct an image.

        Args:
            latent_vector: Sampled latent vector (batch, latent_dim).

        Returns:
            Reconstructed image tensor (batch, 1, 28, 28).
        """
        # Transform latent vector to spatial feature map
        features = self.fc_layers(latent_vector)
        # Upsample features to full image resolution
        reconstructed = self.conv_transpose_layers(features)
        return reconstructed


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder, reparameterization, and decoder.

    The VAE learns a continuous, structured latent space by encoding
    inputs as distributions and training with a combined reconstruction
    and KL divergence loss.
    """

    def __init__(self, latent_dim: int) -> None:
        """
        Initialize the VAE with encoder and decoder.

        Args:
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()

        # Probabilistic encoder producing distribution parameters
        self.encoder = VAEEncoder(latent_dim)
        # Deterministic decoder reconstructing from latent samples
        self.decoder = VAEDecoder(latent_dim)

        logger.info("VAE model initialized with latent_dim=%d", latent_dim)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the reparameterization trick for backprop through sampling.

        Samples from N(mu, sigma^2) using: z = mu + sigma * epsilon,
        where epsilon ~ N(0, 1). This allows gradients to flow through
        the sampling operation.

        Args:
            mu: Mean of the latent distribution (batch, latent_dim).
            logvar: Log-variance of the latent distribution (batch, latent_dim).

        Returns:
            Sampled latent vector (batch, latent_dim).
        """
        # Compute standard deviation from log-variance
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from standard normal distribution
        epsilon = torch.randn_like(std)
        # Apply reparameterization: z = mu + std * epsilon
        latent_sample = mu + epsilon * std
        return latent_sample

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform full VAE forward pass: encode, sample, decode.

        Args:
            input_tensor: Batch of input images (batch, 1, 28, 28).

        Returns:
            Tuple of (reconstructed_images, mu, logvar).
        """
        # Encode input to distribution parameters
        mu, logvar = self.encoder(input_tensor)
        # Sample from the latent distribution
        latent_sample = self.reparameterize(mu, logvar)
        # Decode the sample back to image space
        reconstructed = self.decoder(latent_sample)
        return reconstructed, mu, logvar


def compute_vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the VAE loss combining reconstruction and KL divergence.

    The loss has two components:
    1. Reconstruction loss (BCE): How well the decoder reconstructs the input.
    2. KL divergence: How close the learned distribution is to N(0, 1).

    Args:
        reconstructed: Decoder output images (batch, 1, 28, 28).
        original: Original input images (batch, 1, 28, 28).
        mu: Encoder mean output (batch, latent_dim).
        logvar: Encoder log-variance output (batch, latent_dim).

    Returns:
        Combined scalar loss value.
    """
    # Binary cross-entropy reconstruction loss (sum over all pixels)
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed, original, reduction="sum"
    )

    # KL divergence: D_KL(q(z|x) || p(z)) for Gaussian prior
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Sum both loss components for total VAE objective
    total_loss = reconstruction_loss + kl_divergence
    return total_loss


class VAETrainer:
    """
    Trainer class for the Variational Autoencoder.

    Manages training loop, loss computation, and optimization
    with all hyperparameters loaded from configuration.
    """

    def __init__(self) -> None:
        """Initialize VAE trainer with model, optimizer, and data."""
        # Load hyperparameters from configuration
        settings = get_settings()
        self.batch_size: int = settings.get("vae", "batch_size", default=32)
        self.learning_rate: float = settings.get("vae", "learning_rate", default=0.001)
        self.num_epochs: int = settings.get("vae", "num_epochs", default=15)
        self.latent_dim: int = settings.get("vae", "latent_dim", default=50)

        # Select compute device
        self.device = get_device()

        # Initialize VAE model and move to device
        self.model = VAE(self.latent_dim).to(self.device)

        # Adam optimizer for parameter updates
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Create data loaders for MNIST
        self.train_loader, self.test_loader = create_mnist_data_loaders(
            batch_size=self.batch_size
        )

        logger.info(
            "VAETrainer initialized: latent_dim=%d, epochs=%d, lr=%f",
            self.latent_dim,
            self.num_epochs,
            self.learning_rate,
        )

    def train(self) -> list[float]:
        """
        Execute the full VAE training loop.

        Returns:
            List of average loss values per epoch.
        """
        epoch_losses: list[float] = []  # Track loss per epoch

        logger.info("Starting VAE training for %d epochs", self.num_epochs)

        for epoch in range(self.num_epochs):
            # Train single epoch and record average loss
            average_loss = self._train_one_epoch(epoch)
            epoch_losses.append(average_loss)

            logger.info(
                "Epoch [%d/%d], Average Loss: %.4f",
                epoch + 1,
                self.num_epochs,
                average_loss,
            )

        logger.info("VAE training complete. Final loss: %.4f", epoch_losses[-1])
        return epoch_losses

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train the VAE for a single epoch.

        Args:
            epoch: Current epoch index for logging.

        Returns:
            Average loss normalized by dataset size.
        """
        # Set model to training mode
        self.model.train()

        total_loss = 0.0  # Accumulate epoch loss
        total_samples = 0  # Count total samples for normalization

        for batch_data in self.train_loader:
            # Unpack images (labels unused for unsupervised VAE)
            images, _ = batch_data
            images = images.to(self.device)  # Move to compute device

            # Clear accumulated gradients from previous step
            self.optimizer.zero_grad()

            # Forward pass through VAE
            reconstructed, mu, logvar = self.model(images)

            # Compute combined VAE loss
            loss = compute_vae_loss(reconstructed, images, mu, logvar)

            # Backward pass and parameter update
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update parameters

            # Accumulate loss normalized by batch size
            total_loss += loss.item()
            total_samples += images.size(0)

        # Return per-sample average loss
        return total_loss / total_samples


def run_variable_auto_encoder() -> None:
    """
    Entry point function to train the Variational Autoencoder.

    Creates a trainer instance and executes the full training loop.
    """
    try:
        trainer = VAETrainer()  # Create trainer with config
        trainer.train()  # Execute training
    except RuntimeError as runtime_error:
        logger.error("Runtime error during VAE training: %s", runtime_error)
        raise
    except Exception as unexpected_error:
        logger.error("Unexpected error during VAE training: %s", unexpected_error)
        raise
