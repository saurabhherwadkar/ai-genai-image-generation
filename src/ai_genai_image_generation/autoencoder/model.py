"""
Convolutional Autoencoder model architecture.

This module defines the Autoencoder neural network that compresses
28x28 grayscale images into a compact latent representation using
convolutional layers, then reconstructs them using transposed
convolutional layers.
"""

import logging  # Logging framework for model events

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network building blocks

# Module-level logger for autoencoder model events
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Convolutional encoder that compresses images to a latent representation.

    Architecture: Three convolutional layers with stride-based downsampling
    that reduce the 28x28 input to a 64-channel 1x1 feature map.
    """

    def __init__(self) -> None:
        """Initialize encoder convolutional layers."""
        super().__init__()

        # Sequential stack of convolutional layers for feature extraction
        self.layers = nn.Sequential(
            # First conv: 1 channel -> 16 channels, halves spatial dims (28->14)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),  # Non-linear activation for feature learning
            # Second conv: 16 -> 32 channels, halves spatial dims (14->7)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),  # Non-linear activation for feature learning
            # Third conv: 32 -> 64 channels, reduces to 1x1 spatial (7->1)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
            ),
        )

        logger.debug("Encoder initialized with 3 convolutional layers")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to latent representation.

        Args:
            input_tensor: Batch of images with shape (batch, 1, 28, 28).

        Returns:
            Encoded tensor with shape (batch, 64, 1, 1).
        """
        # Pass input through all encoder layers sequentially
        encoded = self.layers(input_tensor)
        return encoded


class Decoder(nn.Module):
    """
    Transposed convolutional decoder that reconstructs images from latent space.

    Architecture: Three transposed convolutional layers that upsample
    the 64-channel 1x1 feature map back to a 28x28 grayscale image.
    """

    def __init__(self) -> None:
        """Initialize decoder transposed convolutional layers."""
        super().__init__()

        # Sequential stack of transposed conv layers for upsampling
        self.layers = nn.Sequential(
            # First deconv: 64 -> 32 channels, expands spatial (1->7)
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
            ),
            nn.ReLU(),  # Non-linear activation for reconstruction
            # Second deconv: 32 -> 16 channels, doubles spatial (7->14)
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),  # Non-linear activation for reconstruction
            # Third deconv: 16 -> 1 channel, doubles spatial (14->28)
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),  # Constrain output pixels to [0, 1] range
        )

        logger.debug("Decoder initialized with 3 transposed convolutional layers")

    def forward(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to image space.

        Args:
            latent_tensor: Encoded tensor with shape (batch, 64, 1, 1).

        Returns:
            Reconstructed images with shape (batch, 1, 28, 28).
        """
        # Pass latent representation through all decoder layers
        reconstructed = self.layers(latent_tensor)
        return reconstructed


class Autoencoder(nn.Module):
    """
    Complete convolutional autoencoder combining encoder and decoder.

    This model learns to compress 28x28 grayscale images into a compact
    64-dimensional latent space and reconstruct them with minimal loss.
    The encoder-decoder architecture is symmetric.
    """

    def __init__(self) -> None:
        """Initialize the autoencoder with encoder and decoder components."""
        super().__init__()

        # Encoder sub-network for image compression
        self.encoder = Encoder()
        # Decoder sub-network for image reconstruction
        self.decoder = Decoder()

        logger.info("Autoencoder model initialized")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform full encode-decode forward pass.

        Args:
            input_tensor: Batch of input images (batch, 1, 28, 28).

        Returns:
            Reconstructed images with same shape as input.
        """
        # Compress input to latent representation
        latent = self.encoder(input_tensor)
        # Reconstruct image from latent representation
        reconstructed = self.decoder(latent)
        return reconstructed
