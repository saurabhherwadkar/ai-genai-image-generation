"""
Autoencoder package for convolutional image compression and reconstruction.

This package provides a convolutional autoencoder architecture that learns
to encode MNIST images into a compact latent representation and decode
them back to the original image space.
"""

from ai_genai_image_generation.autoencoder.model import Autoencoder  # noqa: F401
from ai_genai_image_generation.autoencoder.trainer import AutoencoderTrainer  # noqa: F401
