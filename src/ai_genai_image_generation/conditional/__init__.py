"""
Conditional VAE package for class-conditioned image generation.

This package provides a Conditional Variational Autoencoder that can
generate specific digit classes by conditioning the encoder and decoder
on class label embeddings.
"""

from ai_genai_image_generation.conditional.cvae import ConditionalVAE  # noqa: F401
from ai_genai_image_generation.conditional.cvae import CVAETrainer  # noqa: F401
