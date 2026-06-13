"""
GAN package for adversarial image generation.

This package implements a Wasserstein GAN (WGAN) with weight clipping
for stable adversarial training of image generators on MNIST.
"""

from ai_genai_image_generation.gan.wgan import Generator  # noqa: F401
from ai_genai_image_generation.gan.wgan import Critic  # noqa: F401
from ai_genai_image_generation.gan.wgan import WGANTrainer  # noqa: F401
