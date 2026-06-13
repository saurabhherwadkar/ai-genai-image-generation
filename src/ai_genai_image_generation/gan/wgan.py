"""
Wasserstein GAN (WGAN) model and trainer with weight clipping.

This module implements a WGAN that uses the Wasserstein distance
(Earth Mover's distance) as the training objective instead of the
standard GAN's Jensen-Shannon divergence. Weight clipping enforces
the Lipschitz constraint on the critic.
"""

import logging  # Logging framework for WGAN events

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network building blocks
import torch.optim as optim  # Optimization algorithms

from ai_genai_image_generation.config import get_settings  # Configuration access
from ai_genai_image_generation.data_loader import create_mnist_data_loaders  # Data
from ai_genai_image_generation.device import get_device  # Device selection

# Module-level logger for WGAN events
logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    WGAN Generator network that maps noise vectors to images.

    Transforms a 100-dimensional noise vector into a 28x28 grayscale
    image using fully connected and transposed convolutional layers.
    """

    def __init__(self, noise_dim: int = 100) -> None:
        """
        Initialize the generator network.

        Args:
            noise_dim: Dimensionality of the input noise vector.
        """
        super().__init__()

        # Sequential model from noise to image
        self.model = nn.Sequential(
            # FC layer: noise vector to spatial feature representation
            nn.Linear(noise_dim, 256 * 7 * 7),
            # Reshape flat vector to 3D feature map (256, 7, 7)
            nn.Unflatten(1, (256, 7, 7)),
            # First deconv: 256 -> 128 channels, doubles spatial (7->14)
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),  # Normalize activations for stable training
            nn.LeakyReLU(0.02),  # Leaky ReLU prevents dead neurons
            # Second deconv: 128 -> 64 channels, same spatial (14->14)
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.LeakyReLU(0.02),  # Leaky ReLU activation
            # Final deconv: 64 -> 1 channel, doubles spatial (14->28)
            nn.ConvTranspose2d(
                64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Tanh(),  # Output range [-1, 1] matching normalized images
        )

        logger.debug("Generator initialized with noise_dim=%d", noise_dim)

    def forward(self, noise_vector: torch.Tensor) -> torch.Tensor:
        """
        Generate fake images from random noise.

        Args:
            noise_vector: Random noise input (batch, noise_dim).

        Returns:
            Generated images (batch, 1, 28, 28).
        """
        # Pass noise through generator to produce images
        generated_images = self.model(noise_vector)
        return generated_images


class Critic(nn.Module):
    """
    WGAN Critic (discriminator) that scores image realism.

    Unlike a standard discriminator, the WGAN critic outputs an
    unbounded real-valued score rather than a probability. Higher
    scores indicate more realistic images.
    """

    def __init__(self) -> None:
        """Initialize the critic convolutional network."""
        super().__init__()

        # Sequential model from image to realism score
        self.model = nn.Sequential(
            # First conv: 1 -> 64 channels, halves spatial (28->14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),  # Normalize activations
            nn.LeakyReLU(0.02),  # Leaky activation for gradient flow
            # Second conv: 64 -> 128 channels, halves spatial (14->7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Normalize activations
            nn.LeakyReLU(0.02),  # Leaky activation for gradient flow
            # Final conv: 128 -> 1 channel, reduces to scalar (7->1)
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0),
            nn.Flatten(),  # Flatten to scalar per sample
        )

        logger.debug("Critic initialized")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Score images for realism (higher = more realistic).

        Args:
            images: Input images (batch, 1, 28, 28).

        Returns:
            Realism scores (batch, 1).
        """
        # Pass images through critic to compute scores
        scores = self.model(images)
        return scores


class WGANTrainer:
    """
    Trainer class for the Wasserstein GAN.

    Implements the WGAN training algorithm with alternating critic
    and generator updates, weight clipping for Lipschitz constraint,
    and configurable update ratios.
    """

    def __init__(self) -> None:
        """Initialize WGAN trainer with models, optimizers, and config."""
        # Load hyperparameters from configuration
        settings = get_settings()
        self.batch_size: int = settings.get("wgan", "batch_size", default=64)
        self.learning_rate: float = settings.get("wgan", "learning_rate", default=0.0001)
        self.num_epochs: int = settings.get("wgan", "num_epochs", default=250)
        self.noise_dim: int = settings.get("wgan", "noise_dim", default=100)
        self.clip_value: float = settings.get("wgan", "clip_value", default=1.0)
        self.n_critic: int = settings.get("wgan", "n_critic", default=2)
        self.beta1: float = settings.get("wgan", "beta1", default=0.6)
        self.beta2: float = settings.get("wgan", "beta2", default=0.999)

        # Select compute device
        self.device = get_device()

        # Initialize generator and critic networks on device
        self.generator = Generator(self.noise_dim).to(self.device)
        self.critic = Critic().to(self.device)

        # Adam optimizer for critic with custom betas
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        # Adam optimizer for generator with custom betas
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        # Create data loaders with normalization (images in [-1, 1])
        self.train_loader, _ = create_mnist_data_loaders(
            batch_size=self.batch_size,
            normalize=True,
        )

        logger.info(
            "WGANTrainer initialized: epochs=%d, n_critic=%d, clip=%.2f",
            self.num_epochs,
            self.n_critic,
            self.clip_value,
        )

    def train(self) -> dict[str, list[float]]:
        """
        Execute the full WGAN training loop.

        Returns:
            Dictionary with 'critic_losses' and 'generator_losses' histories.
        """
        # Track loss histories for both networks
        critic_losses: list[float] = []
        generator_losses: list[float] = []

        logger.info("Starting WGAN training for %d epochs", self.num_epochs)

        for epoch in range(self.num_epochs):
            # Train one epoch and collect losses
            epoch_critic_loss, epoch_gen_loss = self._train_one_epoch(epoch)
            critic_losses.append(epoch_critic_loss)
            generator_losses.append(epoch_gen_loss)

            # Log progress periodically (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Critic Loss: %.4f, Generator Loss: %.4f",
                    epoch + 1,
                    self.num_epochs,
                    epoch_critic_loss,
                    epoch_gen_loss,
                )

        logger.info("WGAN training complete")
        return {"critic_losses": critic_losses, "generator_losses": generator_losses}

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Train both critic and generator for one epoch.

        The critic is updated n_critic times per generator update
        to maintain stable training dynamics.

        Args:
            epoch: Current epoch number for logging.

        Returns:
            Tuple of (average_critic_loss, average_generator_loss).
        """
        total_critic_loss = 0.0  # Accumulate critic losses
        total_generator_loss = 0.0  # Accumulate generator losses
        critic_updates = 0  # Count critic update steps
        generator_updates = 0  # Count generator update steps

        for batch_idx, (real_images, _) in enumerate(self.train_loader):
            # Move real images to compute device
            real_images = real_images.to(self.device)

            # Train critic on this batch
            critic_loss = self._train_critic(real_images)
            total_critic_loss += critic_loss
            critic_updates += 1

            # Train generator every n_critic batches
            if batch_idx % self.n_critic == 0:
                gen_loss = self._train_generator(real_images.size(0))
                total_generator_loss += gen_loss
                generator_updates += 1

        # Compute average losses for this epoch
        avg_critic = total_critic_loss / max(critic_updates, 1)
        avg_generator = total_generator_loss / max(generator_updates, 1)
        return avg_critic, avg_generator

    def _train_critic(self, real_images: torch.Tensor) -> float:
        """
        Perform one critic training step.

        Maximizes the Wasserstein distance between real and fake
        score distributions, then clips weights for Lipschitz constraint.

        Args:
            real_images: Batch of real training images.

        Returns:
            Critic loss value for this step.
        """
        # Clear critic gradients
        self.critic_optimizer.zero_grad()

        # Generate fake images from random noise
        noise = torch.randn(real_images.size(0), self.noise_dim).to(self.device)
        fake_images = self.generator(noise).detach()  # Detach to avoid generator gradients

        # Score real images (should be high)
        critic_real_scores = self.critic(real_images).reshape(-1)
        # Score fake images (should be low)
        critic_fake_scores = self.critic(fake_images).reshape(-1)

        # Wasserstein loss: minimize -(E[real] - E[fake])
        critic_loss = -(torch.mean(critic_real_scores) - torch.mean(critic_fake_scores))

        # Backpropagate and update critic
        critic_loss.backward()
        self.critic_optimizer.step()

        # Enforce Lipschitz constraint via weight clipping
        self._clip_critic_weights()

        return critic_loss.item()

    def _train_generator(self, batch_size: int) -> float:
        """
        Perform one generator training step.

        Minimizes the negative critic score on generated images,
        encouraging the generator to produce more realistic samples.

        Args:
            batch_size: Number of fake samples to generate.

        Returns:
            Generator loss value for this step.
        """
        # Clear generator gradients
        self.generator_optimizer.zero_grad()

        # Generate fake images from noise
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_images = self.generator(noise)

        # Get critic scores on fake images (no detach - need gradients)
        critic_fake_scores = self.critic(fake_images).reshape(-1)

        # Generator loss: minimize -E[critic(fake)]
        generator_loss = -torch.mean(critic_fake_scores)

        # Backpropagate and update generator
        generator_loss.backward()
        self.generator_optimizer.step()

        return generator_loss.item()

    def _clip_critic_weights(self) -> None:
        """
        Clip critic weights to enforce Lipschitz constraint.

        Clamps all critic parameters to [-clip_value, clip_value]
        as required by the original WGAN algorithm.
        """
        # Iterate over all critic parameters and clamp values
        for parameter in self.critic.parameters():
            parameter.data.clamp_(-self.clip_value, self.clip_value)


def run_wgan() -> None:
    """
    Entry point function to train the Wasserstein GAN.

    Creates a trainer instance and executes the full training loop.
    """
    try:
        trainer = WGANTrainer()  # Initialize trainer with config
        trainer.train()  # Run adversarial training
    except RuntimeError as runtime_error:
        logger.error("Runtime error during WGAN training: %s", runtime_error)
        raise
    except Exception as unexpected_error:
        logger.error("Unexpected error during WGAN training: %s", unexpected_error)
        raise
