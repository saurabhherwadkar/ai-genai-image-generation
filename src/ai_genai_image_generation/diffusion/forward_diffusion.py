"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.

This module implements the forward diffusion process and the UNet-based
noise prediction network for DDPM. The forward process gradually adds
Gaussian noise to images, and the UNet learns to predict and remove
that noise for image generation.
"""

import logging  # Logging framework for diffusion events

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network building blocks
import torch.nn.functional as F  # Functional API for activations
import torch.optim as optim  # Optimization algorithms

from ai_genai_image_generation.config import get_settings  # Configuration access
from ai_genai_image_generation.data_loader import create_mnist_data_loaders  # Data
from ai_genai_image_generation.device import get_device  # Device selection

# Module-level logger for diffusion model events
logger = logging.getLogger(__name__)


def create_sinusoidal_embedding(num_timesteps: int, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for timestep encoding.

    Generates fixed sinusoidal embeddings that encode the diffusion
    timestep as a continuous vector, similar to positional encoding
    in Transformers.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        embedding_dim: Dimension of each embedding vector.

    Returns:
        Embedding tensor of shape (num_timesteps, embedding_dim).
    """
    # Initialize empty embedding matrix
    embedding = torch.zeros(num_timesteps, embedding_dim)

    # Compute frequency scaling factors for each dimension
    frequencies = torch.tensor(
        [1.0 / 10_000 ** (2 * j / embedding_dim) for j in range(embedding_dim)]
    )
    # Reshape frequencies for broadcasting with timestep indices
    frequencies = frequencies.reshape((1, embedding_dim))

    # Create timestep index column vector
    timesteps = torch.arange(num_timesteps).reshape((num_timesteps, 1))

    # Fill even indices with sine values
    embedding[:, ::2] = torch.sin(timesteps * frequencies[:, ::2])
    # Fill odd indices with cosine values
    embedding[:, 1::2] = torch.cos(timesteps * frequencies[:, ::2])

    return embedding


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with optional skip connection.

    Applies two sequential convolutions with batch normalization and
    GELU activation. When is_res=True, adds the input (or projected
    input) to the output for residual learning.
    """

    def __init__(
        self, in_channels: int, out_channels: int, is_residual: bool = False
    ) -> None:
        """
        Initialize the residual convolutional block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            is_residual: Whether to use residual (skip) connection.
        """
        super().__init__()

        # Track whether input/output channels match for skip connection
        self.same_channels = in_channels == out_channels
        # Flag for residual connection usage
        self.is_residual = is_residual

        # First convolution with batch norm and GELU activation
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Normalize feature maps
            nn.GELU(),  # Gaussian Error Linear Unit activation
        )

        # Second convolution with batch norm and GELU activation
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Normalize feature maps
            nn.GELU(),  # Gaussian Error Linear Unit activation
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            input_tensor: Input feature map (batch, in_channels, H, W).

        Returns:
            Output feature map (batch, out_channels, H, W).
        """
        if self.is_residual:
            # Apply first convolution block
            intermediate = self.conv_block_1(input_tensor)
            # Apply second convolution block
            output = self.conv_block_2(intermediate)

            if self.same_channels:
                # Direct residual addition when channels match
                output = input_tensor + output
            else:
                # Use intermediate as skip when channels differ
                output = intermediate + output

            # Scale by 1/sqrt(2) to maintain variance stability
            return output / 1.414
        else:
            # Non-residual path: simple sequential convolutions
            intermediate = self.conv_block_1(input_tensor)
            output = self.conv_block_2(intermediate)
            return output


class UNetBlock(nn.Module):
    """
    UNet building block with time embedding injection.

    Combines convolution, time embedding addition, residual blocks,
    and spatial up/downsampling for the UNet architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        upsample: bool = False,
    ) -> None:
        """
        Initialize a UNet block for either downsampling or upsampling.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embedding_dim: Dimension of time embedding vectors.
            upsample: If True, performs upsampling; otherwise downsampling.
        """
        super().__init__()

        # Linear projection for time embedding to channel space
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)

        if upsample:
            # Upsampling path: input has skip connection (2x channels)
            self.initial_conv = nn.Conv2d(
                2 * in_channels, out_channels, kernel_size=3, padding=1
            )
            # Transposed conv for spatial upsampling
            self.spatial_transform = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            # Downsampling path: standard input channels
            self.initial_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
            # Strided conv for spatial downsampling
            self.spatial_transform = nn.Conv2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )

        # Residual blocks for feature refinement
        self.residual_block_1 = ResidualConvBlock(out_channels, out_channels, is_residual=True)
        self.residual_block_2 = ResidualConvBlock(out_channels, out_channels, is_residual=True)

        # ReLU for time embedding activation
        self.activation = nn.ReLU()

    def forward(
        self, features: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Process features with time conditioning and spatial transform.

        Args:
            features: Input feature map (batch, channels, H, W).
            time_embedding: Time step embedding vector (batch, time_dim).

        Returns:
            Transformed feature map with changed spatial resolution.
        """
        # Apply initial convolution to project channels
        hidden = self.initial_conv(features)

        # Project time embedding to channel dimension and add
        time_projected = self.activation(self.time_mlp(time_embedding))
        # Expand time embedding to spatial dimensions for broadcasting
        time_projected = time_projected[(...,) + (None,) * 2]
        # Add time information to spatial features
        hidden = hidden + time_projected

        # Refine with residual blocks
        hidden = self.residual_block_1(hidden)
        hidden = self.residual_block_2(hidden)

        # Apply spatial up/downsampling
        output = self.spatial_transform(hidden)
        return output


class UNet(nn.Module):
    """
    UNet architecture for noise prediction in diffusion models.

    Implements a symmetric encoder-decoder with skip connections
    and time embedding conditioning at each level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_embedding_dim: int = 256,
    ) -> None:
        """
        Initialize the UNet noise prediction network.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output channels (predicted noise).
            time_embedding_dim: Dimension of time step embeddings.
        """
        super().__init__()

        # Store time dimension for embedding creation
        self.time_embedding_dim = time_embedding_dim

        # Time embedding MLP for processing timestep encodings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),  # Non-linear transformation
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # Initial projection from image channels to feature space
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Encoder (downsampling) path with increasing channel depth
        self.down_blocks = nn.ModuleList([
            UNetBlock(64, 128, time_embedding_dim),    # 32->16
            UNetBlock(128, 256, time_embedding_dim),   # 16->8
            UNetBlock(256, 512, time_embedding_dim),   # 8->4
            UNetBlock(512, 1024, time_embedding_dim),  # 4->2
        ])

        # Decoder (upsampling) path with skip connections
        self.up_blocks = nn.ModuleList([
            UNetBlock(1024, 512, time_embedding_dim, upsample=True),   # 2->4
            UNetBlock(512, 256, time_embedding_dim, upsample=True),    # 4->8
            UNetBlock(256, 128, time_embedding_dim, upsample=True),    # 8->16
            UNetBlock(128, 64, time_embedding_dim, upsample=True),     # 16->32
        ])

        # Final 1x1 convolution to map features to output channels
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        logger.info("UNet initialized with time_dim=%d", time_embedding_dim)

    def forward(
        self, noisy_images: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in images conditioned on timestep.

        Args:
            noisy_images: Noisy input images (batch, channels, H, W).
            timesteps: Diffusion timestep indices (batch,).

        Returns:
            Predicted noise tensor with same shape as input.
        """
        # Create sinusoidal timestep embeddings and move to device
        time_emb = create_sinusoidal_embedding(
            timesteps.shape[0], self.time_embedding_dim
        ).to(noisy_images.device)
        # Process through time MLP
        time_emb = self.time_mlp(time_emb)

        # Initial feature projection
        features = self.initial_conv(noisy_images)

        # Encoder path: collect skip connections at each level
        skip_connections = []
        for down_block in self.down_blocks:
            features = down_block(features, time_emb)
            skip_connections.append(features)  # Store for decoder

        # Decoder path: concatenate skip connections from encoder
        for up_block in self.up_blocks:
            skip_features = skip_connections.pop()  # Get matching skip
            # Concatenate skip connection along channel dimension
            features = torch.cat((features, skip_features), dim=1)
            features = up_block(features, time_emb)

        # Final projection to noise prediction
        predicted_noise = self.output_conv(features)
        return predicted_noise


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Implements the forward (noise addition) and reverse (denoising)
    diffusion processes with a linear beta schedule.
    """

    def __init__(
        self,
        network: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the DDPM with noise schedule and denoising network.

        Args:
            network: Neural network for noise prediction (UNet).
            num_timesteps: Total number of diffusion timesteps.
            beta_start: Starting value of the linear beta schedule.
            beta_end: Ending value of the linear beta schedule.
            device: Compute device string ("cuda" or "cpu").
        """
        super().__init__()

        # Move noise prediction network to device
        self.network = network.to(device)
        # Store number of diffusion steps
        self.num_timesteps = num_timesteps
        # Store device reference
        self.device = device

        # Linear beta schedule from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

        # Pre-compute alpha values: alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # Cumulative product of alphas for closed-form forward process
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Shifted cumulative product for posterior variance
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # Square root of reciprocal alphas for reverse process mean
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Square root of cumulative alpha products for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # Square root of (1 - cumulative alphas) for noise scaling
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

        # Posterior variance for reverse process sampling
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        logger.info(
            "DDPM initialized: timesteps=%d, beta=[%.5f, %.4f]",
            num_timesteps,
            beta_start,
            beta_end,
        )

    def forward_diffusion(
        self, clean_images: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion (add noise) at specified timesteps.

        Uses the closed-form solution: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise

        Args:
            clean_images: Original clean images (batch, C, H, W).
            timesteps: Timestep indices for each image (batch,).

        Returns:
            Tuple of (noisy_images, noise) for loss computation.
        """
        # Sample Gaussian noise with same shape as images
        noise = torch.randn_like(clean_images)

        # Get cumulative alpha values for the given timesteps
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting with image dimensions
        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None, None]

        # Apply closed-form forward diffusion equation
        noisy_images = (
            sqrt_alpha_cumprod * clean_images + sqrt_one_minus_alpha * noise
        )

        return noisy_images, noise


class DiffusionTrainer:
    """
    Trainer class for the Denoising Diffusion Probabilistic Model.

    Manages the training loop that teaches the UNet to predict noise
    at various diffusion timesteps.
    """

    def __init__(self) -> None:
        """Initialize diffusion trainer with model, optimizer, and data."""
        # Load hyperparameters from configuration
        settings = get_settings()
        self.batch_size: int = settings.get("diffusion", "batch_size", default=32)
        self.learning_rate: float = settings.get("diffusion", "learning_rate", default=0.001)
        self.num_epochs: int = settings.get("diffusion", "num_epochs", default=100)
        self.num_timesteps: int = settings.get("diffusion", "num_timesteps", default=1000)
        self.beta_start: float = settings.get("diffusion", "beta_start", default=0.0001)
        self.beta_end: float = settings.get("diffusion", "beta_end", default=0.02)
        self.time_dim: int = settings.get("diffusion", "time_embedding_dim", default=256)
        self.image_size: int = settings.get("diffusion", "image_size", default=32)

        # Select compute device
        self.device = get_device()

        # Initialize UNet noise prediction network
        unet = UNet(
            in_channels=1,
            out_channels=1,
            time_embedding_dim=self.time_dim,
        )

        # Initialize DDPM with UNet and noise schedule
        self.ddpm = DDPM(
            network=unet,
            num_timesteps=self.num_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=str(self.device),
        )

        # Adam optimizer for UNet parameters
        self.optimizer = optim.Adam(
            self.ddpm.network.parameters(), lr=self.learning_rate
        )

        # Create data loaders with resizing and normalization
        self.train_loader, _ = create_mnist_data_loaders(
            batch_size=self.batch_size,
            normalize=True,
            resize=self.image_size,
        )

        logger.info(
            "DiffusionTrainer initialized: timesteps=%d, epochs=%d",
            self.num_timesteps,
            self.num_epochs,
        )

    def train(self) -> list[float]:
        """
        Execute the full diffusion model training loop.

        Returns:
            List of average loss values per epoch.
        """
        epoch_losses: list[float] = []  # Track training progress

        logger.info("Starting diffusion training for %d epochs", self.num_epochs)

        for epoch in range(self.num_epochs):
            # Train one epoch of noise prediction
            average_loss = self._train_one_epoch(epoch)
            epoch_losses.append(average_loss)

            logger.info(
                "Epoch [%d/%d], MSE Loss: %.6f",
                epoch + 1,
                self.num_epochs,
                average_loss,
            )

        logger.info("Diffusion training complete. Final loss: %.6f", epoch_losses[-1])
        return epoch_losses

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train the noise prediction network for one epoch.

        For each batch, samples random timesteps, applies forward
        diffusion, and trains the UNet to predict the added noise.

        Args:
            epoch: Current epoch index for logging.

        Returns:
            Average MSE loss for this epoch.
        """
        # Set network to training mode
        self.ddpm.network.train()

        total_loss = 0.0  # Accumulate epoch loss
        batch_count = 0  # Count batches for averaging

        for batch_data in self.train_loader:
            # Unpack images and move to device
            images, _ = batch_data
            images = images.to(self.device)

            # Sample random timesteps uniformly for each image
            timesteps = torch.randint(
                0, self.num_timesteps, (images.size(0),)
            ).to(self.device)

            # Apply forward diffusion to get noisy images and target noise
            noisy_images, target_noise = self.ddpm.forward_diffusion(
                images, timesteps
            )

            # Predict noise using UNet conditioned on timestep
            predicted_noise = self.ddpm.network(noisy_images, timesteps)

            # Compute MSE loss between predicted and actual noise
            loss = F.mse_loss(predicted_noise, target_noise)

            # Backward pass and parameter update
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update parameters

            total_loss += loss.item()  # Accumulate loss
            batch_count += 1

        # Return average loss for the epoch
        return total_loss / max(batch_count, 1)


def run_diffusion() -> None:
    """
    Entry point function to train the diffusion model.

    Creates a trainer instance and executes the full training loop.
    """
    try:
        trainer = DiffusionTrainer()  # Initialize with config
        trainer.train()  # Run training
    except RuntimeError as runtime_error:
        logger.error("Runtime error during diffusion training: %s", runtime_error)
        raise
    except Exception as unexpected_error:
        logger.error("Unexpected error during diffusion training: %s", unexpected_error)
        raise
