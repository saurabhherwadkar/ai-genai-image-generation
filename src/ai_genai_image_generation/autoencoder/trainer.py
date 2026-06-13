"""
Training logic for the Convolutional Autoencoder.

This module encapsulates the training loop, loss computation, and
optimizer management for the Autoencoder model. It follows the
single-responsibility principle by separating training concerns
from model architecture.
"""

import logging  # Logging framework for training progress

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network loss functions
import torch.optim as optim  # Optimization algorithms

from ai_genai_image_generation.autoencoder.model import Autoencoder  # Model class
from ai_genai_image_generation.config import get_settings  # Configuration access
from ai_genai_image_generation.data_loader import create_mnist_data_loaders  # Data
from ai_genai_image_generation.device import get_device  # Device selection

# Module-level logger for training events
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Trainer class for the Convolutional Autoencoder.

    Manages the training loop, loss function, optimizer, and device
    placement. All hyperparameters are loaded from configuration.
    """

    def __init__(self) -> None:
        """Initialize trainer with model, optimizer, and data loaders."""
        # Load hyperparameters from configuration file
        settings = get_settings()
        self.batch_size: int = settings.get("autoencoder", "batch_size", default=32)
        self.learning_rate: float = settings.get(
            "autoencoder", "learning_rate", default=0.0001
        )
        self.num_epochs: int = settings.get("autoencoder", "num_epochs", default=30)

        # Select compute device (GPU or CPU)
        self.device = get_device()

        # Initialize the autoencoder model and move to device
        self.model = Autoencoder().to(self.device)

        # Mean Squared Error loss for pixel-wise reconstruction
        self.criterion = nn.MSELoss()

        # Adam optimizer for adaptive learning rate per parameter
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Create data loaders for MNIST dataset
        self.train_loader, self.test_loader = create_mnist_data_loaders(
            batch_size=self.batch_size
        )

        logger.info(
            "AutoencoderTrainer initialized: epochs=%d, lr=%f, batch_size=%d",
            self.num_epochs,
            self.learning_rate,
            self.batch_size,
        )

    def train(self) -> list[float]:
        """
        Execute the full training loop over all epochs.

        Returns:
            List of average loss values per epoch for monitoring.
        """
        # Store loss history for tracking training progress
        epoch_losses: list[float] = []

        logger.info("Starting autoencoder training for %d epochs", self.num_epochs)

        # Iterate over each training epoch
        for epoch in range(self.num_epochs):
            # Train one epoch and get the average loss
            average_loss = self._train_one_epoch(epoch)
            epoch_losses.append(average_loss)  # Record epoch loss

            # Log progress at each epoch completion
            logger.info(
                "Epoch [%d/%d], Average Loss: %.4f",
                epoch + 1,
                self.num_epochs,
                average_loss,
            )

        logger.info("Training complete. Final loss: %.4f", epoch_losses[-1])
        return epoch_losses  # Return full loss history

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train the model for a single epoch.

        Args:
            epoch: Current epoch number (zero-indexed) for logging.

        Returns:
            Average loss value across all batches in this epoch.
        """
        # Set model to training mode (enables dropout, batchnorm updates)
        self.model.train()

        # Accumulate total loss across all batches
        total_loss = 0.0
        batch_count = 0  # Track number of batches processed

        # Iterate over all mini-batches in the training data
        for batch_data in self.train_loader:
            # Unpack images and labels (labels unused for autoencoder)
            images, _ = batch_data
            # Move images to the compute device
            images = images.to(self.device)

            # Forward pass: reconstruct images through the autoencoder
            reconstructed = self.model(images)

            # Compute pixel-wise reconstruction loss
            loss = self.criterion(reconstructed, images)

            # Backward pass: compute gradients
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients via backpropagation
            self.optimizer.step()  # Update model parameters

            # Accumulate loss for epoch average calculation
            total_loss += loss.item()
            batch_count += 1

        # Calculate and return average loss for this epoch
        average_loss = total_loss / batch_count
        return average_loss

    def evaluate(self) -> float:
        """
        Evaluate model reconstruction quality on the test dataset.

        Returns:
            Average reconstruction loss on the test set.
        """
        # Set model to evaluation mode (disables dropout, freezes batchnorm)
        self.model.eval()

        total_loss = 0.0  # Accumulate test loss
        batch_count = 0  # Count test batches

        # Disable gradient computation for efficiency during evaluation
        with torch.no_grad():
            for batch_data in self.test_loader:
                # Unpack and move test images to device
                images, _ = batch_data
                images = images.to(self.device)

                # Forward pass only (no gradient tracking)
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)

                total_loss += loss.item()  # Accumulate batch loss
                batch_count += 1

        # Compute average test loss
        average_test_loss = total_loss / batch_count
        logger.info("Evaluation complete. Test Loss: %.4f", average_test_loss)
        return average_test_loss


def run_auto_encoder() -> None:
    """
    Entry point function to train the autoencoder.

    Creates a trainer instance and executes the full training loop.
    This function serves as the public API for running autoencoder training.
    """
    try:
        # Create trainer with configuration-driven parameters
        trainer = AutoencoderTrainer()
        # Execute training and capture loss history
        trainer.train()
    except RuntimeError as runtime_error:
        # Handle CUDA out-of-memory and other runtime errors
        logger.error("Runtime error during training: %s", runtime_error)
        raise
    except Exception as unexpected_error:
        # Catch and log any unforeseen errors
        logger.error("Unexpected error during training: %s", unexpected_error)
        raise
