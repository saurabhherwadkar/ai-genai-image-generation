"""
Main entry point for the AI Generative Image Generation project.

This module provides CLI entry points for training each generative model.
It initializes logging, loads configuration, and dispatches to the
appropriate training function.
"""

import logging  # Logging framework for main module events
import sys  # System module for exit codes

from ai_genai_image_generation.logging_config import setup_logging  # Log setup

# Module-level logger (initialized after setup_logging is called)
logger = logging.getLogger(__name__)


def run_auto_encoder() -> None:
    """
    CLI entry point to train the Convolutional Autoencoder.

    Sets up logging, then delegates to the autoencoder trainer.
    """
    # Initialize logging system from configuration
    setup_logging()
    logger.info("Starting Autoencoder training pipeline")

    try:
        # Import trainer here to ensure logging is configured first
        from ai_genai_image_generation.autoencoder.trainer import run_auto_encoder as train

        # Execute the training function
        train()
        logger.info("Autoencoder training completed successfully")
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as error:
        # Log fatal errors and exit with error code
        logger.critical("Fatal error in autoencoder training: %s", error)
        sys.exit(1)


def run_vae() -> None:
    """
    CLI entry point to train the Variational Autoencoder.

    Sets up logging, then delegates to the VAE trainer.
    """
    # Initialize logging system from configuration
    setup_logging()
    logger.info("Starting VAE training pipeline")

    try:
        from ai_genai_image_generation.autoencoder.variable_auto_encoder import (
            run_variable_auto_encoder as train,
        )

        train()
        logger.info("VAE training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as error:
        logger.critical("Fatal error in VAE training: %s", error)
        sys.exit(1)


def run_cvae() -> None:
    """
    CLI entry point to train the Conditional VAE.

    Sets up logging, then delegates to the cVAE trainer.
    """
    # Initialize logging system from configuration
    setup_logging()
    logger.info("Starting Conditional VAE training pipeline")

    try:
        from ai_genai_image_generation.conditional.cvae import run_cvae as train

        train()
        logger.info("Conditional VAE training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as error:
        logger.critical("Fatal error in cVAE training: %s", error)
        sys.exit(1)


def run_wgan() -> None:
    """
    CLI entry point to train the Wasserstein GAN.

    Sets up logging, then delegates to the WGAN trainer.
    """
    # Initialize logging system from configuration
    setup_logging()
    logger.info("Starting WGAN training pipeline")

    try:
        from ai_genai_image_generation.gan.wgan import run_wgan as train

        train()
        logger.info("WGAN training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as error:
        logger.critical("Fatal error in WGAN training: %s", error)
        sys.exit(1)


def run_diffusion() -> None:
    """
    CLI entry point to train the Diffusion Model (DDPM).

    Sets up logging, then delegates to the diffusion trainer.
    """
    # Initialize logging system from configuration
    setup_logging()
    logger.info("Starting Diffusion Model training pipeline")

    try:
        from ai_genai_image_generation.diffusion.forward_diffusion import (
            run_diffusion as train,
        )

        train()
        logger.info("Diffusion model training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as error:
        logger.critical("Fatal error in diffusion training: %s", error)
        sys.exit(1)
