"""
Device selection module for PyTorch compute device management.

This module provides a centralized way to determine and access the
appropriate compute device (CPU or CUDA GPU) based on configuration
and hardware availability.
"""

import logging  # Logging framework for device selection messages

import torch  # PyTorch framework for device management

from ai_genai_image_generation.config import get_settings  # Configuration access

# Module-level logger for device selection events
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Determine and return the appropriate PyTorch compute device.

    Checks the configuration for a preferred device setting. If set to
    "auto", it will select CUDA if a GPU is available, otherwise falls
    back to CPU.

    Returns:
        torch.device: The selected compute device for tensor operations.
    """
    # Retrieve preferred device from configuration settings
    settings = get_settings()
    preferred = settings.get("device", "preferred", default="auto")

    if preferred == "auto":  # Auto-detect best available device
        if torch.cuda.is_available():  # Check for CUDA GPU availability
            device = torch.device("cuda")  # Use GPU acceleration
            logger.info("Auto-selected CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")  # Fall back to CPU computation
            logger.info("CUDA not available, using CPU device")
    else:
        # Use the explicitly configured device preference
        device = torch.device(preferred)
        logger.info("Using configured device: %s", preferred)

    return device  # Return the determined compute device
