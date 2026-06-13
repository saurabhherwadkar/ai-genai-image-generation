"""
Logging configuration module for the AI Image Generation project.

This module sets up the application-wide logging system based on
settings from the YAML configuration file. It supports configurable
log levels, formats, and optional file output.
"""

import logging  # Standard Python logging framework
import sys  # System-specific parameters for stdout access
from pathlib import Path  # Object-oriented path handling

from ai_genai_image_generation.config import get_settings  # Configuration accessor


def setup_logging() -> None:
    """
    Configure the application-wide logging system.

    Reads logging settings from the configuration file and sets up
    handlers for console output and optional file output. This should
    be called once at application startup before any logging occurs.
    """
    # Retrieve logging configuration from settings
    settings = get_settings()

    # Get log level string and convert to logging constant
    log_level_str = settings.get("logging", "level", default="INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get log format pattern from settings
    log_format = settings.get(
        "logging",
        "format",
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get optional log file path from settings
    log_file = settings.get("logging", "file", default=None)

    # Get the root logger to configure application-wide logging
    root_logger = logging.getLogger()

    # Set the minimum logging level on the root logger
    root_logger.setLevel(log_level)

    # Remove any existing handlers to prevent duplicate log output
    root_logger.handlers.clear()

    # Create a formatter with the configured pattern
    formatter = logging.Formatter(log_format)

    # Create and configure console handler for stdout output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)  # Set level on handler
    console_handler.setFormatter(formatter)  # Apply format pattern
    root_logger.addHandler(console_handler)  # Attach handler to root logger

    # Create and configure file handler if a log file path is specified
    if log_file is not None:
        # Ensure the log file's parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)  # Set level on file handler
        file_handler.setFormatter(formatter)  # Apply format pattern
        root_logger.addHandler(file_handler)  # Attach to root logger

    # Log confirmation that logging system is initialized
    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, file=%s", log_level_str, log_file
    )
