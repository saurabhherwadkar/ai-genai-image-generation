"""
Configuration module for the AI Image Generation project.

This module handles loading settings from YAML configuration files,
supporting environment-specific overrides and providing a single
source of truth for all configurable parameters.
"""

import os  # Operating system interface for environment variables
import logging  # Standard logging module for log output
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Any  # Type hint for generic values

import yaml  # YAML parser for configuration files


# Module-level logger instance for this configuration module
logger = logging.getLogger(__name__)


class Settings:
    """
    Singleton settings class that loads and provides access to configuration.

    This class reads the YAML configuration file and provides dictionary-style
    access to nested configuration values. It supports environment-specific
    overrides through the APP_ENV environment variable.
    """

    _instance: "Settings | None" = None  # Singleton instance reference
    _config: dict[str, Any] = {}  # Loaded configuration dictionary

    def __new__(cls) -> "Settings":
        """Ensure only one Settings instance exists (Singleton pattern)."""
        if cls._instance is None:  # Check if instance already created
            cls._instance = super().__new__(cls)  # Create new instance
            cls._instance._load_config()  # Load configuration on first creation
        return cls._instance  # Return the single instance

    def _load_config(self) -> None:
        """
        Load configuration from YAML files.

        Loads the base settings.yaml first, then overlays any
        environment-specific settings file if it exists.
        """
        # Determine the config directory path relative to project root
        config_dir = self._find_config_dir()

        # Load base configuration file
        base_config_path = config_dir / "settings.yaml"
        self._config = self._read_yaml(base_config_path)

        # Determine current environment from environment variable
        environment = os.environ.get("APP_ENV", "development")

        # Load environment-specific override file if it exists
        env_config_path = config_dir / f"settings.{environment}.yaml"
        if env_config_path.exists():  # Only load if override file exists
            env_config = self._read_yaml(env_config_path)
            self._merge_config(self._config, env_config)  # Deep merge override values

        logger.debug("Configuration loaded for environment: %s", environment)

    def _find_config_dir(self) -> Path:
        """
        Locate the configuration directory by searching upward from this file.

        Returns:
            Path to the config directory.

        Raises:
            FileNotFoundError: If no config directory is found.
        """
        # Start from the current file's parent directory
        current = Path(__file__).resolve().parent

        # Traverse upward until we find the config directory
        for _ in range(10):  # Limit search depth to prevent infinite loop
            config_path = current / "config"
            if config_path.exists():  # Found the config directory
                return config_path
            current = current.parent  # Move up one directory level

        # Raise error if config directory not found after traversal
        raise FileNotFoundError(
            "Configuration directory 'config/' not found in project hierarchy."
        )

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """
        Read and parse a YAML configuration file.

        Args:
            path: Path to the YAML file to read.

        Returns:
            Dictionary containing parsed YAML content.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not path.exists():  # Verify file exists before reading
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Open and parse the YAML file with safe loader to prevent code execution
        with open(path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}

    def _merge_config(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> None:
        """
        Deep merge override configuration into base configuration.

        Recursively merges nested dictionaries. Non-dict values in
        override will replace values in base.

        Args:
            base: Base configuration dictionary to merge into.
            override: Override configuration dictionary with new values.
        """
        for key, value in override.items():  # Iterate override key-value pairs
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                self._merge_config(base[key], value)
            else:
                # Replace value directly for non-dict types
                base[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested configuration value using dot-notation-style keys.

        Args:
            *keys: Sequence of keys to traverse the config hierarchy.
            default: Value to return if the key path does not exist.

        Returns:
            The configuration value, or default if not found.

        Example:
            settings.get("autoencoder", "batch_size")  # Returns 32
        """
        current = self._config  # Start at the root of configuration

        for key in keys:  # Traverse each level of the key path
            if isinstance(current, dict) and key in current:
                current = current[key]  # Descend into nested dictionary
            else:
                return default  # Key not found, return default value

        return current  # Return the found value

    def reload(self) -> None:
        """Force reload configuration from disk (useful for testing)."""
        self._load_config()  # Re-read and parse all config files


def get_settings() -> Settings:
    """
    Factory function to get the Settings singleton instance.

    Returns:
        The global Settings instance.
    """
    return Settings()  # Returns existing instance due to Singleton pattern
