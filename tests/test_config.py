"""
Unit tests for the configuration module.

Tests configuration loading, YAML parsing, environment-specific
overrides, and the Settings singleton behavior.
"""

import os  # Environment variable manipulation
from pathlib import Path  # Path handling
from unittest.mock import patch  # Mocking for isolation

import pytest  # Test framework


class TestSettings:
    """Test suite for the Settings configuration class."""

    def test_settings_loads_from_yaml(self, sample_config_dir):
        """Verify that Settings correctly loads values from YAML file."""
        # Patch the config directory finder to use our temp directory
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.config import Settings

            # Create a fresh settings instance
            settings = Settings()

            # Verify nested config values are accessible
            assert settings.get("autoencoder", "batch_size") == 4
            assert settings.get("logging", "level") == "DEBUG"
            assert settings.get("device", "preferred") == "cpu"

    def test_settings_returns_default_for_missing_keys(self, sample_config_dir):
        """Verify default values are returned for missing config keys."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.config import Settings

            settings = Settings()

            # Missing key should return provided default
            result = settings.get("nonexistent", "key", default=42)
            assert result == 42

    def test_settings_returns_none_for_missing_without_default(self, sample_config_dir):
        """Verify None is returned when no default is specified."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.config import Settings

            settings = Settings()

            # Missing key without default should return None
            result = settings.get("nonexistent", "key")
            assert result is None

    def test_settings_singleton_pattern(self, sample_config_dir):
        """Verify Settings implements singleton correctly."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.config import Settings

            # Two instantiations should return the same object
            settings_a = Settings()
            settings_b = Settings()
            assert settings_a is settings_b

    def test_settings_environment_override(self, sample_config_dir):
        """Verify environment-specific config overrides base values."""
        # Create an environment-specific override file
        env_config = sample_config_dir / "config" / "settings.production.yaml"
        env_config.write_text(
            "autoencoder:\n  batch_size: 128\n",
            encoding="utf-8",
        )

        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            # Set environment variable to trigger override
            with patch.dict(os.environ, {"APP_ENV": "production"}):
                from ai_genai_image_generation.config import Settings

                settings = Settings()

                # Overridden value should take effect
                assert settings.get("autoencoder", "batch_size") == 128
                # Non-overridden values should remain from base
                assert settings.get("logging", "level") == "DEBUG"

    def test_settings_deep_merge(self, sample_config_dir):
        """Verify deep merge correctly overlays nested dictionaries."""
        from ai_genai_image_generation.config import Settings

        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            settings = Settings()

            # Test the merge function directly
            base = {"a": {"b": 1, "c": 2}, "d": 3}
            override = {"a": {"b": 99}, "e": 5}
            settings._merge_config(base, override)

            # Overridden value should update
            assert base["a"]["b"] == 99
            # Non-overridden nested value should remain
            assert base["a"]["c"] == 2
            # Top-level non-overridden should remain
            assert base["d"] == 3
            # New key should be added
            assert base["e"] == 5

    def test_get_settings_factory(self, sample_config_dir):
        """Verify get_settings factory returns Settings singleton."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.config import get_settings

            settings = get_settings()
            assert settings is not None
            assert settings.get("autoencoder", "batch_size") == 4
