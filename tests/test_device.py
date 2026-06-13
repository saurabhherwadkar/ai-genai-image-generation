"""
Unit tests for the device selection module.

Tests automatic device detection, explicit device configuration,
and correct device object creation.
"""

from unittest.mock import patch, MagicMock  # Mocking for GPU simulation

import pytest  # Test framework
import torch  # PyTorch for device verification


class TestGetDevice:
    """Test suite for device selection logic."""

    def test_returns_cpu_when_configured(self, sample_config_dir):
        """Verify CPU device is returned when explicitly configured."""
        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            from ai_genai_image_generation.device import get_device

            # Test config specifies "cpu" as preferred device
            device = get_device()
            assert device == torch.device("cpu")

    def test_auto_selects_cpu_when_no_cuda(self, sample_config_dir):
        """Verify auto mode falls back to CPU when CUDA unavailable."""
        # Override config to use "auto" mode
        config_file = sample_config_dir / "config" / "settings.yaml"
        content = config_file.read_text(encoding="utf-8")
        content = content.replace('preferred: "cpu"', 'preferred: "auto"')
        config_file.write_text(content, encoding="utf-8")

        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            # Mock CUDA as unavailable
            with patch("torch.cuda.is_available", return_value=False):
                from ai_genai_image_generation.device import get_device

                device = get_device()
                assert device == torch.device("cpu")

    def test_auto_selects_cuda_when_available(self, sample_config_dir):
        """Verify auto mode selects CUDA when GPU is available."""
        # Override config to use "auto" mode
        config_file = sample_config_dir / "config" / "settings.yaml"
        content = config_file.read_text(encoding="utf-8")
        content = content.replace('preferred: "cpu"', 'preferred: "auto"')
        config_file.write_text(content, encoding="utf-8")

        with patch(
            "ai_genai_image_generation.config.Settings._find_config_dir",
            return_value=sample_config_dir / "config",
        ):
            # Mock CUDA as available
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.get_device_name", return_value="Test GPU"):
                    from ai_genai_image_generation.device import get_device

                    device = get_device()
                    assert device == torch.device("cuda")
