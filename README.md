# AI Generative Image Generation

A collection of generative deep learning models for image synthesis, implemented in PyTorch and trained on the MNIST handwritten digit dataset. This project demonstrates progressively sophisticated generative architectures from basic autoencoders to diffusion models.

## Models Implemented

| Model | Description | Module |
|-------|-------------|--------|
| **Autoencoder** | Convolutional autoencoder for image compression and reconstruction | `autoencoder/model.py` |
| **VAE** | Variational Autoencoder with probabilistic latent space | `autoencoder/variable_auto_encoder.py` |
| **Conditional VAE** | Class-conditioned VAE for targeted digit generation | `conditional/cvae.py` |
| **WGAN** | Wasserstein GAN with weight clipping for stable adversarial training | `gan/wgan.py` |
| **DDPM** | Denoising Diffusion Probabilistic Model with UNet backbone | `diffusion/forward_diffusion.py` |

## Project Structure

```
ai-genai-image-generation/
├── config/
│   └── settings.yaml              # Central configuration (hyperparameters, logging, device)
├── src/
│   └── ai_genai_image_generation/
│       ├── __init__.py            # Package marker with version
│       ├── main.py                # CLI entry points for all models
│       ├── config.py              # Configuration loader (YAML, env overrides)
│       ├── logging_config.py      # Logging setup from configuration
│       ├── device.py              # PyTorch device selection (CPU/CUDA)
│       ├── data_loader.py         # MNIST DataLoader factory
│       ├── autoencoder/
│       │   ├── __init__.py
│       │   ├── model.py           # Autoencoder architecture (Encoder + Decoder)
│       │   ├── trainer.py         # Autoencoder training loop
│       │   └── variable_auto_encoder.py  # VAE model and trainer
│       ├── conditional/
│       │   ├── __init__.py
│       │   └── cvae.py            # Conditional VAE model and trainer
│       ├── diffusion/
│       │   ├── __init__.py
│       │   └── forward_diffusion.py  # DDPM, UNet, and training
│       └── gan/
│           ├── __init__.py
│           └── wgan.py            # WGAN Generator, Critic, and trainer
├── tests/
│   ├── conftest.py                # Shared test fixtures
│   ├── test_config.py            # Configuration tests
│   ├── test_device.py            # Device selection tests
│   ├── test_data_loader.py       # Data loading tests
│   ├── test_autoencoder.py       # Autoencoder model/trainer tests
│   ├── test_vae.py               # VAE model/trainer tests
│   ├── test_cvae.py              # Conditional VAE tests
│   ├── test_wgan.py              # WGAN tests
│   └── test_diffusion.py         # Diffusion model tests
├── data/                          # MNIST dataset (auto-downloaded, gitignored)
├── pyproject.toml                 # Project metadata, dependencies, scripts
├── poetry.lock                    # Locked dependency versions
└── README.md                      # This file
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.12 | Language runtime |
| PyTorch | >= 2.3.0 | Deep learning framework |
| TorchVision | >= 0.18.0 | Datasets and image transforms |
| Matplotlib | >= 3.9.0 | Visualization |
| tqdm | >= 4.66.0 | Progress bars |
| PyYAML | >= 6.0 | Configuration file parsing |
| pytest | >= 8.2.0 | Testing framework (dev) |
| pytest-cov | >= 5.0.0 | Coverage reporting (dev) |
| ruff | >= 0.4.0 | Linting and formatting (dev) |

## Deployment / Setup

### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/docs/#installation) package manager
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-genai-image-generation

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Configuration

All hyperparameters and settings are in `config/settings.yaml`. To override for a specific environment:

1. Set the `APP_ENV` environment variable:
   ```bash
   export APP_ENV=production
   ```

2. Create `config/settings.production.yaml` with overrides:
   ```yaml
   logging:
     level: "WARNING"
     file: "logs/app.log"
   autoencoder:
     batch_size: 128
     num_epochs: 100
   device:
     preferred: "cuda"
   ```

### Running Models

```bash
# Train the Convolutional Autoencoder
poetry run run-auto-encoder

# Train the Variational Autoencoder
poetry run run-vae

# Train the Conditional VAE
poetry run run-cvae

# Train the Wasserstein GAN
poetry run run-wgan

# Train the Diffusion Model (DDPM)
poetry run run-diffusion
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test module
poetry run pytest tests/test_autoencoder.py

# Run with verbose output
poetry run pytest -v

# Run with specific coverage threshold
poetry run pytest --cov-fail-under=80
```

### Linting

```bash
# Check code style
poetry run ruff check src/ tests/

# Auto-fix issues
poetry run ruff check --fix src/ tests/
```

## Logging

Log level is configurable in `config/settings.yaml`:

```yaml
logging:
  level: "DEBUG"    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/training.log"  # Set to null for console-only
```

## Architecture Notes

- **Single Responsibility**: Each class has one job (model definition vs. training logic vs. data loading).
- **Configuration-Driven**: All hyperparameters are externalized to YAML, not hardcoded.
- **Device-Agnostic**: Models run on CPU or GPU based on configuration and availability.
- **Error Handling**: Training loops catch and log runtime errors (e.g., CUDA OOM).
- **Testable**: Trainer classes accept mock data loaders for unit testing without MNIST download.
