# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 10GB for dependencies and datasets

### Software
- **Operating System**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.0+ (if using GPU)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/wazder/NEST.git
cd NEST

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Method 2: Using conda

```bash
# Clone the repository
git clone https://github.com/wazder/NEST.git
cd NEST

# Create conda environment
conda create -n nest python=3.10
conda activate nest

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Method 3: Development Installation

For contributors or those who want to modify the codebase:

```bash
# Clone the repository
git clone https://github.com/wazder/NEST.git
cd NEST

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with development dependencies
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Dataset Setup

### ZuCo Dataset

The NEST project uses the ZuCo dataset. Follow these steps to download and prepare it:

```bash
# Create data directories
mkdir -p data/raw data/processed

# Download ZuCo dataset (manual download required)
# Visit: https://osf.io/q3zws/
# Download ZuCo 1.0 and/or ZuCo 2.0
# Extract to data/raw/zuco/

# Verify dataset structure
ls data/raw/zuco/
# Expected: task1-SR/, task2-NR/, task3-TSR/, etc.
```

### Preprocess Dataset

```bash
# Run preprocessing pipeline
python -m src.preprocessing.pipeline \
    --config configs/preprocessing.yaml \
    --input data/raw/zuco/ \
    --output data/processed/zuco/

# This will create:
# - data/processed/zuco/train/
# - data/processed/zuco/val/
# - data/processed/zuco/test/
```

## Verify Installation

Run the test suite to ensure everything is properly installed:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific phase tests
python test_phase2.py
```

## Common Installation Issues

### Issue: CUDA not found

**Solution**: Install CUDA toolkit matching your PyTorch version:

```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching CUDA toolkit from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
```

### Issue: MNE installation fails

**Solution**: Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# macOS
brew install hdf5

# Then reinstall
pip install --force-reinstall mne
```

### Issue: Out of memory during preprocessing

**Solution**: Adjust batch size in configuration:

```yaml
# configs/preprocessing.yaml
preprocessing:
  batch_size: 16  # Reduce from 32
```

### Issue: ImportError for torch extensions

**Solution**: Rebuild PyTorch from source or use compatible version:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install specific version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Next Steps

After successful installation:

1. **Explore Examples**: Check [examples/](../examples/) directory
2. **Read Documentation**: See [docs/](../docs/) for detailed guides
3. **Run Tutorials**: Try [notebooks/](../notebooks/) for interactive tutorials
4. **Train Models**: Follow [USAGE.md](USAGE.md) for training instructions

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/wazder/NEST/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/wazder/NEST/discussions)
- **Documentation**: Check [docs/](../docs/) for detailed guides

## Environment Variables

Optional environment variables for configuration:

```bash
# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Disable CUDA (CPU-only mode)
export CUDA_VISIBLE_DEVICES=-1

# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/cache

# Enable debugging
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## Docker Installation (Optional)

For containerized deployment:

```bash
# Build Docker image
docker build -t nest:latest .

# Run container
docker run -it --gpus all -v $(pwd):/workspace nest:latest

# Run with Jupyter
docker run -it --gpus all -p 8888:8888 -v $(pwd):/workspace nest:latest jupyter lab
```

Note: Dockerfile creation is planned for future releases.
