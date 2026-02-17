# 💡 RECOMMENDATIONS & BEST PRACTICES

Suggestions for improving code quality, maintainability, and project organization.

## 1. Immediate Actions (Critical Fixes)

### Priority 1: Fix Runtime Errors 🔴

**Estimated Time**: 2-3 hours

#### 1.1 Fix Type Annotation
```python
# File: src/data/zuco_dataset.py, line 174
# Change:
def get_dataset_info(self) -> Dict[str, any]:
# To:
def get_dataset_info(self) -> Dict[str, Any]:
```

#### 1.2 Standardize Task Directory Naming
Choose one format and update all scripts:

**Option A** (Recommended): Use underscores everywhere
```python
# Update these files:
# - scripts/train_with_real_zuco.py
# - scripts/inspect_zuco_mat.py
# - scripts/download_zuco_manual.sh

# From:
task_dirs = ['task1-SR', 'task2-NR', 'task3-TSR']
# To:
task_dirs = ['task1_SR', 'task2_NR', 'task3_TSR']
```

**Option B**: Use hyphens everywhere
```python
# Update these files:
# - scripts/download_zuco.py
# - scripts/generate_synthetic_data.py
# - configs/preprocessing.yaml

# From:
task_dirs = ['task1_SR', 'task2_NR', 'task3_TSR']
# To:
task_dirs = ['task1-SR', 'task2-NR', 'task3-TSR']
```

**Recommendation**: Use underscores (Option A) - more Pythonic.

#### 1.3 Fix Hardcoded User Path
```markdown
# File: docs/guides/RUN_ME_FIRST.md, line 8
# Change:
cd /Users/wazder/Documents/GitHub/NEST
# To:
cd /path/to/your/NEST
# Or better:
# Navigate to the NEST project directory
cd NEST
```

---

## 2. High Priority Improvements

### Priority 2: Configuration Management 🟠

**Estimated Time**: 4-6 hours

#### 2.1 Consolidate Configuration Files

**Step 1**: Migrate all tool configs to `pyproject.toml`

```toml
# pyproject.toml

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmarks",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*conftest.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Step 2**: Remove tool configurations from `setup.cfg`

Keep only packaging metadata:
```ini
# setup.cfg
[metadata]
name = nest-eeg
version = attr: src.__version__

[options]
packages = find:
python_requires = >=3.8
install_requires = file: requirements.txt
```

---

#### 2.2 Separate Production and Development Dependencies

**requirements.txt** (Production only):
```text
# ====================
# Core ML Framework
# ====================
torch>=2.0.0,<3.0.0
transformers>=4.30.0,<5.0.0
lightning>=2.0.0,<3.0.0

# ====================
# Scientific Computing
# ====================
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0
scikit-learn>=1.0.0,<2.0.0

# ====================
# Data Processing
# ====================
pandas>=1.3.0,<3.0.0
h5py>=3.7.0
mne>=1.0.0

# ====================
# Utilities
# ====================
pyyaml>=6.0
tqdm>=4.62.0
wandb>=0.12.0
omegaconf>=2.1.0

# ====================
# Tokenization
# ====================
sentencepiece>=0.1.96
```

**requirements-dev.txt** (Development only):
```text
# Install production requirements first
-r requirements.txt

# ====================
# Testing
# ====================
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
pytest-timeout>=2.1.0
pytest-benchmark>=4.0.0

# ====================
# Code Quality
# ====================
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=1.0.0
pylint>=2.17.0

# ====================
# Security
# ====================
bandit>=1.7.0
safety>=2.3.0

# ====================
# Analysis
# ====================
radon>=5.1.0
coverage>=7.0.0

# ====================
# Documentation
# ====================
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0

# ====================
# Development Tools
# ====================
ipython>=8.0.0
jupyter>=1.0.0
pre-commit>=2.20.0
```

---

#### 2.3 Create Centralized Path Configuration

**config/paths.yaml** (new file):
```yaml
# Default paths configuration
# Override with environment variables or command-line args

data:
  raw: "${DATA_DIR:data/raw}"
  processed: "${DATA_DIR:data/processed}"
  zuco: "${ZUCO_DATA_DIR:data/raw/zuco}"

results:
  base: "${RESULTS_DIR:results}"
  checkpoints: "${RESULTS_DIR:results}/checkpoints"
  logs: "${RESULTS_DIR:results}/logs"
  figures: "${RESULTS_DIR:results}/figures"

models:
  pretrained: "${MODEL_DIR:models/pretrained}"
  checkpoints: "${MODEL_DIR:models/checkpoints}"

cache:
  dir: "${CACHE_DIR:.cache}"
```

**src/utils/paths.py** (new file):
```python
"""Centralized path configuration."""
from pathlib import Path
import os
from omegaconf import OmegaConf

# Load default paths
_config_path = Path(__file__).parent.parent.parent / "config" / "paths.yaml"
_paths = OmegaConf.load(_config_path)

def get_path(key: str, default: str = None) -> Path:
    """Get path from config with environment variable support."""
    try:
        path_str = OmegaConf.select(_paths, key)
        if path_str is None and default:
            path_str = default
        # Expand environment variables
        path_str = os.path.expandvars(path_str)
        return Path(path_str)
    except Exception:
        if default:
            return Path(default)
        raise

# Commonly used paths
DATA_DIR = get_path("data.raw")
ZUCO_DATA_DIR = get_path("data.zuco")
RESULTS_DIR = get_path("results.base")
```

**Usage in scripts**:
```python
# Old way (hardcoded):
data_dir = "data/raw/zuco"

# New way:
from src.utils.paths import ZUCO_DATA_DIR
data_dir = ZUCO_DATA_DIR
```

---

### Priority 3: Documentation Improvements 🟠

**Estimated Time**: 1-2 days

#### 3.1 Create Missing Documentation Files

**docs/TROUBLESHOOTING.md**:
```markdown
# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### CUDA/GPU not detected
**Symptoms**: PyTorch doesn't detect GPU...
**Solution**: ...

#### Out of memory errors
**Symptoms**: CUDA out of memory...
**Solution**: ...

### Data Download Issues

#### Download fails or hangs
**Symptoms**: ...
**Solution**: ...

### Training Issues

#### Loss not decreasing
**Symptoms**: ...
**Solution**: ...
```

**docs/SUBJECT_ADAPTATION.md**:
```markdown
# Subject Adaptation Guide

Guide for adapting models to new subjects...
```

Similar for other missing docs:
- `docs/HYPERPARAMETER_TUNING.md`
- `docs/CUSTOM_DATASET.md`
- `docs/DEPLOYMENT.md`

---

#### 3.2 Standardize Terminology

Create a **docs/GLOSSARY.md**:
```markdown
# Glossary

## Data Terms
- **EEG signals**: Raw electrical brain activity data
- **EEG recordings**: Complete dataset files
- **EEG features**: Processed/extracted features

## Model Terms
- **NEST-RNN-T**: Display name for RNN Transducer model
- **nest_rnn_t**: Configuration file identifier
- **NestRNNTransducer**: Python class name

## Metrics
- **WER (Word Error Rate)**: Percentage of incorrectly predicted words
- **CER (Character Error Rate)**: For Turkish evaluation
- **BLEU**: Translation quality metric
```

Then reference glossary in documentation:
```markdown
We evaluate using [WER](GLOSSARY.md#wer) (Word Error Rate)...
```

---

#### 3.3 Clarify Dataset Sizes

Add section to **docs/DATA_REQUIREMENTS.md** (new or in existing doc):
```markdown
# Data Storage Requirements

## ZuCo Dataset Sizes by Stage

| Stage | Size | Description |
|-------|------|-------------|
| **Compressed download** | ~15 GB | Original .zip/.mat files |
| **Extracted raw data** | ~18 GB | Uncompressed .mat files |
| **After preprocessing** | 5-10 GB | Processed tensors |
| **With augmentation** | 30-40 GB | Including augmented samples |
| **Working space** | +10-20 GB | Temporary files during processing |
| **Total recommended** | **50-70 GB** | Safe total space allocation |

## Task-Specific Sizes

- **task1_SR** (Sentence Reading): ~6 GB
- **task2_NR** (Normal Reading): ~7 GB  
- **task3_TSR** (Task-Specific Reading): ~5 GB

## Storage Recommendations

- **Minimum**: 25 GB free (single task)
- **Recommended**: 50 GB free (all tasks)
- **Optimal**: 100 GB+ (with experiments/results)
```

---

## 3. Medium Priority Enhancements

### Priority 4: Error Handling & Robustness 🟡

**Estimated Time**: 1 week

#### 4.1 Add Comprehensive Error Handling

**Template for all scripts**:
```python
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point with error handling."""
    try:
        # Validate inputs
        validate_environment()
        
        # Run main logic
        result = run_training()
        
        logger.info("✓ Training completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"✗ Required file not found: {e}")
        logger.info("Did you download the data? See docs/guides/HOW_TO_DOWNLOAD_ZUCO.md")
        return 1
        
    except RuntimeError as e:
        logger.error(f"✗ Runtime error: {e}")
        logger.info("Check TROUBLESHOOTING.md for common issues")
        return 2
        
    except KeyboardInterrupt:
        logger.info("✗ Interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

#### 4.2 Implement Module Fusion or Add Warning

**Option 1**: Implement fusion:
```python
# src/evaluation/quantization.py
def _fuse_modules(self):
    """Fuse modules for better quantization."""
    from torch.quantization import fuse_modules
    
    # Define fusion patterns
    fusion_patterns = [
        ['conv1', 'bn1', 'relu'],
        ['conv2', 'bn2'],
    ]
    
    for pattern in fusion_patterns:
        try:
            fuse_modules(self.model, pattern, inplace=True)
            logger.info(f"✓ Fused modules: {' -> '.join(pattern)}")
        except Exception as e:
            logger.warning(f"Could not fuse {pattern}: {e}")
```

**Option 2**: Add warning:
```python
def _fuse_modules(self):
    """Fuse modules for better quantization."""
    warnings.warn(
        "Module fusion not yet implemented. "
        "This may result in suboptimal quantization performance.",
        UserWarning
    )
```

---

### Priority 5: Testing Enhancements 🟡

**Estimated Time**: 3-4 days

#### 5.1 Add Script Tests

**tests/unit/test_scripts.py** (new file):
```python
"""Tests for scripts functionality."""
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

def test_download_script_import():
    """Test that download script can be imported."""
    import download_zuco
    assert hasattr(download_zuco, 'download_task')

def test_task_naming_consistency():
    """Test that all scripts use consistent task naming."""
    import download_zuco
    import train_with_real_zuco
    
    # Both should use same format
    assert download_zuco.TASK_NAMES[0] == train_with_real_zuco.TASK_NAMES[0]

@patch('subprocess.run')
def test_pipeline_script_error_handling(mock_run):
    """Test that pipeline handles subprocess errors."""
    mock_run.return_value = MagicMock(returncode=1)
    
    from run_full_pipeline import run_pipeline
    
    result = run_pipeline()
    assert result != 0  # Should return error code
```

---

#### 5.2 Add Configuration Tests

**tests/unit/test_configs.py** (new file):
```python
"""Tests for configuration files."""
import pytest
from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

def test_model_config_valid():
    """Test that model.yaml is valid."""
    config = OmegaConf.load(CONFIG_DIR / "model.yaml")
    assert "nest_rnn_t" in config
    assert "nest_attention" in config

def test_dimension_compatibility():
    """Test that model dimensions are compatible."""
    config = OmegaConf.load(CONFIG_DIR / "model.yaml")
    
    # For nest_rnn_t
    model = config.nest_rnn_t
    assert model.temporal_encoder.input_size == model.spatial_cnn.out_channels
    
    if model.temporal_encoder.bidirectional:
        expected_dim = model.temporal_encoder.hidden_size * 2
    else:
        expected_dim = model.temporal_encoder.hidden_size
    
    assert model.joint.encoder_dim == expected_dim

def test_preprocessing_config_valid():
    """Test that preprocessing.yaml is valid."""
    config = OmegaConf.load(CONFIG_DIR / "preprocessing.yaml")
    assert "tasks" in config
    assert len(config.tasks) > 0
```

---

## 4. Long-Term Improvements

### Priority 6: Architecture Enhancements 🟢

**Estimated Time**: Ongoing

#### 6.1 Add Dependency Injection

Instead of hardcoded imports, use dependency injection:

```python
# src/training/trainer.py

class Trainer:
    def __init__(
        self, 
        model,
        optimizer,
        loss_fn,
        metrics=None,
        logger=None,
        # Dependencies injected
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.logger = logger or logging.getLogger(__name__)
```

**Benefits**:
- Easier testing (mock dependencies)
- More flexible configuration
- Better separation of concerns

---

#### 6.2 Add Abstract Base Classes

Define interfaces for key components:

```python
# src/models/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoders."""
    
    @abstractmethod
    def forward(self, x):
        """Encode input sequence."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Get output dimension."""
        pass

class BaseDecoder(nn.Module, ABC):
    """Abstract base class for decoders."""
    
    @abstractmethod
    def forward(self, encoder_outputs, targets=None):
        """Decode encoder outputs."""
        pass
```

**Benefits**:
- Clear contracts
- Type checking
- Documentation

---

### Priority 7: Performance Optimization 🟢

#### 7.1 Profile and Optimize Bottlenecks

```python
# Add profiling decorators
from functools import wraps
import time

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@profile
def preprocess_data(data):
    # ... implementation
    pass
```

---

#### 7.2 Add Caching for Expensive Operations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_pretrained_embeddings(path):
    """Load embeddings with caching."""
    return torch.load(path)
```

---

### Priority 8: Monitoring & Observability 🟢

#### 8.1 Enhanced Logging

```python
# src/utils/logging.py
import logging
import sys
from pathlib import Path

def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """Setup logging configuration."""
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
```

---

#### 8.2 Progress Tracking

```python
from tqdm import tqdm
import logging

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
```

---

## 5. Best Practices Summary

### Code Quality Checklist

- [ ] **Type hints** on all public functions
- [ ] **Docstrings** with examples
- [ ] **Error handling** for edge cases
- [ ] **Logging** instead of print statements
- [ ] **Tests** for new functionality
- [ ] **Configuration** over hardcoding
- [ ] **Validation** of inputs
- [ ] **Documentation** updates

### Before Each Commit

```bash
# Run code quality checks
make lint        # or: black src/ && isort src/ && flake8 src/
make typecheck   # or: mypy src/
make test        # or: pytest tests/
make security    # or: bandit -r src/ && safety check
```

### Before Each Release

- [ ] Update `CHANGELOG.md`
- [ ] Update `VERSION` file
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create git tag
- [ ] Build and test package

---

## 6. Tooling Recommendations

### Development Tools

```bash
# Install recommended tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Install additional tools
pip install ipdb  # Better debugger
pip install ptpython  # Better REPL
pip install jupyterlab  # Notebooks
```

### VSCode Extensions (if using VSCode)

- Python (Microsoft)
- Pylance
- Python Test Explorer
- GitLens
- Markdown All in One
- YAML

### PyCharm Plugins (if using PyCharm)

- Requirements
- .ignore
- Rainbow Brackets
- String Manipulation

---

## 7. Action Plan Summary

### Week 1: Critical Fixes
- Day 1: Fix type annotation, task naming, user path
- Day 2: Consolidate configuration files
- Day 3: Separate production/dev dependencies
- Day 4: Create path configuration system
- Day 5: Add basic error handling

### Week 2: High Priority
- Day 1-2: Create missing documentation
- Day 3: Standardize terminology
- Day 4: Clarify dataset sizes
- Day 5: Review and test all changes

### Week 3-4: Medium Priority
- Week 3: Enhance error handling across all scripts
- Week 4: Add comprehensive tests

### Ongoing: Long-term Improvements
- Continuous refactoring
- Performance optimization
- Enhanced monitoring
- Community feedback integration

---

## Conclusion

The NEST project is already at a high quality level. These recommendations focus on:

1. **Fixing critical issues** that cause runtime errors
2. **Improving maintainability** through better configuration
3. **Enhancing robustness** with error handling and tests
4. **Standardizing practices** for consistency
5. **Preparing for scale** with better architecture

Prioritize based on:
- User impact (critical issues first)
- Development efficiency (configuration, tooling)
- Long-term maintainability (tests, documentation)
