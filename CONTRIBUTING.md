# Contributing to NEST

Thank you for your interest in contributing to NEST (Neural EEG Sequence Transducer)! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submission Guidelines](#submission-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs. actual behavior**
- **Environment details** (OS, Python version, GPU/CPU)
- **Minimal reproducible example**
- **Error messages and logs**

Use the bug report template when creating the issue.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case and motivation**
- **Proposed implementation** (if applicable)
- **Potential alternatives** considered

### Contributing Code

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Installation

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/NEST.git
cd NEST
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. Install in editable mode:
```bash
pip install -e .
```

5. Verify installation:
```bash
python -m pytest tests/
```

### Development Dependencies

Create `requirements-dev.txt` with:
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
isort>=5.12.0
pre-commit>=3.0.0
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized using `isort`
- **Formatting**: Automated with `black`

### Code Formatting

Use `black` for consistent formatting:
```bash
black src/ tests/ examples/
```

Use `isort` for import sorting:
```bash
isort src/ tests/ examples/
```

### Linting

Run `flake8` before committing:
```bash
flake8 src/ tests/ --max-line-length=100
```

### Type Hints

Use type hints for all function signatures:
```python
def preprocess_eeg(
    data: np.ndarray,
    sampling_rate: float,
    filter_range: Tuple[float, float]
) -> np.ndarray:
    """Process EEG data with filtering."""
    pass
```

Run `mypy` for type checking:
```bash
mypy src/
```

### Documentation

All public functions, classes, and modules should have docstrings:

```python
def compute_wer(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute Word Error Rate (WER) between predictions and references.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
        
    Returns:
        Word Error Rate as a percentage (0-100)
        
    Raises:
        ValueError: If predictions and references have different lengths
        
    Example:
        >>> predictions = ["hello world"]
        >>> references = ["hello word"]
        >>> wer = compute_wer(predictions, references)
        >>> print(f"WER: {wer:.2f}%")
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory mirroring `src/` structure
- Use `pytest` framework
- Aim for >80% code coverage
- Test both success and failure cases
- Use fixtures for common setup

Example test:
```python
import pytest
import numpy as np
from src.preprocessing.filtering import bandpass_filter

class TestBandpassFilter:
    def test_filter_shape_preservation(self):
        """Test that filtering preserves data shape."""
        data = np.random.randn(105, 1000)  # channels x time
        filtered = bandpass_filter(data, fs=500, lowcut=0.5, highcut=50)
        assert filtered.shape == data.shape
        
    def test_filter_frequency_response(self):
        """Test that filter attenuates out-of-band frequencies."""
        # Implementation
        pass
        
    def test_invalid_cutoff_frequencies(self):
        """Test error handling for invalid frequencies."""
        data = np.random.randn(105, 1000)
        with pytest.raises(ValueError):
            bandpass_filter(data, fs=500, lowcut=60, highcut=50)
```

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html tests/
```

Run specific test file:
```bash
pytest tests/test_filtering.py
```

Run specific test:
```bash
pytest tests/test_filtering.py::TestBandpassFilter::test_filter_shape_preservation
```

### Test Coverage

Maintain high test coverage:
- **Minimum**: 70% overall coverage
- **Target**: 80%+ coverage for core modules
- **Critical**: 90%+ for preprocessing and model code

View coverage report:
```bash
pytest --cov=src --cov-report=term-missing tests/
```

## Submission Guidelines

### Git Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make commits with clear messages:
```bash
git commit -m "Add bandpass filter for EEG preprocessing"
```

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

3. Keep commits focused and atomic
4. Rebase on `main` before submitting

### Pull Request Process

1. **Update documentation** for new features
2. **Add tests** for new functionality
3. **Ensure CI passes** (all tests, linting, type checking)
4. **Update CHANGELOG.md** with changes
5. **Request review** from maintainers

### Pull Request Template

Your PR should include:
- **Description**: What changes does this PR introduce?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Checklist**:
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] Code formatted with `black`
  - [ ] Linting passes (`flake8`)
  - [ ] Type checking passes (`mypy`)
  - [ ] All tests pass

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Follow Google/NumPy docstring style
- Include examples in docstrings
- Document parameters, return values, and exceptions

### User Documentation

When adding features, update:
- `README.md` if it affects basic usage
- `docs/USAGE.md` for detailed usage examples
- `docs/API.md` for API reference
- Appropriate phase documentation files

### Examples

Add usage examples to `examples/` for significant features:
```python
# examples/05_custom_feature.py
"""
Example: Using Custom Feature

This example demonstrates how to use the new custom feature
for EEG preprocessing.
"""

from src.preprocessing import CustomFeature

# Your example code here
```

## Development Best Practices

### Performance

- Profile code for performance bottlenecks
- Use vectorized NumPy operations
- Avoid unnecessary copies of large arrays
- Use generators for large datasets

### Memory Management

- Be mindful of GPU memory usage
- Clear unnecessary tensors with `del`
- Use `torch.cuda.empty_cache()` after large operations
- Implement batch processing for large datasets

### Reproducibility

- Set random seeds for reproducible results
- Document all hyperparameters
- Version control configuration files
- Include environment specifications

### Error Handling

- Use specific exception types
- Provide informative error messages
- Validate inputs at function boundaries
- Log warnings for non-critical issues

## Getting Help

- **Documentation**: Check [docs/](docs/) for detailed guides
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Acknowledged in release notes
- Credited in academic papers (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions! Create an issue with the "question" label or start a discussion.

---

Thank you for contributing to NEST! 🧠🚀
