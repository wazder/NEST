# NEST — Neural EEG Sequence Transducer

## Project
Python ML project for EEG signal processing and sequence transduction.

## Stack
- Python 3.9+ (check .python-version or pyproject.toml)
- ML: PyTorch (primary), scikit-learn
- M2 optimization available (M2_OPTIMIZATION_INFO.sh)

## Commands
- `make install` — install dependencies
- `make install-dev` — install with dev dependencies
- `make test` — full test suite
- `make test-fast` — fast tests only
- `make test-parallel` — parallel test run
- `make test-unit` / `make test-integration` — targeted tests
- `make coverage` — test coverage report
- `make lint` — lint check
- `make format` — auto format
- `make type-check` — mypy/pyright
- `make pre-commit` — run all pre-commit hooks

## Code Style
- Type hints on all function signatures
- Use pathlib over os.path
- f-strings only
- No bare try/except — only at boundaries
