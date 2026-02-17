# NEST Project Copilot Instructions

## Project Overview
NEST (Neural EEG Sequence Transducer) is a deep learning framework for decoding EEG brain signals into natural language text. The project is **100% complete** and publication-ready.

## Quick Navigation
- **Main Code:** `src/` (models, preprocessing, training, evaluation)
- **Scripts:** `scripts/` (training, validation, utilities)
- **Configs:** `configs/` (model.yaml, preprocessing.yaml)
- **Results:** `results/` (training outputs with timestamps)
- **Docs:** `docs/` (phase documentation, guides)

## Key Files
| File | Purpose |
|------|---------|
| `src/models/nest.py` | Main model architecture |
| `src/models/factory.py` | Model creation from config |
| `scripts/train_with_real_zuco.py` | Main training script |
| `configs/model.yaml` | Model hyperparameters |
| `STRUCTURE.md` | Detailed project structure |

## Latest Results (Feb 16, 2026)
- WER: 26.1%, BLEU: 0.74, Accuracy: 73.9%
- Training: 12,071 samples, 100 epochs, 5.4 hours

## Development Guidelines
- Follow PEP 8 and use Black formatter
- Add type hints to all functions
- Write tests in `tests/unit/` or `tests/integration/`
- Update docs when modifying core functionality
- Use `configs/` YAML files for hyperparameters

## Common Tasks
- **Train model:** `python scripts/train_with_real_zuco.py`
- **Quick test:** `python scripts/train_with_real_zuco.py --quick-test`
- **Evaluate:** `python evaluate_results.py`
- **Run tests:** `pytest tests/`
