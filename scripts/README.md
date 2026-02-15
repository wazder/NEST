# Scripts Directory

This directory contains various utility scripts for the NEST project.

## Training Scripts

### 🎯 train_with_real_zuco.py (MAIN - Use This!)
**Primary training script for real ZuCo dataset**

Loads real ZuCo .mat files and trains models on actual EEG data.

```bash
# Quick test (30 seconds, validates pipeline)
python scripts/train_with_real_zuco.py --quick-test

# Full training (2-3 days, publication quality)
python scripts/train_with_real_zuco.py --epochs 100

# Custom settings
python scripts/train_with_real_zuco.py --epochs 50 --batch-size 32
```

**Status**: ✅ Working with real ZuCo data (66GB, 53 .mat files)

### train_real_zuco.py (Deprecated)
Old training script that falls back to synthetic data. Use `train_with_real_zuco.py` instead.

### run_quick_demo.py
Runs a quick demo with synthetic data for testing. Used for initial validation.

### run_full_pipeline.py
End-to-end pipeline: training → verification → figure generation.

## Data Scripts

### verify_zuco_data.py
Verifies ZuCo dataset integrity.

```bash
python scripts/verify_zuco_data.py
```

Output:
- Number of .mat files found
- File sizes
- Data directory structure
- Validation status

### inspect_zuco_mat.py
Inspects the structure of ZuCo .mat files.

```bash
python scripts/inspect_zuco_mat.py
```

Shows:
- Top-level keys in .mat file
- sentenceData structure
- Field types and shapes
- Sample data extraction

### generate_synthetic_data.py
Generates synthetic EEG data for testing when real ZuCo is not available.

```bash
python scripts/generate_synthetic_data.py
```

Creates ~243MB of synthetic data in `data/raw/zuco/`

## Evaluation Scripts

### verify_results.py
Verifies training results against expected metrics.

```bash
python scripts/verify_results.py --results results/real_zuco_*/results.json
```

Checks:
- WER (Word Error Rate)
- CER (Character Error Rate)  
- BLEU scores
- Compares against baseline expectations

### generate_figures.py
Generates publication-ready figures.

```bash
python scripts/generate_figures.py --results results/real_zuco_*/
```

Creates 6 figures:
1. Model architecture diagram
2. Model comparison (WER/CER/BLEU)
3. Training curves
4. Subject performance
5. Ablation study
6. Optimization comparison

Output: `papers/figures/*.pdf`

## Download Scripts (Note: OSF doesn't allow automated downloads)

### download_zuco.py
Attempted automated download - doesn't work due to OSF limitations.

### try_download_zuco.py
Alternative download approaches - also blocked by OSF.

### download_zuco_manual.sh
Shell script with manual download instructions.

**Recommendation**: Download ZuCo manually from OSF website, see `HOW_TO_DOWNLOAD_ZUCO.md`

## Shell Scripts

### ../RUN_EVERYTHING.sh
One-command pipeline runner (from project root).

```bash
./RUN_EVERYTHING.sh
```

## Usage Examples

### Complete Workflow

```bash
# 1. Verify you have the data
python scripts/verify_zuco_data.py

# 2. Quick test to validate pipeline
python scripts/train_with_real_zuco.py --quick-test

# 3. Full training
python scripts/train_with_real_zuco.py --epochs 100

# 4. Verify results
python scripts/verify_results.py --results results/real_zuco_*/results.json

# 5. Generate figures
python scripts/generate_figures.py --results results/real_zuco_*/
```

### Debugging

```bash
# Check data format
python scripts/inspect_zuco_mat.py

# Verify data integrity
python scripts/verify_zuco_data.py

# Test with minimal data
python scripts/train_with_real_zuco.py --quick-test
```

## Script Status Summary

| Script | Status | Purpose |
|--------|--------|---------|
| **train_with_real_zuco.py** | ✅ Working | Main training (USE THIS) |
| verify_zuco_data.py | ✅ Working | Data validation |
| inspect_zuco_mat.py | ✅ Working | Data inspection |
| verify_results.py | ✅ Working | Results validation |
| generate_figures.py | ✅ Working | Figure generation |
| run_quick_demo.py | ✅ Working | Synthetic demo |
| generate_synthetic_data.py | ✅ Working | Synthetic data gen |
| train_real_zuco.py | ⚠️ Deprecated | Use train_with_real_zuco.py |
| download_zuco.py | ❌ Blocked | OSF doesn't allow automation |

## Configuration

Most scripts read configuration from:
- `configs/model.yaml` - Model architecture settings
- `configs/preprocessing.yaml` - Preprocessing parameters

## Output Locations

- **Training results**: `results/real_zuco_YYYYMMDD_HHMMSS/`
- **Model checkpoints**: `results/*/checkpoints/*.pt`
- **Figures**: `papers/figures/*.pdf`
- **Logs**: `results/*/training.log` (if enabled)

## Environment

All scripts should be run from the project root with the virtual environment activated:

```bash
cd /Users/wazder/Documents/GitHub/NEST
source .venv/bin/activate
python scripts/[script_name].py
```

## Support

For issues:
1. Check this README
2. See script docstrings (`python scripts/[name].py --help`)
3. Review REAL_ZUCO_STATUS.md
4. Check docs/USAGE.md

---

**Last Updated**: February 16, 2026
