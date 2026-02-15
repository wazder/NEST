#!/bin/bash
# ONE-COMMAND FULL PIPELINE
# 
# This script runs everything after you download real ZuCo data

set -e

echo "================================================================================"
echo "NEST: One-Command Pipeline Runner"
echo "================================================================================"
echo ""

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -f ".venv-1/bin/activate" ]; then
    source .venv-1/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No virtual environment found"
    echo "Creating one now..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

echo ""

# Check if real data is available
echo "Checking for ZuCo data..."
PYTHON_CMD=$(which python)

DATA_TYPE=$($PYTHON_CMD -c "
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from run_full_pipeline import check_data_type
data_dir = Path('data/raw/zuco')
if data_dir.exists():
    print(check_data_type(data_dir))
else:
    print('none')
")

echo "Data type: $DATA_TYPE"
echo ""

if [ "$DATA_TYPE" = "none" ]; then
    echo "================================================================================"
    echo "NO DATA FOUND!"
    echo "================================================================================"
    echo ""
    echo "Please either:"
    echo "  1. Download real ZuCo from: https://osf.io/q3zws/"
    echo "     Save to: data/raw/zuco/"
    echo ""
    echo "  2. Generate synthetic test data:"
    echo "     python scripts/generate_synthetic_data.py"
    echo ""
    exit 1
fi

if [ "$DATA_TYPE" = "synthetic" ]; then
    echo "⚠️  USING SYNTHETIC DATA (for testing only)"
    echo "   For publication, download real ZuCo from: https://osf.io/q3zws/"
    echo ""
else
    echo "✓ USING REAL ZuCo DATA (publication-ready)"
    echo ""
fi

# Confirm
read -p "Continue with $DATA_TYPE data? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "================================================================================"
echo "Starting Full Pipeline..."
echo "================================================================================"
echo ""

# Run the pipeline
time $PYTHON_CMD scripts/run_full_pipeline.py

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - Models: results/demo/checkpoints/ (or results/run_*/)"
echo "  - Results: results/demo/results.json"
echo "  - Verification: results/demo/verification_report.md"
echo "  - Figures: papers/figures/"
echo ""
echo "Next steps:"
if [ "$DATA_TYPE" = "synthetic" ]; then
    echo "  1. Download real ZuCo: https://osf.io/q3zws/"
    echo "  2. Run this script again for publication results"
else
    echo "  1. Review: cat results/*/verification_report.md"
    echo "  2. Check figures: open papers/figures/"
    echo "  3. Update paper: papers/NEST_manuscript.md"
    echo "  4. Submit to IEEE EMBC by March 15, 2026"
fi
echo ""
echo "================================================================================"
