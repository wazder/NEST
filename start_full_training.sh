#!/bin/bash

# NEST Full Training for M2 Air
# Uses MPS (Metal Performance Shaders) GPU acceleration
# Estimated time: 4-8 hours (vs 2-3 days on CPU)

echo "======================================================================"
echo " NEST: Full Training on Real ZuCo - M2 Air Optimized"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - Device: M2 Air with MPS (GPU acceleration)"
echo "  - Epochs: 100"
echo "  - Dataset: Real ZuCo (~20,000 samples)"
echo "  - Estimated time: 4-8 hours"
echo ""
echo "IMPORTANT:"
echo "  - Keep MacBook plugged in"
echo "  - Don't close this terminal"
echo "  - Results: results/real_zuco_*/"
echo ""
read -p "Start training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting GPU-accelerated training..."
echo ""

# Activate virtual environment
source .venv-1/bin/activate

# Start training
python scripts/train_with_real_zuco.py --epochs 100

echo ""
echo "======================================================================"
echo " Training Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check results: ls -lh results/real_zuco_*/"
echo "  2. Verify metrics: python scripts/verify_results.py"
echo "  3. Generate figures: python scripts/generate_figures.py"
echo ""
