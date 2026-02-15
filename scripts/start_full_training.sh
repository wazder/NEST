#!/bin/bash

# NEST Full Training Launcher
# Starts complete training on real ZuCo dataset
# Estimated time: 2-3 days on CPU

echo "======================================================================"
echo " NEST: Starting Full Training on Real ZuCo Data"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - Epochs: 100"
echo "  - Dataset: Real ZuCo (66 GB, ~20,000 samples)"
echo "  - Estimated time: 2-3 days"
echo ""
echo "IMPORTANT:"
echo "  - Keep computer running for 2-3 days"
echo "  - Don't close this terminal"
echo "  - Results will be in: results/real_zuco_*/"
echo ""
read -p "Start training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Activate virtual environment
source .venv/bin/activate

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
