#!/usr/bin/env bash
#
# Quick Start Script for NEST Training
#
# This script demonstrates how to quickly run NEST training pipeline
# from scratch. Customize as needed for your environment.

set -e  # Exit on error

echo "============================================"
echo "NEST: Quick Start Training Script"
echo "============================================"
echo ""

# Configuration
DOWNLOAD_DATA=${1:-"false"}  # Set to "true" to download ZuCo
OUTPUT_DIR="results/quickstart"
CONFIG_FILE="configs/model.yaml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function
print_step() {
    echo -e "${BLUE}==> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Check dependencies
print_step "Step 1: Checking dependencies..."

if ! command -v python &> /dev/null; then
    echo "Python not found! Please install Python 3.8+"
    exit 1
fi

python_version=$(python --version 2>&1 | awk '{print $2}')
print_success "Found Python $python_version"

# Check if required packages are installed
python -c "import torch" 2>/dev/null || {
    echo "PyTorch not installed. Installing dependencies..."
    pip install -r requirements.txt
}

print_success "Dependencies OK"

# Step 2: Data preparation
if [ "$DOWNLOAD_DATA" = "true" ]; then
    print_step "Step 2: Downloading ZuCo dataset..."
    print_warning "This will download ~5GB and may take 1-2 hours"
    
    python scripts/train_zuco_full.py \
        --download \
        --config $CONFIG_FILE \
        --output $OUTPUT_DIR \
        --no-training
        
    print_success "Dataset downloaded and preprocessed"
else
    print_step "Step 2: Skipping data download (using cached data)"
    print_warning "If you need to download data, run: $0 true"
    
    # Check if processed data exists
    if [ ! -d "data/processed/zuco/train" ]; then
        echo "Error: No preprocessed data found!"
        echo "Please run with data download: $0 true"
        exit 1
    fi
    
    print_success "Found cached processed data"
fi

# Step 3: Train models
print_step "Step 3: Training NEST models..."
echo "This may take 8-12 hours depending on your GPU"
echo ""

python scripts/train_zuco_full.py \
    --config $CONFIG_FILE \
    --output $OUTPUT_DIR

print_success "Training complete!"

# Step 4: Evaluate results
print_step "Step 4: Evaluating models..."

python scripts/evaluate_models.py \
    --results $OUTPUT_DIR \
    --output $OUTPUT_DIR/evaluation

print_success "Evaluation complete!"

# Step 5: Generate report
print_step "Step 5: Generating results report..."

echo ""
echo "============================================"
echo "TRAINING SUMMARY"
echo "============================================"
cat $OUTPUT_DIR/RESULTS_REPORT.md
echo ""

print_success "All done! Results saved to $OUTPUT_DIR/"

# Step 6: Next steps
echo ""
echo "============================================"
echo "NEXT STEPS"
echo "============================================"
echo ""
echo "1. View detailed results:"
echo "   cat $OUTPUT_DIR/results.json"
echo ""
echo "2. View training history for a model:"
echo "   cat $OUTPUT_DIR/checkpoints/nest_conformer/history.json"
echo ""
echo "3. Test inference with a checkpoint:"
echo "   python examples/04_deployment.py --checkpoint $OUTPUT_DIR/checkpoints/nest_conformer/best_model.pt"
echo ""
echo "4. Optimize model for deployment:"
echo "   python scripts/optimize_for_deployment.py --model $OUTPUT_DIR/checkpoints/nest_conformer/best_model.pt"
echo ""
echo "5. Start TensorBoard:"
echo "   tensorboard --logdir $OUTPUT_DIR/logs/"
echo ""

print_success "Training pipeline completed successfully!"
