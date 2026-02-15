#!/bin/bash
# Simple script to download ZuCo dataset from OSF
#
# ZuCo dataset is available at: https://osf.io/q3zws/
# This script downloads the three main task files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/raw/zuco"

echo "======================================================================"
echo "ZuCo Dataset Download"
echo "======================================================================"
echo "Data directory: $DATA_DIR"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "ZuCo Dataset Download Information"
echo "======================================================================"
echo ""
echo "The ZuCo (Zurich Cognitive Language Processing Corpus) dataset"
echo "contains EEG recordings from 12 subjects reading text."
echo ""
echo "Dataset size: ~18 GB"
echo "Download source: Open Science Framework (OSF)"
echo "URL: https://osf.io/q3zws/"
echo ""
echo "======================================================================"
echo ""
echo "MANUAL DOWNLOAD REQUIRED:"
echo ""
echo "Due to OSF authentication requirements, please download manually:"
echo ""
echo "1. Visit: https://osf.io/q3zws/"
echo "2. Download the following files:"
echo "   - ZuCo_v1_Task1-SR.zip (Sentence Reading)"
echo "   - ZuCo_v1_Task2-NR.zip (Normal Reading)"  
echo "   - ZuCo_v1_Task3-TSR.zip (Task-Specific Reading)"
echo ""
echo "3. Place the .zip files in: $DATA_DIR"
echo ""
echo "4. Run this script again with --extract flag to extract"
echo ""
echo "======================================================================"
echo ""

# Check if user wants to extract existing files
if [[ "$1" == "--extract" ]]; then
    echo "Extracting ZuCo dataset files..."
    
    for zipfile in *.zip; do
        if [[ -f "$zipfile" ]]; then
            echo "  Extracting $zipfile..."
            unzip -q "$zipfile"
            echo "  ✓ $zipfile extracted"
        fi
    done
    
    echo ""
    echo "✓ Extraction complete!"
    echo ""
    echo "Data structure:"
    ls -lh
    
elif [[ "$1" == "--verify" ]]; then
    echo "Verifying dataset structure..."
    echo ""
    
    total_size=$(du -sh . | cut -f1)
    echo "Total size: $total_size"
    
    mat_files=$(find . -name "*.mat" | wc -l | tr -d ' ')
    echo "MATLAB files found: $mat_files"
    
    if [[ $mat_files -gt 0 ]]; then
        echo ""
        echo "✓ ZuCo dataset appears to be downloaded correctly!"
    else
        echo ""
        echo "✗ No .mat files found. Dataset may not be downloaded yet."
        echo "   Please download from: https://osf.io/q3zws/"
    fi
    
else
    echo "Alternative: Use wget to download (if OSF allows):"
    echo ""
    echo "  wget -O ZuCo_Task1.zip 'https://osf.io/download/q3zws/'"
    echo "  wget -O ZuCo_Task2.zip 'https://osf.io/download/sjezn/'"
    echo "  wget -O ZuCo_Task3.zip 'https://osf.io/download/2urht/'"
    echo ""
    echo "Then run: $0 --extract"
    echo ""
    echo "======================================================================"
fi

echo ""
echo "Next steps after download:"
echo "  1. Extract: $0 --extract"
echo "  2. Verify: $0 --verify"
echo "  3. Preprocess: python scripts/train_zuco_full.py --preprocess-only"
echo ""
