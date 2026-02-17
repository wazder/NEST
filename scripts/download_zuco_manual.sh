#!/bin/bash
# Manual ZuCo Dataset Download Guide
# 
# OSF doesn't allow direct programmatic downloads, so manual download is required.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "ZuCo Dataset Manual Download Guide"
echo "================================================================================"
echo ""
echo "The ZuCo dataset must be downloaded manually from Open Science Framework (OSF)."
echo ""
echo -e "${YELLOW}STEP 1: Download from OSF${NC}"
echo "-------------------------------------------------------------------------------"
echo ""
echo "Visit the ZuCo dataset page:"
echo -e "${GREEN}https://osf.io/q3zws/${NC}"
echo ""
echo "On the OSF page, you'll find files for ZuCo 1.0 and ZuCo 2.0:"
echo ""
echo "For this project, download:"
echo "  • Matlab files for Task 1 (SR - Sentence Reading)"
echo "  • Matlab files for Task 2 (NR - Normal Reading)"  
echo "  • Matlab files for Task 3 (TSR - Task-Specific Reading)"
echo ""
echo "Alternative OSF links:"
echo "  • ZuCo 1.0: https://osf.io/q3zws/"
echo "  • ZuCo 2.0: https://osf.io/2urht/"
echo ""
echo "Total download size: ~10-15 GB"
echo "Download time: 20-60 minutes (depends on your connection)"
echo ""
echo -e "${YELLOW}STEP 2: Save to Project Directory${NC}"
echo "-------------------------------------------------------------------------------"
echo ""
echo "Save downloaded files to:"
echo -e "${GREEN}$(pwd)/data/raw/zuco/${NC}"
echo ""
echo "Create the directory if it doesn't exist:"
echo "  mkdir -p $(pwd)/data/raw/zuco"
echo ""
echo -e "${YELLOW}STEP 3: Extract Files${NC}"
echo "-------------------------------------------------------------------------------"
echo ""
echo "If you downloaded zip files, extract them:"
echo "  cd data/raw/zuco"
echo "  unzip '*.zip'"
echo ""
echo "Or use this script with --extract flag:"
echo "  $0 --extract"
echo ""
echo -e "${YELLOW}STEP 4: Verify Download${NC}"
echo "-------------------------------------------------------------------------------"
echo ""
echo "After download and extraction, verify the data:"
echo "  $0 --verify"
echo ""
echo "Or run Python verification:"
echo "  python scripts/verify_zuco_data.py"
echo ""
echo "================================================================================"
echo ""

# Handle flags
if [[ "$1" == "--extract" ]]; then
    echo "Extracting ZuCo files..."
    cd data/raw/zuco || exit 1
    
    if ls *.zip 1> /dev/null 2>&1; then
        for zipfile in *.zip; do
            echo "  Extracting $zipfile..."
            unzip -q "$zipfile"
            echo "  ✓ $zipfile extracted"
        done
        echo ""
        echo "✓ All files extracted!"
    else
        echo "✗ No .zip files found in data/raw/zuco/"
        echo "  Please download files from https://osf.io/q3zws/ first"
        exit 1
    fi
    
elif [[ "$1" == "--verify" ]]; then
    echo "Verifying ZuCo dataset structure..."
    cd data/raw/zuco || exit 1
    
    total_size=$(du -sh . | cut -f1)
    echo "  Total size: $total_size"
    
    mat_files=$(find . -name "*.mat" 2>/dev/null | wc -l | tr -d ' ')
    echo "  MATLAB files found: $mat_files"
    
    if [[ $mat_files -gt 0 ]]; then
        echo ""
        echo "✓ ZuCo dataset files detected!"
        echo ""
        echo "Next step: Run preprocessing"
        echo "  python scripts/train_zuco_full.py --preprocess-only"
    else
        echo ""
        echo "✗ No .mat files found"
        echo "  Please download from: https://osf.io/q3zws/"
        echo ""
        echo "Expected structure:"
        echo "  data/raw/zuco/"
        echo "    ├── task1_SR/ (or similar)"
        echo "    ├── task2_NR/"
        echo "    └── task3_TSR/"
    fi
    
else
    echo "Usage:"
    echo "  $0              Show download instructions"
    echo "  $0 --extract    Extract downloaded zip files"
    echo "  $0 --verify     Verify dataset structure"
    echo ""
    echo "Quick workflow:"
    echo "  1. Visit https://osf.io/q3zws/ and download files"
    echo "  2. Save to data/raw/zuco/"
    echo "  3. Run: $0 --extract"
    echo "  4. Run: $0 --verify"
    echo ""
fi
