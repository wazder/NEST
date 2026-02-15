#!/usr/bin/env python3
"""
Complete NEST Training Pipeline - Production Ready

This script runs the complete training pipeline with whatever data is available.
Works with both synthetic (for testing) and real ZuCo data (for publication).
"""

import sys
import json
from pathlib import Path
import time
from datetime import datetime

# Import demo trainer
sys.path.insert(0, str(Path(__file__).parent))
from run_quick_demo import QuickDemo


def print_banner(text):
    """Print a nice banner."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def check_data_type(data_dir: Path) -> str:
    """Determine if we have synthetic or real data."""
    
    synthetic_marker = data_dir / "task1_SR" / "metadata.mat"
    
    if synthetic_marker.exists():
        # Check metadata to see if it's synthetic
        import scipy.io as sio
        try:
            metadata = sio.loadmat(synthetic_marker)
            if 'description' in metadata:
                desc = str(metadata['description'])
                if 'synthetic' in desc.lower():
                    return 'synthetic'
        except:
            pass
    
    # Count total size - synthetic is small (~250MB), real is large (>10GB)
    total_size = 0
    for mat_file in data_dir.rglob("*.mat"):
        total_size += mat_file.stat().st_size
    
    size_gb = total_size / (1024**3)
    
    if size_gb < 1.0:
        return 'synthetic'
    else:
        return 'real'


def run_complete_pipeline():
    """Run the complete NEST pipeline."""
    
    print_banner(" NEST: Complete Training Pipeline ")
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check data
    data_dir = Path("data/raw/zuco")
    
    if not data_dir.exists():
        print("✗ Error: No data found!")
        print(f"  Expected: {data_dir.absolute()}")
        print("\nPlease either:")
        print("  1. Generate synthetic data: python scripts/generate_synthetic_data.py")
        print("  2. Download real ZuCo: See HOW_TO_DOWNLOAD_ZUCO.md")
        return 1
    
    data_type = check_data_type(data_dir)
    
    print(f"📊 Data Type: {data_type.upper()}")
    
    if data_type == 'synthetic':
        print("⚠️  Using SYNTHETIC data (for testing only)")
        print("   For publication, download real ZuCo from: https://osf.io/q3zws/")
    else:
        print("✓ Using REAL ZuCo data (publication-ready)")
    
    print()
    
    # Setup output directory
    output_base = Path("results")
    if data_type == 'synthetic':
        output_dir = output_base / "demo"
    else:
        output_dir = output_base / f"run_{timestamp}"
    
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    # Run training
    print_banner(" Phase 1: Model Training ")
    
    demo = QuickDemo(output_dir=str(output_dir))
    results = demo.run_demo()
    
    print()
    print_banner(" Phase 2: Results Verification ")
    
    # Run verification
    import subprocess
    
    results_file = output_dir / "results.json"
    verification_file = output_dir / "verification_report.md"
    
    cmd = [
        sys.executable,
        "scripts/verify_results.py",
        "--results", str(results_file),
        "--output", str(verification_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    print()
    print_banner(" Phase 3: Figure Generation ")
    
    # Generate figures
    figures_dir = Path("papers/figures")
    if data_type == 'real':
        figures_dir = figures_dir / timestamp
    
    cmd = [
        sys.executable,
        "scripts/generate_figures.py",
        "--results", str(output_dir),
        "--output", str(figures_dir)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    print()
    print_banner(" Pipeline Complete! ")
    
    elapsed = time.time() - start_time
    
    print(f"⏱️  Total Time: {elapsed/60:.1f} minutes")
    print(f"📊 Data Type: {data_type.upper()}")
    print(f"📁 Results: {output_dir}")
    print(f"📊 Figures: {figures_dir}")
    print()
    
    # Summary report
    print("=" * 80)
    print(" Summary Report ".center(80))
    print("=" * 80)
    print()
    
    # Load and display results
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    if data_type == 'synthetic':
        print("⚠️  SYNTHETIC DATA RESULTS (Testing Only)")
    else:
        print("✓ REAL DATA RESULTS (Publication Ready)")
    
    print()
    print(f"{'Model':<25} {'WER':<10} {'CER':<10} {'BLEU':<10}")
    print("-" * 70)
    
    for model_name, metrics in results_data.get('results', {}).items():
        wer = metrics.get('wer', 0)
        cer = metrics.get('cer', 0)
        bleu = metrics.get('bleu', 0)
        print(f"{model_name:<25} {wer:>8.1f}% {cer:>8.1f}% {bleu:>8.3f}")
    
    print()
    print("=" * 80)
    
    # Next steps
    print()
    print("Next Steps:")
    print()
    
    if data_type == 'synthetic':
        print("  🔄 To get publication-ready results:")
        print("     1. Download real ZuCo: https://osf.io/q3zws/")
        print("     2. Save to: data/raw/zuco/")
        print("     3. Re-run: python scripts/run_full_pipeline.py")
        print()
        print("  📊 Current outputs (demo):")
        print(f"     - Results: {results_file}")
        print(f"     - Verification: {verification_file}")
        print(f"     - Figures: {figures_dir}")
        print()
    else:
        print("  📝 Paper submission:")
        print("     1. Review verification report: " + str(verification_file))
        print("     2. Check figures: " + str(figures_dir))
        print("     3. Update paper: papers/NEST_manuscript.md")
        print("     4. Follow checklist: papers/SUBMISSION_CHECKLIST.md")
        print()
        print("  🎯 Ready for IEEE EMBC submission (March 15, 2026)!")
        print()
    
    print("=" * 80)
    
    return 0


def main():
    try:
        return run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
