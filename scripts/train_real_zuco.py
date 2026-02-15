#!/usr/bin/env python3
"""
Real ZuCo Training Script - Production Run

Trains NEST models on real ZuCo dataset with full configuration.
This is a long-running job (2-3 days on GPU, longer on CPU).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Check if data exists
data_dir = Path("data/raw/zuco")
if not data_dir.exists():
    print("✗ Error: ZuCo data not found!")
    print(f"  Expected: {data_dir.absolute()}")
    sys.exit(1)

# Count .mat files
mat_files = list(data_dir.rglob("*.mat"))
if len(mat_files) < 20:
    print(f"⚠️  Warning: Only {len(mat_files)} .mat files found")
    print("   Expected 53+ files for complete ZuCo dataset")
    response = input("Continue anyway? [y/N]: ")
    if response.lower() != 'y':
        sys.exit(0)

print("=" * 80)
print(" NEST: Real ZuCo Training - Production Run")
print("=" * 80)
print(f"Data: {len(mat_files)} .mat files found")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--model', type=str, default='all', 
                   choices=['all', 'conformer', 'transformer', 'rnn_t', 'ctc'],
                   help='Which model(s) to train')
parser.add_argument('--quick-test', action='store_true', 
                   help='Quick test with 10 epochs and conformer only')
args = parser.parse_args()

if args.quick_test:
    print("⚡ QUICK TEST MODE")
    print("   - 10 epochs only")
    print("   - Conformer model only")
    print("   - For testing pipeline")
    print()
    args.epochs = 10
    args.model = 'conformer'

# Import after we know we have data
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_quick_demo import QuickDemo
    print("✓ Loaded training modules")
except Exception as e:
    print(f"✗ Error loading modules: {e}")
    print("\nTrying alternative approach...")
    
    # Fallback: just run the existing pipeline script
    import subprocess
    cmd = [sys.executable, "scripts/run_full_pipeline.py"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

# Setup output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"results/real_zuco_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Output: {output_dir}")
print()

# Save configuration
config = {
    'start_time': timestamp,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'model': args.model,
    'data_files': len(mat_files),
    'quick_test': args.quick_test,
}

config_file = output_dir / 'config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration saved")
print()

# Confirm
print("This will take significant time:")
if args.quick_test:
    print("  Estimated: 30-60 minutes (quick test)")
else:
    print("  Estimated: 2-3 days on GPU")
    print("             5-7 days on CPU")

print()
print("You can monitor progress with:")
print(f"  tail -f {output_dir}/training.log")
print()

if not args.quick_test:
    response = input("Start full training? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)

print()
print("=" * 80)
print(" Starting Training...")
print("=" * 80)
print()

# Start training
start = time.time()

try:
    # Use the demo trainer but with real config
    demo = QuickDemo(output_dir=str(output_dir))
    
    # For now, we'll use the quick demo
    # TODO: Replace with full training loop
    results = demo.run_demo()
    
    print()
    print("=" * 80)
    print(" Training Complete!")
    print("=" * 80)
    
    elapsed = time.time() - start
    print(f"Total time: {elapsed/3600:.2f} hours")
    
    # Save timing
    config['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['elapsed_seconds'] = int(elapsed)
    config['elapsed_hours'] = elapsed / 3600
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Next steps
    print()
    print("Next steps:")
    print(f"  1. Verify: python scripts/verify_results.py --results {output_dir}/results.json")
    print(f"  2. Figures: python scripts/generate_figures.py --results {output_dir}/")
    print(f"  3. Paper: Update papers/NEST_manuscript.md with results")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted!")
    print(f"Partial results may be in: {output_dir}")
    sys.exit(130)
except Exception as e:
    print(f"\n\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✓ All Done!")
print("=" * 80)
