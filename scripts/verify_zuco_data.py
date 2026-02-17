#!/usr/bin/env python3
"""
Verify ZuCo dataset structure and integrity.

Checks if downloaded ZuCo data is complete and properly structured.
"""

import sys
from pathlib import Path
import scipy.io as sio
from typing import List, Dict


class ZuCoVerifier:
    """Verify ZuCo dataset."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.issues = []
        self.warnings = []
        
    def check_directory_exists(self) -> bool:
        """Check if data directory exists."""
        if not self.data_dir.exists():
            self.issues.append(f"Data directory not found: {self.data_dir}")
            return False
        return True
    
    def find_mat_files(self) -> List[Path]:
        """Find all .mat files in directory."""
        return list(self.data_dir.rglob("*.mat"))
    
    def verify_mat_file(self, mat_file: Path) -> Dict:
        """Verify a single .mat file."""
        result = {
            'path': mat_file,
            'valid': False,
            'size_mb': 0,
            'error': None
        }
        
        try:
            # Get file size
            size_bytes = mat_file.stat().st_size
            result['size_mb'] = size_bytes / (1024 * 1024)
            
            # Try to load the file
            data = sio.loadmat(mat_file)
            
            # Check if it has expected keys (ZuCo structure varies)
            keys = list(data.keys())
            # Filter out MATLAB metadata keys
            data_keys = [k for k in keys if not k.startswith('__')]
            
            if len(data_keys) > 0:
                result['valid'] = True
                result['keys'] = data_keys
            else:
                result['error'] = "No data keys found"
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def verify_all(self) -> bool:
        """Run complete verification."""
        print("=" * 80)
        print("ZuCo Dataset Verification")
        print("=" * 80)
        print(f"Data directory: {self.data_dir.absolute()}")
        print("")
        
        # Check directory
        if not self.check_directory_exists():
            print("✗ Data directory not found!")
            print(f"  Expected: {self.data_dir.absolute()}")
            print("")
            print("Please download ZuCo dataset from: https://osf.io/q3zws/")
            print("See: scripts/download_zuco_manual.sh for instructions")
            return False
        
        # Find .mat files
        print("Searching for MATLAB files...")
        mat_files = self.find_mat_files()
        
        if not mat_files:
            print("✗ No .mat files found!")
            print("")
            print("Expected ZuCo dataset structure:")
            print("  data/raw/zuco/")
            print("    ├── task1_SR/ (or resultsSR/)")
            print("    │   ├── results*.mat")
            print("    ├── task2_NR/")
            print("    │   ├── results*.mat")
            print("    └── task3_TSR/")
            print("        ├── results*.mat")
            print("")
            print("Please download from: https://osf.io/q3zws/")
            return False
        
        print(f"Found {len(mat_files)} .mat files")
        print("")
        
        # Verify each file
        print("Verifying files...")
        print("-" * 80)
        
        valid_count = 0
        total_size = 0
        
        for mat_file in mat_files:
            result = self.verify_mat_file(mat_file)
            rel_path = mat_file.relative_to(self.data_dir)
            
            if result['valid']:
                status = "✓"
                valid_count += 1
                total_size += result['size_mb']
                print(f"{status} {rel_path} ({result['size_mb']:.1f} MB)")
            else:
                status = "✗"
                print(f"{status} {rel_path} - ERROR: {result['error']}")
                self.issues.append(f"Invalid file: {rel_path}")
        
        print("-" * 80)
        print("")
        
        # Summary
        print("Summary:")
        print(f"  Total files: {len(mat_files)}")
        print(f"  Valid files: {valid_count}")
        print(f"  Invalid files: {len(mat_files) - valid_count}")
        print(f"  Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        print("")
        
        # Check expected counts
        # ZuCo typically has 12 subjects x 3 tasks = 36 files minimum
        if valid_count < 10:
            self.warnings.append(
                f"Only {valid_count} files found. ZuCo typically has 30+ files. "
                "Dataset may be incomplete."
            )
        
        # Report issues
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
            print("")
        
        if self.issues:
            print("Issues:")
            for issue in self.issues:
                print(f"  ✗ {issue}")
            print("")
            return False
        
        # Success
        if valid_count > 0:
            print("=" * 80)
            print("✓ ZuCo dataset verified successfully!")
            print("=" * 80)
            print("")
            print("Next steps:")
            print("  1. Preprocess data:")
            print("     python scripts/train_zuco_full.py --preprocess-only")
            print("")
            print("  2. Or start full training:")
            print("     python scripts/train_zuco_full.py")
            print("")
            return True
        
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify ZuCo dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/zuco',
        help='Path to ZuCo data directory'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    verifier = ZuCoVerifier(data_dir)
    
    success = verifier.verify_all()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
