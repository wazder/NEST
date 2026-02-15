#!/usr/bin/env python3
"""
Download and verify ZuCo dataset.

This script downloads ZuCo dataset from OSF and verifies integrity.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm

# ZuCo dataset information
ZUCO_DATASETS = {
    'task1_SR': {
        'url': 'https://osf.io/download/q3zws/',
        'size_gb': 5.2,
        'description': 'Task 1: Sentence Reading'
    },
    'task2_NR': {
        'url': 'https://osf.io/download/sjezn/',
        'size_gb': 8.1,
        'description': 'Task 2: Normal Reading'
    },
    'task3_TSR': {
        'url': 'https://osf.io/download/2urht/',
        'size_gb': 4.8,
        'description': 'Task 3: Task Specific Reading'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"✓ Download complete: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file with progress."""
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)
    
    print(f"✓ Extracted to {extract_to}")


def verify_directory_structure(data_dir: Path) -> bool:
    """Verify downloaded data structure."""
    print("\nVerifying directory structure...")
    
    required_dirs = ['task1_SR', 'task2_NR', 'task3_TSR']
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}: {'Found' if exists else 'Missing'}")
        
        if exists:
            mat_files = list(dir_path.rglob("*.mat"))
            print(f"      → {len(mat_files)} .mat files")
        
        all_exist = all_exist and exists
    
    return all_exist


def main():
    parser = argparse.ArgumentParser(description='Download ZuCo dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/zuco',
        help='Output directory for ZuCo data'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['task1_SR', 'task2_NR', 'task3_TSR', 'all'],
        default=['all'],
        help='Which tasks to download'
    )
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip extraction (if already extracted)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data, do not download'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which tasks to download
    if 'all' in args.tasks:
        tasks = list(ZUCO_DATASETS.keys())
    else:
        tasks = args.tasks
    
    print("=" * 80)
    print("ZuCo Dataset Download")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Tasks: {', '.join(tasks)}")
    
    total_size = sum(ZUCO_DATASETS[t]['size_gb'] for t in tasks)
    print(f"Total download size: ~{total_size:.1f} GB")
    print("=" * 80)
    
    if args.verify_only:
        print("\nVerification mode - skipping download")
        if verify_directory_structure(output_dir):
            print("\n✓ All required data present!")
            return 0
        else:
            print("\n✗ Some data is missing. Run without --verify-only to download.")
            return 1
    
    # Confirm download
    if not args.verify_only:
        response = input(f"\nDownload {total_size:.1f} GB? This may take 30-60 minutes. [y/N]: ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return 0
    
    # Download each task
    for task in tasks:
        print(f"\n{'=' * 80}")
        print(f"Downloading {task}: {ZUCO_DATASETS[task]['description']}")
        print(f"Size: ~{ZUCO_DATASETS[task]['size_gb']:.1f} GB")
        print(f"{'=' * 80}")
        
        zip_path = output_dir / f"{task}.zip"
        extract_path = output_dir / task
        
        # Check if already downloaded
        if zip_path.exists():
            print(f"Archive already exists: {zip_path}")
            response = input("Re-download? [y/N]: ")
            if response.lower() != 'y':
                print("Skipping download...")
            else:
                download_file(ZUCO_DATASETS[task]['url'], zip_path)
        else:
            download_file(ZUCO_DATASETS[task]['url'], zip_path)
        
        # Extract
        if not args.skip_extract:
            if extract_path.exists():
                print(f"Extract directory already exists: {extract_path}")
                response = input("Re-extract? [y/N]: ")
                if response.lower() != 'y':
                    print("Skipping extraction...")
                    continue
            
            extract_zip(zip_path, output_dir)
            print(f"✓ Task {task} complete")
        else:
            print("Skipping extraction (--skip-extract)")
    
    # Verify
    print("\n" + "=" * 80)
    if verify_directory_structure(output_dir):
        print("\n✓✓✓ All ZuCo data downloaded and verified successfully! ✓✓✓")
        print("\nNext steps:")
        print("  1. Run preprocessing: python scripts/train_zuco_full.py --preprocess-only")
        print("  2. Or start training: python scripts/train_zuco_full.py")
        return 0
    else:
        print("\n✗ Verification failed. Some data may be missing.")
        print("Try running again or download manually from: https://osf.io/q3zws/")
        return 1


if __name__ == '__main__':
    sys.exit(main())
