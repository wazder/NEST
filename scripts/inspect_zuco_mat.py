#!/usr/bin/env python3
"""Inspect ZuCo .mat file structure to understand the data format."""

import scipy.io
import numpy as np
from pathlib import Path
import sys

# Find first .mat file
data_dir = Path("data/raw/zuco")
if not data_dir.exists():
    data_dir = Path("ZuCo_Dataset/ZuCo")

mat_files = []
for task in ["task1_SR", "task2_NR", "task3_TSR"]:
    task_dir = data_dir / task / "Matlab_files"
    if task_dir.exists():
        mat_files.extend(list(task_dir.glob("*.mat")))

if not mat_files:
    print("No .mat files found!")
    sys.exit(1)

mat_file = mat_files[0]
print(f"Inspecting: {mat_file.name}")
print("=" * 80)

# Load .mat file
print("\nLoading .mat file...")
mat_data = scipy.io.loadmat(str(mat_file), simplify_cells=True)

print("\nTop-level keys:")
for key in mat_data.keys():
    if not key.startswith('__'):
        value = mat_data[key]
        print(f"  {key}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"    Shape: {value.shape}, dtype: {value.dtype}")

# Inspect sentenceData if it exists
if 'sentenceData' in mat_data:
    print("\n" + "=" * 80)
    print("sentenceData structure:")
    print("=" * 80)
    
    sent_data = mat_data['sentenceData']
    print(f"Type: {type(sent_data)}")
    
    if isinstance(sent_data, dict):
        print("sentenceData is a single dict")
        print("Keys:", list(sent_data.keys()))
        for key, val in sent_data.items():
            print(f"  {key}: {type(val)}")
            if isinstance(val, np.ndarray):
                print(f"    Shape: {val.shape}, dtype: {val.dtype}")
    elif isinstance(sent_data, (list, np.ndarray)):
        print(f"sentenceData is array/list with {len(sent_data)} items")
        if len(sent_data) > 0:
            first = sent_data[0]
            print(f"\nFirst item type: {type(first)}")
            if isinstance(first, dict):
                print("First item keys:", list(first.keys()))
                for key, val in first.items():
                    print(f"  {key}: {type(val)}")
                    if isinstance(val, np.ndarray):
                        print(f"    Shape: {val.shape}, dtype: {val.dtype}")
                    elif isinstance(val, str):
                        print(f"    Value: {val[:100]}...")

print("\n" + "=" * 80)

# Try to extract a sample EEG + text pair
print("\nAttempting to extract first sample...")
try:
    if 'sentenceData' in mat_data:
        sent_data = mat_data['sentenceData']
        
        if isinstance(sent_data, dict):
            items = [sent_data]
        else:
            items = sent_data
        
        if len(items) > 0:
            sample = items[0]
            print(f"\nSample type: {type(sample)}")
            
            if isinstance(sample, dict):
                print("\nAll fields in sample:")
                for key in sorted(sample.keys()):
                    val = sample[key]
                    if isinstance(val, np.ndarray):
                        print(f"  {key}: array shape {val.shape}")
                    elif isinstance(val, (list, tuple)):
                        print(f"  {key}: {type(val).__name__} length {len(val)}")
                    else:
                        print(f"  {key}: {type(val).__name__}")
                        if isinstance(val, str) and len(val) < 200:
                            print(f"      = '{val}'")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Inspection complete!")
