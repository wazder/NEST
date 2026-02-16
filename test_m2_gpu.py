#!/usr/bin/env python3
"""
Quick test to check if M2 GPU (MPS) is available and working.
"""
import torch

print("=" * 70)
print("M2 Air GPU Test")
print("=" * 70)
print()

# Check MPS availability
print("Checking M2 GPU (MPS)...")
if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is AVAILABLE")
    print("   Your M2 Air GPU can be used for training!")
    
    # Test MPS
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✅ MPS test computation successful")
        print()
        print("🚀 Training will use M2 GPU acceleration")
        print("   Expected time: 4-8 hours (instead of 2-3 days on CPU)")
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        print("   Will fall back to CPU")
else:
    print("❌ MPS not available")
    print("   Training will use CPU (slower)")

print()

# Check CUDA (unlikely on Mac)
if torch.cuda.is_available():
    print("✅ CUDA GPU available:", torch.cuda.get_device_name(0))
else:
    print("ℹ️  CUDA not available (expected on Mac)")

print()

# System info
print("PyTorch version:", torch.__version__)
print("Python version:", torch.version.__version__ if hasattr(torch.version, '__version__') else 'N/A')

print()
print("=" * 70)
