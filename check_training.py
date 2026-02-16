#!/usr/bin/env python3
"""Quick progress checker for training."""
import subprocess
import time

print("\n🔍 Training Status Check\n")
print("=" * 60)

# Check if process is running
result = subprocess.run(
    "ps aux | grep 'train_with_real_zuco' | grep -v grep | wc -l",
    shell=True,
    capture_output=True,
    text=True
)

if int(result.stdout.strip()) > 0:
    print("✅ Training process is RUNNING")
    
    # Get process details
    result = subprocess.run(
        "ps aux | grep 'train_with_real_zuco' | grep -v grep | awk '{print $3,$4,$10}'",
        shell=True,
        capture_output=True,
        text=True
    )
    cpu, mem, time_str = result.stdout.strip().split()
    print(f"   CPU: {cpu}%")
    print(f"   Memory: {mem}%")
    print(f"   Runtime: {time_str}")
    
    print("\n💡 If CPU is low (%0-5):")
    print("   - Model is being loaded to GPU (first batch)")
    print("   - This is NORMAL and takes 1-2 minutes")
    print("   - GPU usage will spike to 80-95% soon!")
    
    print("\n📊 Expected timeline:")
    print("   Minutes 0-2:  GPU %0-20  (initialization)")
    print("   Minutes 2-4:  GPU %80-95 (training starts!)")
    print("   Then: Each epoch ~3-4 minutes")
    
else:
    print("❌ Training process not found")
    print("   Check the terminal where you ran ./start_full_training.sh")

print("\n" + "=" * 60)
print("\n⏱️  Give it 2-3 more minutes, then check GPU again!")
print("   Activity Monitor → Window → GPU History\n")
