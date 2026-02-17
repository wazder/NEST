# 🟠 SCRIPT ISSUES

Issues found in Python and Shell scripts in the `/scripts` directory.

## 1. Task Directory Naming Inconsistency 🔴

**Severity**: CRITICAL  
**Impact**: Runtime FileNotFoundError  
(Already covered in 01-CRITICAL-ISSUES.md)

### Complete List of Affected Files

#### Scripts Using Underscores (`task1_SR`)
1. `scripts/download_zuco.py` - Line ~74
2. `scripts/generate_synthetic_data.py` - Line ~30
3. `scripts/preprocessing/split_data.py`
4. `configs/preprocessing.yaml` - Line 10

#### Scripts Using Hyphens (`task1-SR`)
1. `scripts/train_with_real_zuco.py` - Lines 42-45
2. `scripts/inspect_zuco_mat.py` - Line ~30
3. `scripts/download_zuco_manual.sh` - Line ~20
4. `scripts/download_zuco_simple.sh` - Mixed format

### Example of the Problem

**Download script** creates:
```python
# download_zuco.py
task_dirs = ['task1_SR', 'task2_NR', 'task3_TSR']
for task in task_dirs:
    os.makedirs(f'data/raw/zuco/{task}')
```

**Training script** expects:
```python
# train_with_real_zuco.py
task_dirs = ['task1-SR', 'task2-NR', 'task3-TSR']
for task in task_dirs:
    data_path = f'data/raw/zuco/{task}'  # ← Won't find it!
```

**Result**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/zuco/task1-SR'`

### Fix Required
Choose ONE format and update all scripts. Recommend underscores:
- Consistent with Python naming conventions
- Matches config file format
- No shell escaping issues

---

## 2. Hardcoded Paths 🟡

**Severity**: MEDIUM (except user-specific path which is CRITICAL)

### 2.1 Data Directory Paths

| Script | Line | Hardcoded Path |
|--------|------|----------------|
| `RUN_EVERYTHING.sh` | 39, 56 | `data/raw/zuco/` |
| `RUN_EVERYTHING.sh` | 96-99 | `results/demo/` |
| `quickstart.sh` | 17, 75, 123 | `results/quickstart`, `data/processed/zuco/train` |
| `run_full_pipeline.py` | 66, 89 | `data/raw/zuco`, `results` |
| `generate_figures.py` | 359, 365 | `results`, `papers/figures` |
| `train_real_zuco.py` | 17, 78 | `data/raw/zuco`, `results/real_zuco_*` |
| `train_with_real_zuco.py` | 30-38 | Multiple path attempts |
| `inspect_zuco_mat.py` | 10, 12 | `ZuCo_Dataset/ZuCo` |

### Example Issue
```python
# train_with_real_zuco.py, line 30-38
ZUCO_DATA_DIR = "data/raw/zuco"  # Hardcoded
if not os.path.exists(ZUCO_DATA_DIR):
    ZUCO_DATA_DIR = "/data/zuco"  # Alternate hardcoded
if not os.path.exists(ZUCO_DATA_DIR):
    ZUCO_DATA_DIR = "/mnt/data/zuco"  # Another hardcoded
```

### Problem
- Not configurable without editing source
- Breaks in different environments
- Docker containers need special setup
- CI/CD needs workarounds

### Recommendation
Use environment variables with fallbacks:
```python
import os
from pathlib import Path

# Get from env or use default
ZUCO_DATA_DIR = os.getenv('ZUCO_DATA_DIR', 'data/raw/zuco')
RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')

# Or use a config file
from omegaconf import OmegaConf
config = OmegaConf.load('config/paths.yaml')
ZUCO_DATA_DIR = config.paths.data
```

---

### 2.2 Virtual Environment Paths 🟡

**File**: `start_full_training.sh`  
**Line**: 34  
**Severity**: MEDIUM

```bash
source .venv/bin/activate  # Hardcoded venv path
```

**Problem**:
- Assumes venv is in `.venv`
- No fallback if venv elsewhere
- Fails silently if venv doesn't exist

**Fix**:
```bash
# Try multiple common venv locations
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    # Already in a venv
    :
else
    echo "Error: No virtual environment found"
    exit 1
fi
```

---

### 2.3 User-Specific Path 🔴

**File**: `docs/guides/RUN_ME_FIRST.md`  
**Line**: 8  
**Severity**: CRITICAL

```bash
cd /Users/wazder/Documents/GitHub/NEST  # ← Will fail for everyone else
```

Already covered in critical issues.

---

## 3. Dataset Size Inconsistencies 🟠

**Severity**: HIGH  
**Impact**: User confusion

### Contradictory Size Claims

| Script | Line | Size Claimed |
|--------|------|--------------|
| `start_full_training.sh` | 13 | 66 GB |
| `download_zuco_simple.sh` | 29 | ~18 GB |
| `download_zuco_manual.sh` | 36 | 10-15 GB |
| `docs/TRAINING_GUIDE.md` | 49 | ~5 GB |

### Examples

**start_full_training.sh** (line 13):
```bash
echo "  - Dataset: Real ZuCo (66 GB, ~20,000 samples)"
```

**download_zuco_simple.sh** (line 29):
```bash
echo "Total download size: ~18 GB"
```

**download_zuco_manual.sh** (line 36):
```bash
echo "Expected download size: 10-15 GB"
```

### Problem
Users don't know how much space to allocate.

### Possible Explanation
Different stages:
- **Compressed downloads**: 10-15 GB
- **Extracted raw data**: ~18 GB
- **After preprocessing**: 5-10 GB
- **With all tasks + augmentation**: 50-70 GB

### Fix
Document clearly in each script:
```bash
# Download size (compressed): ~15 GB
# Extracted size (raw): ~18 GB
# Processed size: ~5-10 GB
# Total space needed: ~50 GB (with working space)
```

---

## 4. Missing Error Handling 🟡

**Severity**: MEDIUM

### 4.1 Unconditional Script Execution

**File**: `start_full_training.sh`  
**Lines**: 34, 37  

```bash
source .venv/bin/activate  # No check if exists

python scripts/train_with_real_zuco.py \
    --quick-test  # No check if script exists
```

**Problem**: Fails with cryptic error if files don't exist.

**Fix**:
```bash
if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at .venv/"
    exit 1
fi

source .venv/bin/activate

if [ ! -f "scripts/train_with_real_zuco.py" ]; then
    echo "Error: Training script not found"
    exit 1
fi
```

---

### 4.2 No File Existence Check Before Read

**File**: `quickstart.sh`  
**Line**: 111  

```bash
cat $OUTPUT_DIR/RESULTS_REPORT.md  # No check if exists
```

**Fix**:
```bash
if [ -f "$OUTPUT_DIR/RESULTS_REPORT.md" ]; then
    cat "$OUTPUT_DIR/RESULTS_REPORT.md"
else
    echo "Warning: Results report not found"
fi
```

---

### 4.3 No Error Handling for Corrupted Files

**File**: `inspect_zuco_mat.py`  
**Line**: 30  

```python
data = loadmat(mat_file)  # No try-except for corrupted files
```

**Fix**:
```python
try:
    data = loadmat(mat_file)
except Exception as e:
    print(f"Error loading {mat_file}: {e}")
    continue
```

---

### 4.4 Silent Fallback on Import Errors

**File**: `train_real_zuco.py`  
**Lines**: 62-74  

```python
try:
    from src.models import create_model
except ImportError:
    # Falls back silently without logging
    pass
```

**Problem**: Hides import errors that should be visible.

**Fix**:
```python
try:
    from src.models import create_model
except ImportError as e:
    logger.error(f"Failed to import model: {e}")
    raise
```

---

### 4.5 Subprocess Execution Without Error Check

**File**: `run_full_pipeline.py`  
**Lines**: 120-121  

```python
result = subprocess.run([...])
# Continues without checking result.returncode
```

**Fix**:
```python
result = subprocess.run([...], check=False)
if result.returncode != 0:
    logger.error(f"Command failed with code {result.returncode}")
    sys.exit(1)
```

---

## 5. Logical Errors & Incomplete Implementations 🟡

### 5.1 Redundant File Operations

**File**: `download_zuco.py`  
**Lines**: 149-150  

```python
torch.save(processed_data, save_path)  # Line 149: Save
data = torch.load(save_path)           # Line 150: Immediately load?
```

**Problem**: Why save then immediately load from same path?

**Possible reasons**:
1. Verification check (OK if intentional)
2. Copy-paste error (should be removed)

**Recommendation**: If verification, add comment:
```python
torch.save(processed_data, save_path)
# Verify save was successful
data = torch.load(save_path)
assert data is not None
```

---

### 5.2 Missing Script References

**File**: `quickstart.sh`  
**Lines**: 98, 132  

```bash
python scripts/evaluate_models.py     # Line 98: Doesn't exist
python scripts/optimize_for_deployment.py  # Line 132: Doesn't exist
```

**Problem**: References scripts that don't exist in repository.

**Impact**: Script fails when reaching these lines.

**Fix**:
1. Create the missing scripts
2. Or remove/comment out the references
3. Or add existence checks:
```bash
if [ -f "scripts/evaluate_models.py" ]; then
    python scripts/evaluate_models.py
else
    echo "Skipping evaluation (script not found)"
fi
```

---

### 5.3 Fragile Path Handling

**File**: `RUN_EVERYTHING.sh`  
**Lines**: 34-44  

```bash
python << END
import sys
sys.path.insert(0, 'scripts')  # ← Fragile path manipulation
# ... check data type
END
```

**Problem**: 
- Relies on current working directory
- Breaks if script run from different location
- Better to use absolute paths or proper imports

**Fix**:
```bash
python << END
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))
END
```

---

### 5.4 TODO Comments Indicating Incomplete Code

**File**: `train_real_zuco.py`  
**Line**: 134  

```python
# TODO: Add validation logic here
pass
```

**Problem**: Incomplete implementation left in production code.

**Impact**: Validation not actually performed.

**Recommendation**:
1. Implement the validation logic
2. Or remove TODO if not needed
3. Or create GitHub issue and reference it:
```python
# TODO(#123): Add validation logic here
```

---

### 5.5 In-Place Dictionary Modification

**File**: `verify_results.py`  
**Lines**: 207-209  

```python
TOLERANCES = {'wer': 0.02, 'bleu': 0.01}

if args.strict:
    TOLERANCES['wer'] = 0.01  # Modifies global dict
```

**Problem**: If module is imported multiple times or in REPL, modifications persist.

**Fix**:
```python
def get_tolerances(strict=False):
    tolerances = {'wer': 0.02, 'bleu': 0.01}
    if strict:
        tolerances['wer'] = 0.01
    return tolerances

tolerances = get_tolerances(args.strict)
```

---

## 6. Inconsistent Script Naming 🟢

**Severity**: LOW

### Current Naming Patterns

| Pattern | Examples | Count |
|---------|----------|-------|
| **snake_case.py** | `download_zuco.py`, `generate_figures.py` | ~15 |
| **kebab-case.sh** | `start-pipeline.sh` | ~3 |
| **snake_case.sh** | `start_full_training.sh`, `RUN_EVERYTHING.sh` | ~5 |
| **Mixed case** | `RUN_EVERYTHING.sh` | 1 |

### Issue
No clear naming convention for scripts.

### Recommendation
Standardize:
- **Python scripts**: `snake_case.py`
- **Shell scripts**: `snake_case.sh`
- **Main entry points**: Prefix with `run_` (e.g., `run_training.sh`)
- **Utilities**: Place in `scripts/utils/` subdirectory

---

## 7. Future Date in Demo Message 🟢

**File**: `RUN_EVERYTHING.sh`  
**Line**: 109  
**Severity**: LOW

```bash
echo "Demo deadline: March 15, 2026"
```

**Issue**: Hardcoded future date.

**Recommendation**: Either remove deadline or make it configurable.

---

## Summary

### Issues by Severity

#### 🔴 Critical: 2
1. Task naming inconsistency (causes FileNotFoundError)
2. Hardcoded user-specific path

#### 🟠 High: 3
1. Dataset size inconsistencies (66GB vs 5GB claims)
2. Multiple hardcoded paths without configuration
3. Missing error handling (multiple instances)

#### 🟡 Medium: 8
1. Virtual environment path assumptions
2. Redundant file operations
3. Missing script references
4. Fragile path handling
5. TODO comments with incomplete code
6. In-place dictionary modifications
7. No subprocess return code checks
8. Silent import error fallbacks

#### 🟢 Low: 3
1. Inconsistent script naming conventions
2. Future date in demo message
3. Minor shell script improvements

### Total Script Issues: 16+

### Scripts Examined
- ✅ 20+ scripts reviewed
- ⚠️ 10+ scripts with issues
- 🔴 2 critical issues found

---

## Recommendations

### Immediate (2-3 hours)
1. **Fix task naming** - Standardize to underscores
2. **Fix user path** - Remove from documentation
3. **Add error handling** - Top 5 missing checks

### Short-term (1 day)
1. **Create path config** - Environment variables or config file
2. **Document dataset sizes** - Clear breakdown by stage
3. **Implement missing scripts** - Or remove references

### Long-term (1 week)
1. **Refactor path handling** - Centralized configuration
2. **Add comprehensive error handling** - All scripts
3. **Standardize naming** - Consistent conventions
4. **Complete TODO items** - Or create issues
5. **Add script tests** - Verify behavior

---

## Action Items Checklist

### Critical
- [ ] Standardize task directory naming (underscores vs hyphens)
- [ ] Remove/fix hardcoded user path in RUN_ME_FIRST.md

### High Priority
- [ ] Create centralized path configuration system
- [ ] Document actual dataset sizes by stage
- [ ] Add error handling to script execution paths
- [ ] Check existence of referenced scripts

### Medium Priority
- [ ] Add try-except around file operations
- [ ] Fix or remove redundant file operations
- [ ] Implement or remove TODO sections
- [ ] Add subprocess return code checks

### Low Priority
- [ ] Standardize script naming conventions
- [ ] Remove hardcoded dates
- [ ] Improve shell script error messages
- [ ] Add script usage documentation

### Overall Script Quality: 6.5/10

Scripts are functional but need:
- Better error handling
- Consistent path management  
- Standardized naming
- Complete implementations
