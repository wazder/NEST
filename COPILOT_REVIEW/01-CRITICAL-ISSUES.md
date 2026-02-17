# 🔴 CRITICAL ISSUES

These issues must be fixed as they cause or can cause runtime errors, type checking failures, or data loss.

## 1. Type Annotation Error in ZuCo Dataset

**Severity**: 🔴 CRITICAL  
**File**: `src/data/zuco_dataset.py`  
**Line**: 174  

### Issue
```python
def get_dataset_info(self) -> Dict[str, any]:  # ❌ WRONG
```

### Problem
- Uses lowercase `any` instead of `Any`
- `any` is a built-in function, not a type annotation
- Type checkers (mypy) will fail
- IDE type hints won't work properly

### Fix
```python
def get_dataset_info(self) -> Dict[str, Any]:  # ✓ CORRECT
```

### Impact
- Type checking fails
- CI/CD pipeline mypy checks will fail
- Developer experience degraded (no proper type hints)

---

## 2. Task Directory Naming Inconsistency

**Severity**: 🔴 CRITICAL  
**Files**: Multiple scripts  

### Issue
Scripts use inconsistent task naming conventions:
- Some use: `task1_SR`, `task2_NR`, `task3_TSR` (underscores)
- Others use: `task1-SR`, `task2-NR`, `task3-TSR` (hyphens)

### Affected Files
| File | Format Used |
|------|-------------|
| `scripts/download_zuco.py` | underscores |
| `scripts/generate_synthetic_data.py` | underscores |
| `scripts/train_with_real_zuco.py` | **hyphens** |
| `scripts/inspect_zuco_mat.py` | **hyphens** |
| `scripts/download_zuco_manual.sh` | **hyphens** |
| `scripts/download_zuco_simple.sh` | mixed |

### Problem
```python
# In train_with_real_zuco.py (line ~42-45)
task_dirs = ['task1-SR', 'task2-NR', 'task3-TSR']  # hyphens

# But download_zuco.py creates:
task_dirs = ['task1_SR', 'task2_NR', 'task3_TSR']  # underscores
```

### Impact
- Runtime `FileNotFoundError` when scripts expect one format but data uses another
- Training scripts won't find downloaded data
- Manual intervention required to rename directories

### Fix
Standardize on one convention (recommend underscores to match Python naming):
- Update all scripts to use `task1_SR` format
- Or update all scripts to use `task1-SR` format consistently

---

## 3. Duplicate Configuration Files

**Severity**: 🔴 CRITICAL  
**Files**: `setup.cfg`, `pyproject.toml`  

### Issue
Both files define overlapping tool configurations with potentially conflicting values.

### Conflicts Found

#### pytest Markers
- **setup.cfg** (line 22): Defines "benchmark" marker
- **pyproject.toml** (lines 17-23): Omits "benchmark" marker

#### Coverage Settings
- **setup.cfg** (line 30): Excludes `conftest.py`
- **pyproject.toml**: Doesn't exclude `conftest.py`

### Problem
- Tools may use different configuration files
- Behavior becomes unpredictable
- CI/CD results may differ from local development

### Impact
- Test results inconsistent
- Coverage reports differ between environments
- Difficult to debug test issues

### Fix
Choose one configuration format (recommend `pyproject.toml` as modern standard):
1. Consolidate all tool configs into `pyproject.toml`
2. Remove tool configurations from `setup.cfg`
3. Keep only packaging metadata in `setup.cfg` if needed, or migrate entirely to `pyproject.toml`

---

## 4. Hardcoded User-Specific Path

**Severity**: 🔴 CRITICAL  
**File**: `docs/guides/RUN_ME_FIRST.md`  
**Line**: 8  

### Issue
```bash
cd /Users/wazder/Documents/GitHub/NEST  # ← User-specific path
```

### Problem
- Hardcoded path to specific user's machine
- Won't work for any other user
- Copy-paste error will confuse new users

### Impact
- New users cannot follow quick start guide
- First impression of project is negative
- Documentation appears unprofessional

### Fix
```bash
cd /path/to/your/NEST  # or simply:
# Navigate to your NEST directory
cd NEST
```

---

## 5. Requirements File Duplication

**Severity**: 🔴 CRITICAL  
**Files**: `requirements.txt`, `requirements-dev.txt`  

### Issue
Development dependencies duplicated across both files:

**Duplicated packages** (15+ packages):
- pytest (and all pytest plugins)
- black, isort, flake8, mypy, pylint
- bandit, safety
- radon, coverage
- sphinx, sphinx-rtd-theme

### Problem
```text
requirements.txt (lines 30-43):
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=4.0.0
...

requirements-dev.txt (lines 5-31):
pytest>=7.0.0  # ← DUPLICATE
pytest-cov>=4.0.0  # ← DUPLICATE
black>=22.0.0  # ← DUPLICATE
...
```

### Impact
- Installing `requirements.txt` installs dev tools in production
- Larger Docker images
- Potential security surface increase
- Confusion about which file to use

### Fix
1. Remove all development/testing tools from `requirements.txt`
2. Keep only runtime dependencies in `requirements.txt`:
   - torch, transformers, lightning
   - numpy, scipy, scikit-learn
   - pyyaml, tqdm, wandb
3. Keep all dev tools in `requirements-dev.txt`
4. Update documentation to show:
   ```bash
   pip install -r requirements.txt  # Production
   pip install -r requirements-dev.txt  # Development
   ```

---

## Summary

These 5 critical issues should be addressed immediately:

1. ✅ Fix type annotation (`any` → `Any`)
2. ✅ Standardize task directory naming
3. ✅ Consolidate configuration files
4. ✅ Fix hardcoded user path
5. ✅ Separate production and dev dependencies

**Estimated effort**: 2-3 hours  
**Risk if not fixed**: Runtime errors, failed builds, user confusion
