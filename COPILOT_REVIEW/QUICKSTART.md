# 🚀 QUICK START - Fix Critical Issues

This guide helps you fix the 5 critical issues that can cause runtime errors.

**Total Time**: 2-3 hours  
**Impact**: Eliminates all runtime errors ✅

---

## 1. Fix Type Annotation (5 minutes) 🔴

### Issue
Type annotation uses lowercase `any` instead of `Any`

### File
`src/data/zuco_dataset.py` line 174

### Current Code
```python
def get_dataset_info(self) -> Dict[str, any]:
    """Get dataset information."""
```

### Fixed Code
```python
def get_dataset_info(self) -> Dict[str, Any]:
    """Get dataset information."""
```

### Commands
```bash
cd /home/runner/work/NEST/NEST

# Edit the file
sed -i 's/Dict\[str, any\]/Dict[str, Any]/' src/data/zuco_dataset.py

# Verify the change
grep -n "get_dataset_info" src/data/zuco_dataset.py

# Test
python -m mypy src/data/zuco_dataset.py
```

---

## 2. Standardize Task Naming (30 minutes) 🔴

### Issue
Inconsistent task directory naming: `task1_SR` vs `task1-SR`

### Files to Update
1. `scripts/train_with_real_zuco.py` (lines 42-45)
2. `scripts/inspect_zuco_mat.py` (line ~30)
3. `scripts/download_zuco_manual.sh` (line ~20)

### Decision
Use **underscores** everywhere: `task1_SR`, `task2_NR`, `task3_TSR`

### Commands
```bash
cd /home/runner/work/NEST/NEST

# Update train_with_real_zuco.py
sed -i "s/'task1-SR'/'task1_SR'/g" scripts/train_with_real_zuco.py
sed -i "s/'task2-NR'/'task2_NR'/g" scripts/train_with_real_zuco.py
sed -i "s/'task3-TSR'/'task3_TSR'/g" scripts/train_with_real_zuco.py

# Update inspect_zuco_mat.py
sed -i "s/task1-SR/task1_SR/g" scripts/inspect_zuco_mat.py
sed -i "s/task2-NR/task2_NR/g" scripts/inspect_zuco_mat.py
sed -i "s/task3-TSR/task3_TSR/g" scripts/inspect_zuco_mat.py

# Update download_zuco_manual.sh
sed -i "s/task1-SR/task1_SR/g" scripts/download_zuco_manual.sh
sed -i "s/task2-NR/task2_NR/g" scripts/download_zuco_manual.sh
sed -i "s/task3-TSR/task3_TSR/g" scripts/download_zuco_manual.sh

# Verify
grep -n "task.*SR" scripts/train_with_real_zuco.py
grep -n "task.*SR" scripts/inspect_zuco_mat.py
```

---

## 3. Fix Hardcoded User Path (2 minutes) 🔴

### Issue
Documentation contains user-specific path

### File
`docs/guides/RUN_ME_FIRST.md` line 8

### Current Code
```bash
cd /Users/wazder/Documents/GitHub/NEST
```

### Fixed Code
```bash
# Navigate to your NEST directory
cd /path/to/NEST
# Or if already in the right directory:
# cd .
```

### Commands
```bash
cd /home/runner/work/NEST/NEST

# Create a backup
cp docs/guides/RUN_ME_FIRST.md docs/guides/RUN_ME_FIRST.md.bak

# Edit the file (manual edit recommended)
# Or use sed:
sed -i 's|cd /Users/wazder/Documents/GitHub/NEST|# Navigate to your NEST directory\ncd /path/to/NEST|' docs/guides/RUN_ME_FIRST.md
```

---

## 4. Consolidate Configuration Files (1 hour) 🔴

### Issue
Duplicate tool configurations in `setup.cfg` and `pyproject.toml`

### Solution
Keep all tool configs in `pyproject.toml` only

### Step 1: Backup Files
```bash
cd /home/runner/work/NEST/NEST

cp setup.cfg setup.cfg.bak
cp pyproject.toml pyproject.toml.bak
```

### Step 2: Update pyproject.toml

Add missing configurations:

```toml
# Add to pyproject.toml

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmarks",  # ← Add this
]
testpaths = ["tests"]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*conftest.py",  # ← Add this
]
```

### Step 3: Clean setup.cfg

Remove tool configurations, keep only packaging:

```ini
# setup.cfg - Keep only this section
[metadata]
name = nest-eeg
version = attr: src.__version__

[options]
packages = find:
python_requires = >=3.8
install_requires = file: requirements.txt

[options.packages.find]
where = .
exclude =
    tests*
    docs*
```

### Step 4: Verify
```bash
# Test that tools still work
pytest tests/unit/test_models.py -v
coverage run -m pytest tests/
mypy src/
```

---

## 5. Separate Requirements (30 minutes) 🔴

### Issue
Development dependencies in production requirements

### Solution
Move dev tools to `requirements-dev.txt` only

### Step 1: Backup
```bash
cd /home/runner/work/NEST/NEST

cp requirements.txt requirements.txt.bak
cp requirements-dev.txt requirements-dev.txt.bak
```

### Step 2: Clean requirements.txt

Remove these lines (dev tools):
```text
# Remove from requirements.txt:
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
pytest-timeout>=2.1.0
pytest-benchmark>=4.0.0
black>=22.0.0
flake8>=4.0.0
isort>=5.10.0
mypy>=1.0.0
radon>=5.1.0
pylint>=2.17.0
safety>=2.3.0
bandit>=1.7.0
coverage>=7.0.0
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
```

Keep only:
```text
# Keep in requirements.txt (production):
torch>=2.0.0
transformers>=4.30.0
lightning>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
h5py>=3.7.0
mne>=1.0.0
pyyaml>=6.0
tqdm>=4.62.0
wandb>=0.12.0
omegaconf>=2.1.0
sentencepiece>=0.1.96
```

### Step 3: Update requirements-dev.txt

Make sure it includes requirements.txt:
```text
# requirements-dev.txt
# Install production requirements first
-r requirements.txt

# Then dev tools (already present)
pytest>=7.0.0
...
```

### Step 4: Verify
```bash
# Test with clean venv
python -m venv test_venv
source test_venv/bin/activate

# Test production install
pip install -r requirements.txt
python -c "import torch; print('✓ Production deps OK')"

# Test dev install
pip install -r requirements-dev.txt
pytest --version
black --version

deactivate
rm -rf test_venv
```

---

## Verification Checklist

After making all fixes, verify:

```bash
cd /home/runner/work/NEST/NEST

# 1. Type checking passes
python -m mypy src/data/zuco_dataset.py
# Should not complain about 'any'

# 2. Task naming is consistent
grep -r "task.*SR" scripts/ configs/
# Should all use underscores

# 3. No hardcoded user paths
grep -r "/Users/wazder" docs/
# Should return nothing (or only in backups)

# 4. No duplicate tool configs
grep -A5 "\[tool:" setup.cfg
# Should return minimal or nothing

# 5. Clean requirements split
grep -E "(pytest|black|flake8)" requirements.txt
# Should return nothing
```

---

## Testing

Run these tests to ensure nothing broke:

```bash
cd /home/runner/work/NEST/NEST

# Run unit tests
pytest tests/unit/ -v

# Run type checking
mypy src/

# Run linting
flake8 src/
black --check src/
isort --check src/

# Try a quick training test
python scripts/train_with_real_zuco.py --quick-test
```

---

## If Something Goes Wrong

### Restore Backups
```bash
# If you made backups, restore them:
cp setup.cfg.bak setup.cfg
cp pyproject.toml.bak pyproject.toml
cp requirements.txt.bak requirements.txt
```

### Get Help
1. Check the detailed files in `COPILOT_REVIEW/`
2. See `01-CRITICAL-ISSUES.md` for detailed explanations
3. See `07-RECOMMENDATIONS.md` for more context

---

## After Fixing

Once all critical issues are fixed:

1. **Commit changes**:
   ```bash
   git add .
   git commit -m "Fix critical issues from COPILOT_REVIEW"
   ```

2. **Run full test suite**:
   ```bash
   pytest tests/
   ```

3. **Update documentation** if needed

4. **Move to high priority issues** (see `SUMMARY.md`)

---

## Next Steps

After fixing critical issues, tackle high priority items:

1. Create `docs/TROUBLESHOOTING.md`
2. Fill in template placeholders
3. Create missing guide documents
4. Add centralized path configuration

See `07-RECOMMENDATIONS.md` for detailed instructions.

---

## Summary

✅ **Fixed**: 5 critical issues  
⏱️ **Time**: 2-3 hours  
🎯 **Impact**: All runtime errors eliminated  
📈 **Next**: High priority issues (1-2 days)

Good luck! 🚀
