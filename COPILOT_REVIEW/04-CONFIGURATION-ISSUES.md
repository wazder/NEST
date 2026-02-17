# 🟡 CONFIGURATION FILE ISSUES

Issues found in configuration files (YAML, TOML, CFG, requirements).

## 1. Duplicate Configuration Definitions 🔴

**Severity**: CRITICAL  
**Files**: `setup.cfg`, `pyproject.toml`  
(Already covered in 01-CRITICAL-ISSUES.md)

### Issue
Both files define overlapping tool configurations.

### Specific Conflicts

#### pytest Configuration
**setup.cfg** (line 22):
```ini
[tool:pytest]
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    benchmark: marks tests as benchmarks  # ← Present
```

**pyproject.toml** (lines 17-23):
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    # benchmark marker missing
]
```

#### Coverage Configuration
**setup.cfg** (line 30):
```ini
[coverage:run]
omit =
    */tests/*
    */test_*.py
    *conftest.py  # ← Excludes conftest
```

**pyproject.toml**:
```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/test_*.py",
    # conftest.py not excluded
]
```

### Impact
- Different test behavior depending on which config is used
- Coverage reports may differ
- CI/CD may behave differently than local dev

### Fix
Migrate everything to `pyproject.toml` (modern standard):
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmarks",
]

[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*conftest.py",
]
```

Then remove tool configs from `setup.cfg`, keep only packaging metadata.

---

## 2. Requirements File Duplication 🔴

**Severity**: CRITICAL  
**Files**: `requirements.txt`, `requirements-dev.txt`  
(Already covered in 01-CRITICAL-ISSUES.md)

### Detailed List of Duplicates

| Package | requirements.txt | requirements-dev.txt |
|---------|------------------|----------------------|
| pytest | Line 30 | Line 5 |
| pytest-cov | Line 31 | Line 6 |
| pytest-xdist | Line 32 | Line 7 |
| pytest-timeout | Line 33 | Line 8 |
| pytest-benchmark | Line 34 | Line 9 |
| black | Line 35 | Line 13 |
| flake8 | Line 36 | Line 15 |
| isort | Line 37 | Line 14 |
| mypy | Line 38 | Line 19 |
| radon | Line 39 | Line 26 |
| pylint | Line 40 | Line 18 |
| safety | Line 41 | Line 23 |
| bandit | Line 42 | Line 22 |
| coverage | Line 43 | Line 27 |
| sphinx | Line 50 | Line 30 |
| sphinx-rtd-theme | Line 51 | Line 31 |

### Impact
- Production installs include dev tools (unnecessary)
- Docker images larger than needed
- Potential security surface increase
- Deployment bloat

---

## 3. Model Configuration Analysis 🟢

**File**: `configs/model.yaml`  
**Severity**: LOW (mostly OK)

### Dimension Chain Validation

#### nest_rnn_t (Lines 5-53)
```yaml
spatial_cnn:
  out_channels: 128

temporal_encoder:
  input_size: 128  # ✓ Matches spatial_cnn.out_channels
  hidden_size: 512
  num_layers: 4
  bidirectional: true

joint:
  encoder_dim: 1024  # = hidden_size * 2 (bidirectional) ✓
  decoder_dim: 512
  joint_dim: 640
```

**Status**: ✅ All dimensions compatible

#### nest_attention (Lines 104-147)
```yaml
temporal_encoder:
  hidden_size: 512
  bidirectional: true

attention:
  encoder_dim: 1024  # = hidden_size * 2 ✓
  decoder_dim: 512

decoder:
  encoder_dim: 1024  # ✓ Matches attention.encoder_dim
```

**Status**: ✅ All dimensions compatible

#### nest_conformer (Lines 193-247)
```yaml
# Similar pattern, all dimensions check out ✓
```

**Status**: ✅ All dimensions compatible

### Minor Issues Found

#### 3.1 Missing batch_norm Configuration 🟢
**Severity**: LOW

All CNN configurations missing explicit batch normalization settings:
```yaml
spatial_cnn:
  in_channels: 104
  out_channels: 128
  kernel_size: 3
  # Missing: batch_norm: true
  # Missing: dropout: 0.1
```

**Recommendation**:
```yaml
spatial_cnn:
  in_channels: 104
  out_channels: 128
  kernel_size: 3
  batch_norm: true  # Add explicit flag
  dropout: 0.1      # Add dropout config
```

#### 3.2 Inconsistent Parameter Naming 🟢
**Severity**: LOW

Some places use `n_layers`, others use `num_layers`:
```yaml
# In some configs:
num_layers: 4

# In others:
n_layers: 4
```

**Recommendation**: Standardize on `num_layers` (more explicit).

---

## 4. Preprocessing Configuration 🟢

**File**: `configs/preprocessing.yaml`  
**Severity**: LOW

### Issue: Task Naming Convention
**Line**: 10
```yaml
tasks:
  - task1_SR  # ← Using underscores
# But comments show:
# - task1-SR  # With hyphens
# - task2-NR
# - task3-TSR
```

**Problem**: Inconsistent with task naming in scripts (see Script Issues).

**Recommendation**: Standardize on underscores:
```yaml
tasks:
  - task1_SR
  - task2_NR
  - task3_TSR
```

---

## 5. Version Specifications 🟡

**File**: `requirements.txt`  
**Severity**: MEDIUM

### Issue: No Upper Bounds

Most packages specify only lower bounds:
```txt
torch>=2.0.0         # No upper bound
transformers>=4.30.0 # No upper bound
lightning>=2.0.0     # No upper bound
scipy>=1.7.0         # No upper bound
```

### Problem
- Major version bumps can break compatibility
- No protection against breaking changes
- Unpredictable behavior in future installs

### Example Risks
```txt
torch>=2.0.0  # Could install torch 3.0.0 if it exists
              # Which might have breaking API changes
```

### Recommendation
Add upper bounds for critical packages:
```txt
torch>=2.0.0,<3.0.0
transformers>=4.30.0,<5.0.0
lightning>=2.0.0,<3.0.0
```

Or use caret (poetry-style):
```txt
torch~=2.0.0  # Allows 2.x.x but not 3.0.0
```

### Current Status
✅ **No version conflicts detected** between packages  
⚠️ **But future-proofing needed**

---

## 6. YAML Syntax Validation ✅

**Files**: All YAML files checked  
**Status**: ✅ All valid

Checked files:
- ✅ `configs/model.yaml` - Valid syntax
- ✅ `configs/preprocessing.yaml` - Valid syntax
- ✅ `.github/workflows/*.yml` - Valid syntax
- ✅ `.pre-commit-config.yaml` - Valid syntax

---

## 7. Setup Configuration

### 7.1 setup.cfg Analysis 🟢

**File**: `setup.cfg`  
**Severity**: LOW

#### Python Version Compatibility
```ini
[options]
python_requires = >=3.8
```

**Status**: ✅ Compatible with all specified dependencies

#### Package Discovery
```ini
[options]
packages = find:
```

**Status**: ✅ Will correctly find all packages in `src/`

---

### 7.2 pyproject.toml Analysis 🟢

**File**: `pyproject.toml`  
**Severity**: LOW

#### Build System
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools-scm>=6.2"]
build-backend = "setuptools.build_meta"
```

**Status**: ✅ Modern build system configuration

#### Tool Configurations
Most tool configurations are well-defined:
- ✅ black: max_line_length = 100
- ✅ isort: profile = "black"
- ✅ mypy: strict mode enabled

---

## Summary

### Issues by Severity

#### 🔴 Critical: 2
1. Duplicate configuration files (setup.cfg + pyproject.toml)
2. Requirements file duplication

#### 🟡 Medium: 2
1. No upper version bounds in requirements
2. Task naming inconsistency in preprocessing config

#### 🟢 Low: 5
1. Missing batch_norm configuration
2. Inconsistent parameter naming (n_layers vs num_layers)
3. Minor setup.cfg improvements possible
4. Could add more explicit defaults
5. Task naming in preprocessing.yaml

### Configuration Quality Score: 7/10

**Strengths**:
- ✅ Valid YAML syntax
- ✅ Dimension chains validated
- ✅ No version conflicts
- ✅ Modern build system

**Weaknesses**:
- ⚠️ Configuration duplication
- ⚠️ Requirements duplication
- ⚠️ Missing version upper bounds
- ⚠️ Minor naming inconsistencies

---

## Recommendations

### Immediate (1 hour)
1. Consolidate tool configs to `pyproject.toml`
2. Separate prod and dev requirements

### Short-term (2-3 hours)
1. Add version upper bounds
2. Standardize parameter naming
3. Add batch_norm configs

### Long-term (ongoing)
1. Test with latest package versions
2. Update bounds as needed
3. Monitor for breaking changes

---

## Action Items Checklist

- [ ] Migrate all tool configs to `pyproject.toml`
- [ ] Remove tool configs from `setup.cfg`
- [ ] Split requirements.txt (prod only) and requirements-dev.txt (dev only)
- [ ] Add version upper bounds to critical packages
- [ ] Standardize on `num_layers` everywhere
- [ ] Add batch_norm flags to CNN configs
- [ ] Verify task naming convention matches scripts
- [ ] Test configuration with clean environment
