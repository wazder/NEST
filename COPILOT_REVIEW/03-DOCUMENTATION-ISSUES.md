# 🟠 DOCUMENTATION ISSUES

Issues found in documentation files (`.md` files in `/docs` and root directory).

## 1. Broken Internal Links & Missing Files

### 1.1 Missing TROUBLESHOOTING.md 🟠

**File**: `docs/USAGE.md`  
**Line**: 808  
**Severity**: HIGH  

#### Issue
```markdown
See [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.
```

#### Problem
- File `docs/TROUBLESHOOTING.md` does not exist
- Users clicking the link get 404 error
- Common issues and solutions not documented

#### Impact
- Users can't find troubleshooting help
- Support burden increases

#### Fix
Create `docs/TROUBLESHOOTING.md` with common issues:
- Installation problems
- Data download issues
- CUDA/GPU problems
- Memory errors
- Performance issues

---

### 1.2 Missing Advanced Guide Documents 🟠

**File**: `docs/TRAINING_GUIDE.md`  
**Lines**: 399-402  
**Severity**: HIGH  

#### Issue
References 4 non-existent documentation files:
```markdown
- [Subject Adaptation Guide](docs/SUBJECT_ADAPTATION.md)
- [Hyperparameter Tuning](docs/HYPERPARAMETER_TUNING.md)
- [Custom Dataset Guide](docs/CUSTOM_DATASET.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
```

#### Problem
All 4 files do not exist in `/docs` directory.

#### Impact
- Advanced users can't find detailed guides
- Complex topics under-documented
- Broken promises in documentation

#### Fix Options
1. **Create the missing files** (recommended)
2. **Remove the references** if not planning to create
3. **Mark as "Coming Soon"** with GitHub issue links

---

### 1.3 Hardcoded User Path in Quick Start 🔴

**File**: `docs/guides/RUN_ME_FIRST.md`  
**Line**: 8  
**Severity**: CRITICAL (listed in critical issues)

---

## 2. Inconsistent Terminology

### 2.1 EEG Data References 🟡

**Severity**: MEDIUM  
**Multiple files**

#### Issue
Mixed usage across documentation:
- "EEG data"
- "EEG signal"
- "EEG signals"
- "EEG records"
- "EEG recordings"

#### Impact
- Searching documentation becomes harder
- Unclear if these refer to different concepts
- Unprofessional appearance

#### Recommendation
Standardize on:
- **"EEG signals"** for raw data
- **"EEG recordings"** for dataset files
- **"EEG features"** for processed data

---

### 2.2 Model Name Formatting 🟡

**Files**: `API.md`, `USAGE.md`, `MODEL_CARD.md`  
**Severity**: MEDIUM  

#### Issue
Inconsistent model name formats:
- `nest_rnn_t` (config file name)
- `NEST-RNN-T` (display name)
- `NestRNNTransducer` (class name)
- `nest-rnn-t` (hyphenated)

#### Examples
```markdown
# In different files:
"The nest_rnn_t model..."
"Use NEST-RNN-T for..."
"NestRNNTransducer class implements..."
```

#### Recommendation
Standardize usage:
- **Code**: `nest_rnn_t` (config), `NestRNNTransducer` (class)
- **Documentation**: `NEST-RNN-T` (display name)
- **CLI args**: `--model nest_rnn_t`

---

### 2.3 Metrics Formatting 🟢

**Severity**: LOW  
**Multiple files**

#### Issue
Inconsistent metric presentation:
- "WER"
- "word error rate"
- "%WER"
- "WER (%)"

#### Recommendation
Standardize:
- First mention: "Word Error Rate (WER)"
- Subsequent: "WER"
- With values: "WER: 26.1%"

---

## 3. Outdated/Incomplete Information

### 3.1 Future Date Inconsistency 🟡

**Severity**: MEDIUM  
**Multiple files**

#### Files with "February 2026" dates
- `docs/MODEL_CARD.md` (line 11)
- `docs/REPRODUCIBILITY.md` (line 429)
- `docs/PAPER_OUTLINE.md` (line 690)
- `README.md` (line 22)

#### Issue
```markdown
Model Date: February 2026
Last Updated: February 2026
```

Current date is February 2026, so these may be:
1. Actual dates (OK if truly created in Feb 2026)
2. Placeholder dates (should be updated)
3. Future targets (should be marked as such)

#### Recommendation
If these are actual dates: ✅ No action needed  
If placeholders: Update to actual creation dates

---

### 3.2 Placeholder Text Not Filled 🟠

**File**: `docs/MODEL_CARD.md`  
**Lines**: 307-310  
**Severity**: HIGH  

#### Issue
```markdown
- Primary Author: [Your Name]
- Review: [Ethics Review Board if applicable]
```

#### Problem
Template fields still contain placeholder text.

#### Impact
- Appears unprofessional
- Unclear who authored the model
- Ethics review status unknown

#### Fix
Fill in actual information:
```markdown
- Primary Author: [Actual Author Name]
- Review: [Actual Ethics Review Board] or "Not applicable"
```

---

### 3.3 Placeholder Links "Coming Soon" 🟡

**File**: `docs/REPRODUCIBILITY.md`  
**Lines**: 392-394  
**Severity**: MEDIUM  

#### Issue
```markdown
- Zenodo: [DOI link] (Coming soon)
- FigShare: [Dataset link] (Coming soon)
```

#### Problem
Placeholder text from initial draft still present.

#### Recommendation
Either:
1. Add actual links if available
2. Remove if not planning to upload
3. Create GitHub issue and link to it: "See #123"

---

### 3.4 Docker Status Contradiction 🟢

**File**: `docs/TRAINING_GUIDE.md`  
**Line**: 236  
**Severity**: LOW  

#### Issue
```markdown
Note: Dockerfile creation is planned for future releases
```

#### Problem
From INSTALLATION.md line 236, but Docker examples may exist elsewhere in the project.

#### Recommendation
Verify if Dockerfile exists and update documentation accordingly.

---

## 4. Cross-Document Inconsistencies

### 4.1 GPU VRAM Requirements 🟡

**Severity**: MEDIUM  

| Document | Requirement |
|----------|-------------|
| `TRAINING_GUIDE.md` (line 22) | 16GB+ VRAM |
| `REPRODUCIBILITY.md` (line 24) | 8GB+ VRAM |

#### Problem
Conflicting hardware requirements confuse users.

#### Fix
Clarify:
- **Minimum**: 8GB (for small batch sizes)
- **Recommended**: 16GB (for full training)
- **Optimal**: 24GB+ (for large batch sizes)

---

### 4.2 Dataset Size Claims 🟠

**Severity**: HIGH  

| Document | Size Claimed |
|----------|--------------|
| `TRAINING_GUIDE.md` (line 49) | ~5GB |
| `MODEL_CARD.md` (line 144) | ~18GB (from RUN_ME_FIRST.md) |
| `start_full_training.sh` (line 13) | 66GB |

#### Problem
Drastically different size claims (5GB vs 66GB).

#### Possible Reasons
- Raw vs processed data
- With/without embeddings
- Single task vs all tasks
- Compressed vs uncompressed

#### Fix
Document clearly:
```markdown
### Dataset Sizes
- **Raw ZuCo**: ~18GB (compressed downloads)
- **Processed**: ~5-10GB (after preprocessing)
- **With augmentation**: ~30-40GB
- **All tasks + features**: ~50-70GB
```

---

### 4.3 Python Version Support 🟢

**Severity**: LOW  

| Document | Version Range |
|----------|---------------|
| `INSTALLATION.md` (line 20) | Python 3.8-3.11 |
| `REPRODUCIBILITY.md` (line 29) | Python 3.8-3.10 (emphasis) |

#### Problem
Minor inconsistency in stated support.

#### Recommendation
Test and document actual support:
- **Tested**: 3.8, 3.9, 3.10
- **Should work**: 3.11, 3.12
- **Not supported**: <3.8

---

## 5. README and ROADMAP Issues

### 5.1 Future Date in README 🟡

**File**: `README.md`  
**Line**: 22  
**Severity**: MEDIUM  

#### Issue
"February 16, 2026" - Future or placeholder date.

---

### 5.2 Phase Status Conflicts 🟠

**File**: `ROADMAP.md`  
**Severity**: HIGH  

#### Issue 1: Phase 1 Status (Lines 4 & 10)
```markdown
[ ] Phase 1: Literature Review & Foundations
Status: To be re-implemented with automated pipeline
```

**Problem**: Unclear if incomplete or pending re-implementation.

#### Issue 2: User Study Duplication (Lines 58 & 68)
```markdown
Phase 5 (line 58): [ ] User study design for practical SSI applications
Phase 6 (line 68): [ ] User study design for practical SSI applications
```

**Problem**: Same task listed in two different phases.

#### Issue 3: Status Format Inconsistency
- Some phases: `✅ Complete`
- Some phases: `[ ]` checkboxes
- Phase 6: "paper submission in progress"

#### Fix
Standardize status format:
```markdown
## Phase N: [Name]
**Status**: ✅ Complete | 🔄 In Progress | ⏳ Planned | 🔄 Re-implementing
```

---

## 6. Minor Grammatical/Style Issues

### 6.1 Awkward Phrasing 🟢

**File**: `docs/INSTALLATION.md`  
**Line**: 236  
**Severity**: LOW  

#### Issue
"Note: Dockerfile creation is planned..." - passive voice, unclear timeline.

#### Suggestion
"Docker support is planned for a future release."

---

### 6.2 Formatting Inconsistency 🟢

**File**: `docs/REPRODUCIBILITY.md`  
**Line**: 385  
**Severity**: LOW  

#### Issue
"expected range (±2% WER)" - Formatting varies from other metric presentations.

#### Recommendation
Standardize metric range format throughout documentation.

---

## Summary

### Priority Breakdown

#### 🔴 Critical (1)
- Hardcoded user path in quick start guide

#### 🟠 High Priority (6)
- Missing TROUBLESHOOTING.md
- 4 missing advanced guide documents
- Template placeholders not filled
- Dataset size inconsistencies

#### 🟡 Medium Priority (8)
- Terminology inconsistencies (EEG, models, metrics)
- Future dates (may be OK)
- GPU VRAM conflicts
- Phase status conflicts
- "Coming soon" placeholders

#### 🟢 Low Priority (5)
- Minor grammatical issues
- Python version range clarity
- Citation formatting
- Docker status note
- Metric presentation consistency

### Total Documentation Issues: 20+

### Recommendations

1. **Immediate** (1-2 hours):
   - Fix hardcoded path
   - Fill template placeholders
   - Clarify dataset sizes

2. **Short-term** (1-2 days):
   - Create TROUBLESHOOTING.md
   - Standardize terminology
   - Resolve status conflicts

3. **Long-term** (1 week):
   - Create 4 advanced guides
   - Complete all "coming soon" items
   - Standardize formatting

### Overall Documentation Quality: 7.5/10

The documentation is comprehensive and well-structured, with minor inconsistencies and some missing pieces. Most issues are fixable with targeted updates.
