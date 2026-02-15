# Phase 2: Data Acquisition & Preprocessing

## Overview

Phase 2 implements the complete data acquisition and preprocessing pipeline for the NEST project. This phase focuses on preparing EEG data for neural sequence transduction models.

## Components

### 1. Dataset Management

#### ZuCo Dataset Handler
- **Module**: `src/data/zuco_dataset.py`
- **Features**:
  - Automatic dataset download from OSF repository
  - Support for multiple ZuCo tasks (SR, NR, TSR)
  - MATLAB data file loading and parsing
  - Exploratory data analysis utilities

**Usage**:
```python
from src.data import ZuCoDataset

# Initialize dataset handler
dataset = ZuCoDataset(data_dir="data/raw/zuco")

# Download dataset
dataset.download_dataset(tasks=["task1_SR"])

# Get dataset information
info = dataset.get_dataset_info()
print(info)
```

### 2. Signal Filtering

#### Band-Pass Filtering (0.5-50 Hz)
- **Module**: `src/preprocessing/filtering.py`
- **Features**:
  - Configurable band-pass filtering
  - Notch filtering for powerline noise removal
  - Frequency band extraction (delta, theta, alpha, beta, gamma)
  - Common Average Reference (CAR) filtering
  - Laplacian spatial filtering

**Usage**:
```python
from src.preprocessing import EEGFilter

# Initialize filter
eeg_filter = EEGFilter(
    l_freq=0.5,
    h_freq=50.0,
    notch_freq=50.0,
    sfreq=500.0
)

# Apply complete filtering pipeline
filtered_data = eeg_filter.apply_all_filters(raw_data)

# Extract specific frequency bands
bands = eeg_filter.get_frequency_bands(raw_data)
```

### 3. Artifact Removal

#### Independent Component Analysis (ICA)
- **Module**: `src/preprocessing/artifact_removal.py`
- **Features**:
  - ICA-based artifact removal (FastICA, Infomax, Picard)
  - Automatic EOG component detection
  - Correlation-based artifact identification
  - Amplitude and variance-based rejection

**Usage**:
```python
from src.preprocessing import ICArtifactRemoval

# Initialize ICA
ica_cleaner = ICArtifactRemoval(
    n_components=20,
    method='fastica'
)

# Fit ICA
ica_cleaner.fit(data, sfreq=500.0, ch_names=channel_names)

# Find and exclude artifact components
eog_components = ica_cleaner.find_bads_eog(data, sfreq=500.0)
ica_cleaner.exclude_components(eog_components)

# Apply ICA cleaning
cleaned_data = ica_cleaner.apply(data, sfreq=500.0)
```

### 4. Electrode Selection

#### Channel Optimization
- **Module**: `src/preprocessing/electrode_selection.py`
- **Features**:
  - Selection by channel names or brain regions
  - Variance-based selection
  - Mutual information-based selection
  - Correlation-based selection
  - PCA dimensionality reduction

**Usage**:
```python
from src.preprocessing import ElectrodeSelector

# Initialize selector
selector = ElectrodeSelector(ch_names=channel_names)

# Select by variance
selected_indices, variance = selector.select_by_variance(
    data, n_channels=32, method='highest'
)

# Apply selection
selected_data, selected_names = selector.apply_selection(
    data, selected_indices
)
```

### 5. Data Augmentation

#### EEG-Specific Augmentation
- **Module**: `src/preprocessing/augmentation.py`
- **Features**:
  - Gaussian noise addition
  - Amplitude scaling
  - Time shifting and warping
  - Channel dropout
  - Segment shuffling
  - Mixup augmentation
  - Frequency shifting
  - Band-limited noise

**Usage**:
```python
from src.preprocessing import EEGAugmentation

# Initialize augmenter
augmenter = EEGAugmentation(random_state=42)

# Apply specific augmentations
noisy = augmenter.add_gaussian_noise(data, noise_level=0.1)
scaled = augmenter.amplitude_scaling(data, scale_range=(0.8, 1.2))

# Random augmentation
aug_samples = augmenter.random_augment(
    data, n_augmentations=5, augmentation_prob=0.5
)
```

### 6. Data Splitting

#### Subject-Independent Evaluation
- **Module**: `src/preprocessing/data_split.py`
- **Features**:
  - Subject-independent splitting
  - Random splitting (with stratification)
  - Temporal splitting
  - K-fold cross-validation
  - Leave-one-subject-out (LOSO) CV

**Usage**:
```python
from src.preprocessing import EEGDataSplitter

# Initialize splitter
splitter = EEGDataSplitter(random_state=42)

# Subject-independent split
splits = splitter.subject_independent_split(
    data, labels, subject_ids,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Access splits
train_data, train_labels = splits['train']
val_data, val_labels = splits['val']
test_data, test_labels = splits['test']
```

### 7. Complete Pipeline

#### Orchestrated Preprocessing
- **Module**: `src/preprocessing/pipeline.py`
- **Features**:
  - End-to-end preprocessing workflow
  - YAML-based configuration
  - Progress tracking
  - Intermediate result saving
  - Automatic data export

**Usage**:
```python
from src.preprocessing import PreprocessingPipeline

# Initialize pipeline with config
pipeline = PreprocessingPipeline(
    config_path='configs/preprocessing.yaml'
)

# Run complete pipeline
splits = pipeline.run_pipeline(
    data=raw_data,
    labels=labels,
    sfreq=500.0,
    ch_names=channel_names,
    subject_ids=subject_ids
)
```

## Configuration

The preprocessing pipeline is configured via YAML file (`configs/preprocessing.yaml`):

```yaml
dataset:
  name: zuco
  data_dir: data/raw/zuco
  tasks: [task1_SR]

filtering:
  l_freq: 0.5
  h_freq: 50.0
  notch_freq: 50.0
  sfreq: 500.0

ica:
  enabled: true
  n_components: 20
  method: fastica
  threshold: 3.0

electrode_selection:
  enabled: true
  method: variance
  n_channels: 32

augmentation:
  enabled: true
  n_augmentations: 3
  noise_level: 0.05
  scale_range: [0.9, 1.1]

splitting:
  method: subject_independent
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Directory Structure

```
NEST/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── zuco_dataset.py
│   └── preprocessing/
│       ├── __init__.py
│       ├── filtering.py
│       ├── artifact_removal.py
│       ├── electrode_selection.py
│       ├── augmentation.py
│       ├── data_split.py
│       └── pipeline.py
├── data/
│   ├── raw/
│   │   └── zuco/
│   └── processed/
├── configs/
│   └── preprocessing.yaml
└── notebooks/
```

## Running the Pipeline

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Pipeline

Edit `configs/preprocessing.yaml` to set your preferences.

### 3. Run Preprocessing

```python
from src.preprocessing import PreprocessingPipeline
import numpy as np

# Load your raw data
# data shape: (n_samples, n_channels, n_timepoints)
# labels shape: (n_samples,)
# subject_ids shape: (n_samples,)

# Initialize and run pipeline
pipeline = PreprocessingPipeline('configs/preprocessing.yaml')
splits = pipeline.run_pipeline(
    data=raw_data,
    labels=labels,
    sfreq=500.0,
    ch_names=channel_names,
    subject_ids=subject_ids
)

print(f"Train: {splits['train'][0].shape}")
print(f"Val: {splits['val'][0].shape}")
print(f"Test: {splits['test'][0].shape}")
```

### 4. Access Processed Data

```python
from src.preprocessing import DatasetOrganizer

# Load processed splits
splits = DatasetOrganizer.load_splits('data/processed')

# Get statistics
stats = DatasetOrganizer.get_split_statistics(splits)
print(stats)
```

## Signal Processing Details

### Filtering Specifications
- **High-pass**: 0.5 Hz (removes slow drifts)
- **Low-pass**: 50 Hz (removes high-frequency noise)
- **Notch filter**: 50/60 Hz (removes powerline interference)
- **Method**: FIR (Finite Impulse Response) with zero-phase

### ICA Parameters
- **Algorithm**: FastICA (default), Infomax, or Picard
- **Components**: 20 (typical for 32-64 channel montages)
- **Artifacts**: Eye blinks, eye movements, muscle artifacts
- **Detection**: Automatic using correlation and spatial patterns

### Augmentation Strategy
- **Noise**: SNR-controlled Gaussian noise
- **Scaling**: ±20% amplitude variation
- **Temporal**: Jittering and warping
- **Spatial**: Channel dropout

## Best Practices

1. **Preprocessing Order**:
   - Always filter before ICA
   - Apply ICA before electrode selection
   - Augment after all cleaning steps

2. **Subject Independence**:
   - Use subject-independent splits for realistic evaluation
   - Never mix subjects between train/val/test

3. **Reproducibility**:
   - Set random seeds in config
   - Save preprocessing config with processed data
   - Document all parameter choices

4. **Quality Control**:
   - Visually inspect ICA components
   - Check for remaining artifacts
   - Verify split balance

## Phase 2 Checklist

- [x] ZuCo dataset acquisition and exploratory data analysis
- [x] Implementation of band-pass filtering (0.5–50 Hz) for artifact removal
- [x] Development of Independent Component Analysis (ICA) pipeline for eye-blink/muscle artifact rejection
- [x] Electrode selection and channel optimization strategies
- [x] Data augmentation techniques for limited EEG samples
- [x] Train/validation/test split with subject-independent evaluation protocol

## Next Steps

Phase 3 will focus on model architecture development, including:
- CNN-based spatial feature extraction
- Temporal encoding (LSTM/Transformer)
- Cross-attention mechanisms
- CTC loss integration

## References

1. Hollenstein, N., et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. Nature Scientific Data.
2. Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. Journal of Neuroscience Methods.
3. MNE-Python: https://mne.tools/
