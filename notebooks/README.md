# NEST Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, experimentation, and visualization.

## Notebooks

### Phase 2: Data Acquisition & Preprocessing

- `01_zuco_exploration.ipynb` - ZuCo dataset exploration and visualization
- `02_signal_filtering.ipynb` - EEG filtering techniques demo
- `03_ica_artifacts.ipynb` - ICA artifact removal examples
- `04_electrode_selection.ipynb` - Channel selection strategies
- `05_data_augmentation.ipynb` - Data augmentation visualization
- `06_preprocessing_pipeline.ipynb` - End-to-end pipeline demo

### Phase 3: Model Development

(To be added in Phase 3)

## Usage

1. Install Jupyter:
```bash
pip install jupyter notebook
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open and run notebooks in order

## Best Practices

- Keep notebooks focused on specific topics
- Document all steps with markdown cells
- Save visualizations for documentation
- Export key results to reports
- Version control notebooks (use nbstripout for clean diffs)

## Data

Notebooks expect data in the following structure:
```
NEST/
├── data/
│   ├── raw/
│   │   └── zuco/
│   └── processed/
└── notebooks/
```
