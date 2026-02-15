# NEST Tutorials

Interactive Jupyter notebooks for learning the NEST framework.

## Available Tutorials

### Beginner Level

1. **[Introduction to NEST](01_introduction.ipynb)** (Coming soon)
   - Overview of EEG-to-text decoding
   - NEST architecture basics
   - Loading and visualizing EEG data

2. **[Data Preprocessing](02_preprocessing.ipynb)** (Coming soon)
   - Band-pass filtering
   - Artifact removal with ICA
   - Data augmentation techniques

3. **[Building Your First Model](03_first_model.ipynb)** (Coming soon)
   - Creating a simple NEST model
   - Training loop basics
   - Evaluation metrics

### Intermediate Level

4. **[Advanced Architectures](04_advanced_models.ipynb)** (Coming soon)
   - Comparing NEST variants (RNN-T, Transformer-T, Conformer)
   - Attention mechanisms
   - Model selection

5. **[Subject Adaptation](05_subject_adaptation.ipynb)** (Coming soon)
   - Cross-subject challenges
   - Domain adaptation techniques
   - Transfer learning strategies

6. **[Hyperparameter Tuning](06_hyperparameter_tuning.ipynb)** (Coming soon)
   - Grid search and random search
   - Bayesian optimization
   - Learning rate scheduling

### Advanced Level

7. **[Model Optimization](07_optimization.ipynb)** (Coming soon)
   - Pruning techniques
   - Quantization strategies
   - Profiling and benchmarking

8. **[Real-time Inference](08_realtime_inference.ipynb)** (Coming soon)
   - Streaming EEG processing
   - Latency optimization
   - Production deployment

9. **[Custom Architectures](09_custom_models.ipynb)** (Coming soon)
   - Designing custom encoders
   - Implementing new attention mechanisms
   - Advanced training techniques

## Quick Start

### Running Locally

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
cd notebooks
jupyter notebook

# Open any tutorial notebook
```

### Running in Google Colab

Click the "Open in Colab" button at the top of each notebook.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Basic understanding of deep learning
- Familiarity with Python and NumPy

## Tutorial Structure

Each tutorial includes:
- **Learning Objectives**: What you'll learn
- **Theory**: Background concepts
- **Code Examples**: Working implementations
- **Exercises**: Practice problems
- **Solutions**: Reference implementations
- **Further Reading**: Additional resources

## Recommended Learning Path

### For Beginners
1. Introduction to NEST
2. Data Preprocessing
3. Building Your First Model
4. Advanced Architectures

### For Researchers
1. Advanced Architectures
2. Subject Adaptation
3. Hyperparameter Tuning
4. Custom Architectures

### For Engineers
1. Model Optimization
2. Real-time Inference
3. Building Your First Model
4. Advanced Architectures

## Additional Resources

### Datasets
- [ZuCo Dataset](https://osf.io/q3zws/) - Reading task EEG data
- Download and preprocessing instructions in tutorials

### Documentation
- [USAGE Guide](../docs/USAGE.md) - Complete framework usage
- [API Reference](../docs/API.md) - Detailed API documentation
- [Examples](../examples/) - Standalone Python scripts

### Papers
- Original NEST paper (Coming soon)
- Literature review in [docs/literature-review/](../docs/literature-review/)

## Contributing Tutorials

We welcome tutorial contributions! Guidelines:

1. **Format**: Use Jupyter notebook (.ipynb)
2. **Structure**:
   - Clear learning objectives
   - Theory section (brief)
   - Code with comments
   - Exercises
   - Solutions (in separate cell)
3. **Dependencies**: List all required packages
4. **Testing**: Ensure notebook runs end-to-end
5. **Style**: Follow PEP 8 for Python code

Submit via pull request to the main repository.

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/wazder/NEST/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wazder/NEST/discussions)
- **Examples**: Check [examples/](../examples/) directory

## Notebook Creation Plan

Tutorials are currently under development. Follow the progress:

- [ ] 01 - Introduction to NEST
- [ ] 02 - Data Preprocessing
- [ ] 03 - Building Your First Model
- [ ] 04 - Advanced Architectures
- [ ] 05 - Subject Adaptation
- [ ] 06 - Hyperparameter Tuning
- [ ] 07 - Model Optimization
- [ ] 08 - Real-time Inference
- [ ] 09 - Custom Architectures

Expected completion: Phase 6

## License

Tutorials are provided under MIT license, same as the NEST project.
