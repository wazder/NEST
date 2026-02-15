# How to Cite NEST

If you use NEST in your research, please cite:

## BibTeX

```bibtex
@software{nest2026,
  title = {NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding},
  author = {[Your Name]},
  year = {2026},
  url = {https://github.com/wazder/NEST},
  version = {1.0.0},
  note = {Open-source framework for EEG-to-text decoding}
}
```

## APA Style

[Your Name]. (2026). *NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding* (Version 1.0.0) [Computer software]. https://github.com/wazder/NEST

## IEEE Style

[Your Name], "NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding," ver. 1.0.0, 2026. [Online]. Available: https://github.com/wazder/NEST

## Plain Text

NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding, [Your Name], 2026. Available: https://github.com/wazder/NEST

---

## Related Publications

### Main Paper (Forthcoming)

```bibtex
@article{nest2026paper,
  title = {NEST: A Novel Sequence Transducer Architecture for EEG-to-Text Decoding},
  author = {[Your Name]},
  journal = {[Conference/Journal Name]},
  year = {2026},
  note = {Submitted to [NeurIPS/EMNLP/IEEE EMBC]}
}
```

*Status: In preparation for submission*

### Technical Report

```bibtex
@techreport{nest2026technical,
  title = {NEST Framework: Architecture, Implementation, and Evaluation},
  author = {[Your Name]},
  institution = {[Your Institution]},
  year = {2026},
  type = {Technical Report}
}
```

---

## Citing Components

If you use specific components of NEST, please also cite the original works:

### ZuCo Dataset

```bibtex
@article{hollenstein2018zuco,
  title = {ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  author = {Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius and Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  journal = {Scientific Data},
  volume = {5},
  pages = {180291},
  year = {2018},
  publisher = {Nature Publishing Group}
}

@inproceedings{hollenstein2020zuco2,
  title = {ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation},
  author = {Hollenstein, Nora and Troendle, Marius and Zhang, Ce and Langer, Nicolas},
  booktitle = {Proceedings of LREC},
  year = {2020}
}
```

### EEGNet Architecture

```bibtex
@article{lawhern2018eegnet,
  title = {EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author = {Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal = {Journal of Neural Engineering},
  volume = {15},
  number = {5},
  pages = {056013},
  year = {2018},
  publisher = {IOP Publishing}
}
```

### Conformer Architecture

```bibtex
@inproceedings{gulati2020conformer,
  title = {Conformer: Convolution-augmented transformer for speech recognition},
  author = {Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and others},
  booktitle = {Interspeech},
  pages = {5036--5040},
  year = {2020}
}
```

### RNN Transducer

```bibtex
@article{graves2012sequence,
  title = {Sequence transduction with recurrent neural networks},
  author = {Graves, Alex},
  journal = {arXiv preprint arXiv:1211.3711},
  year = {2012}
}
```

### CTC Loss

```bibtex
@inproceedings{graves2006connectionist,
  title = {Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks},
  author = {Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino and Schmidhuber, J{\"u}rgen},
  booktitle = {Proceedings of ICML},
  pages = {369--376},
  year = {2006}
}
```

---

## Dependencies

NEST builds upon several open-source libraries:

- **PyTorch**: Paszke et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.
- **MNE-Python**: Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.
- **Transformers**: Wolf et al. (2020). Transformers: State-of-the-art natural language processing. EMNLP.

Full dependency citations available in [DEPENDENCIES.md](DEPENDENCIES.md).

---

## Acknowledgments

When using NEST, please acknowledge:

> This work uses the NEST framework (Neural EEG Sequence Transducer) for EEG-to-text decoding, available at https://github.com/wazder/NEST.

---

## License

NEST is released under the MIT License. See [LICENSE](../LICENSE) for details.

When citing, please respect the licenses of all dependencies and datasets used.

---

## Contact

For citation questions or collaboration inquiries:
- GitHub Issues: https://github.com/wazder/NEST/issues
- Email: wazder@github.com

---

Last Updated: February 2026
