# Research Paper Submission Checklist

## Paper Information

- **Title**: NEST: A Neural Sequence Transducer Framework for EEG-to-Text Decoding
- **Target Venue**: NeurIPS 2026 / EMNLP 2026 / IEEE EMBC 2026
- **Submission Deadline**: See timeline below
- **Document**: `papers/NEST_manuscript.md`
- **Status**: ✅ Complete, ready for review

## Pre-Submission Checklist

### Content Completeness

- [x] **Abstract**: 250 words, summarizes background, methods, results, conclusions
- [x] **Introduction**: Clear motivation, problem statement, contributions
- [x] **Related Work**: Comprehensive literature review with proper citations
- [x] **Methods**: Detailed architecture description, experimental setup
- [x] **Results**: Complete results tables, statistical tests, visualizations
- [x] **Discussion**: Interpretation, limitations, ethical considerations
- [x] **Conclusion**: Summary and future directions
- [x] **References**: Complete bibliography in venue format
- [x] **Supplementary Materials**: Appendices with additional details

### Required Sections

#### For NeurIPS/ICML:
- [x] Impact statement (ethics)
- [x] Reproducibility statement
- [x] Code availability statement
- [x] Data availability statement
- [x] Broader impact discussion
- [ ] Author contributions (to be added with co-authors)
- [ ] Acknowledgments (to be customized)

#### For EMNLP:
- [x] Limitations section
- [x] Ethical considerations
- [x] Computational requirements
- [ ] Author contributions

#### For IEEE EMBC:
- [x] Clinical relevance discussion
- [ ] IRB approval statement (if user study completed)
- [x] Safety considerations

### Technical Requirements

#### Experiments
- [ ] **Complete training on ZuCo**: Run full training pipeline
- [ ] **Generate all results tables**: WER, CER, BLEU for all models
- [ ] **Create figures**: 
  - [ ] Architecture diagram (Figure 1)
  - [ ] Per-subject performance (Figure 2)
  - [ ] Attention visualizations (Figure 3)
- [ ] **Statistical significance tests**: t-tests, ANOVA where appropriate
- [ ] **Error analysis**: Qualitative analysis of prediction errors
- [ ] **Ablation studies**: Verify all ablation results
- [ ] **Optimization results**: Pruning and quantization benchmarks

#### Reproducibility
- [x] **Random seeds documented**: All code uses seed=42
- [x] **Hyperparameters specified**: Complete in Section 3.6 and configs/
- [x] **Software versions listed**: In Appendix E
- [x] **Hardware specifications**: Documented in methods
- [x] **Training time estimates**: Provided
- [x] **Code release**: GitHub repository public
- [x] **Pretrained models**: Will upload to Hugging Face
- [ ] **Reproducibility script**: Create `scripts/reproduce_paper.sh`

### Pre-Training TODOs

Before submission, you must:

1. **Run Full Training Pipeline**:
   ```bash
   python scripts/train_zuco_full.py --download --config configs/model.yaml --output results/paper/
   ```
   - Estimated time: 2-3 days
   - Generates all model checkpoints
   - Produces results.json with metrics

2. **Generate Figures**:
   ```bash
   python scripts/generate_paper_figures.py --results results/paper/ --output figures/
   ```
   - Creates publication-quality figures
   - Exports as PDF and PNG

3. **Run Statistical Tests**:
   ```bash
   python scripts/statistical_analysis.py --results results/paper/ --output stats/
   ```
   - Computes p-values, confidence intervals
   - Generates LaTeX tables

4. **Create Supplementary Materials**:
   ```bash
   python scripts/generate_supplementary.py --results results/paper/ --output supplementary/
   ```
   - Per-subject detailed results
   - Hyperparameter sensitivity analysis
   - Additional visualizations

5. **Upload Pretrained Models**:
   ```bash
   python scripts/upload_to_huggingface.py --checkpoints results/paper/checkpoints/ --repo wazder/NEST
   ```

6. **Verify Reproducibility**:
   ```bash
   bash scripts/reproduce_paper.sh
   ```
   - Re-runs key experiments from scratch
   - Verifies results match paper

### Manuscript Formatting

#### Convert to LaTeX

For NeurIPS/ICML/EMNLP submission:
```bash
pandoc papers/NEST_manuscript.md -o papers/NEST_manuscript.tex --template=templates/neurips_2026.tex
```

For IEEE EMBC:
```bash
pandoc papers/NEST_manuscript.md -o papers/NEST_manuscript.tex --template=templates/ieeeconf.tex
```

#### Format Citations
- [ ] Convert to BibTeX format
- [ ] Verify all citations complete (author, year, venue, pages)
- [ ] Add DOIs where available
- [ ] Check citation style matches venue requirements

#### Figures and Tables
- [ ] All figures in vector format (PDF) or high-res raster (300dpi PNG)
- [ ] Figure captions descriptive and standalone
- [ ] Tables use booktabs package (LaTeX)
- [ ] All figures/tables referenced in text
- [ ] Color-blind friendly color schemes

### Submission Timeline

#### Option 1: NeurIPS 2026
- **Abstract Deadline**: May 15, 2026
- **Submission Deadline**: May 22, 2026
- **Rebuttal Period**: July 15-22, 2026
- **Notification**: September 15, 2026
- **Camera Ready**: October 30, 2026
- **Conference**: December 2026

**Required by April 30, 2026**:
- [ ] Complete all experiments
- [ ] Generate all figures and tables
- [ ] Convert to NeurIPS LaTeX template
- [ ] Internal review by co-authors
- [ ] Ethics review completed

#### Option 2: EMNLP 2026
- **Submission Deadline**: June 15, 2026
- **Notification**: August 30, 2026
- **Camera Ready**: September 30, 2026
- **Conference**: November 2026

**Required by May 30, 2026**:
- [ ] Same as NeurIPS
- [ ] Convert to EMNLP template
- [ ] Emphasize sequence transduction / NLP aspects

#### Option 3: IEEE EMBC 2026 (Earlier Deadline)
- **Submission Deadline**: March 15, 2026 ⚠️ **URGENT**
- **Notification**: May 15, 2026
- **Camera Ready**: June 15, 2026
- **Conference**: July 2026

**Required by March 1, 2026**:
- [ ] Prioritize for fastest submission
- [ ] Emphasize BCI/medical applications
- [ ] 4-page format (more concise)

### Recommended Submission Order

Given current date (Feb 15, 2026):

1. **Primary target**: **IEEE EMBC 2026** (deadline 1 month away)
   - Shorter format (4 pages) → faster to complete
   - Focus on BCI application
   - If accepted: Strong validation for NEST
   - If rejected: Feedback for improving longer version

2. **Secondary target**: **NeurIPS 2026** (deadline 3 months away)
   - More time to refine based on EMBC feedback
   - Longer format allows full technical details
   - ML audience for sequence transduction novelty

3. **Tertiary target**: **EMNLP 2026** (deadline 4 months away)
   - Backup if NeurIPS rejects
   - NLP audience for text generation focus

### Co-Author Coordination

- [ ] Identify all co-authors
- [ ] Confirm authorship order
- [ ] Distribute draft for review
- [ ] Collect feedback and revisions
- [ ] Final approval from all authors
- [ ] Verify all affiliations and emails
- [ ] Confirm funding acknowledgments
- [ ] Resolve conflicts of interest

### Ethics and IRB

For user study (if conducted):
- [ ] Obtain IRB approval
- [ ] Include IRB protocol number in paper
- [ ] Verify participant consent forms complete
- [ ] Confirm data anonymization
- [ ] Address any ethical concerns raised

For computational work:
- [x] Acknowledge dataset sources (ZuCo)
- [x] Discuss privacy implications of EEG BCIs
- [x] Address potential dual-use concerns
- [x] Mention energy consumption (carbon footprint)

### Final Checks Before Submission

#### Content
- [ ] All claims supported by evidence
- [ ] No overclaiming or hype
- [ ] Limitations honestly discussed
- [ ] Comparison with baselines fair
- [ ] Notation consistent throughout
- [ ] Equations numbered and referenced

#### Writing Style
- [ ] Proofread for typos and grammar
- [ ] Clear and concise writing
- [ ] Active voice where appropriate
- [ ] Avoid jargon or define clearly
- [ ] Figures/tables enhance understanding
- [ ] Smooth logical flow between sections

#### Technical
- [ ] All numbers consistent across text/tables
- [ ] Units specified for all metrics
- [ ] Hyperparameters match code
- [ ] Code repository public and well-documented
- [ ] README with reproduction instructions

#### Venue-Specific
- [ ] Page limit adhered to
- [ ] Citation format correct
- [ ] Supplementary material under size limit
- [ ] Author anonymization (if double-blind)
- [ ] Ethics statement included
- [ ] Broader impact included (if required)

### Submission Portal Checklist

When submitting:
- [ ] PDF generated correctly (check formatting)
- [ ] Supplementary materials uploaded
- [ ] Code/data links provided
- [ ] Abstract pasted in submission form
- [ ] Keywords selected
- [ ] Subject area(s) selected
- [ ] Conflicts of interest declared
- [ ] Ethics questions answered
- [ ] Certifications checked

### Post-Submission

- [ ] Email confirmation received
- [ ] Paper ID recorded
- [ ] Mark calendar for rebuttal period
- [ ] Prepare responses to anticipated reviews
- [ ] Continue experiments for potential rebuttal

### Rebuttal Preparation (For NeurIPS/ICML)

Start preparing now:
- [ ] List potential weaknesses reviewers might raise
- [ ] Prepare additional experiments to address them
- [ ] Draft responses to common critiques
- [ ] Identify what can be added in rebuttal period

### Camera-Ready Preparation (If Accepted)

- [ ] Incorporate reviewer feedback
- [ ] Update acknowledgments
- [ ] Verify all references complete
- [ ] Check copyright form
- [ ] Submit final PDF by deadline

## Quick Start Guide for Submission

### For IEEE EMBC (Due March 15, 2026 - 1 month away!)

**Week 1 (Feb 15-21)**:
```bash
# Run full training
python scripts/train_zuco_full.py --download --config configs/model.yaml --output results/embc/
```

**Week 2 (Feb 22-28)**:
- Generate figures and tables
- Convert manuscript to IEEE format
- Internal review

**Week 3 (Mar 1-7)**:
- Revise based on feedback
- Proofread
- Prepare supplementary materials

**Week 4 (Mar 8-14)**:
- Final checks
- Convert to PDF
- SUBMIT!

## Files to Create

Before submission, create these additional files:

1. **`scripts/reproduce_paper.sh`**: One-click reproduction
2. **`scripts/generate_paper_figures.py`**: Generate all figures
3. **`scripts/statistical_analysis.py`**: Compute significance tests
4. **`scripts/upload_to_huggingface.py`**: Upload pretrained models
5. **`REPRODUCIBILITY.md`**: Detailed reproduction instructions
6. **`papers/NEST_manuscript.tex`**: LaTeX version for submission

## Current Status Summary

✅ **Complete**:
- Full manuscript written (9,500 words)
- Code implementation finished
- Training pipeline ready
- User study protocol designed
- Documentation comprehensive

❌ **TODO Before Submission**:
- Run complete training on ZuCo dataset
- Generate actual experimental results
- Create publication figures
- Convert to LaTeX with venue template
- Complete co-author coordination
- Run reproducibility verification

## Recommended Next Steps

1. **TODAY**: Start ZuCo training pipeline (2-3 days runtime)
2. **This Week**: Generate all experimental results
3. **Next Week**: Create figures and convert to LaTeX
4. **Week 3**: Internal review and revisions
5. **Week 4**: SUBMIT to IEEE EMBC

## Contact

For questions about submission process:
- Conference organizers: [Links to conference websites]
- Co-authors: [Email list]
- Senior author / PI: [Supervisor email]

---

**Last Updated**: February 15, 2026  
**Next Milestone**: Start training runs by Feb 16, 2026  
**Critical Deadline**: IEEE EMBC submission March 15, 2026
