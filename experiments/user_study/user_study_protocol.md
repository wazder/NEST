# NEST User Study Protocol

**Study Title**: Evaluation of Neural EEG Sequence Transducer for Silent Speech Interface  
**Principal Investigator**: [Your Name]  
**Institution**: [Your Institution]  
**IRB Protocol**: [Protocol Number]  
**Version**: 1.0  
**Date**: February 15, 2026

## 1. Study Overview

### 1.1 Purpose
To evaluate the performance, usability, and user acceptance of the NEST (Neural EEG Sequence Transducer) system for brain-computer interface-based text generation.

### 1.2 Research Questions
1. What is the achievable accuracy (WER/CER) of NEST with real users in controlled conditions?
2. How does subject-specific adaptation affect performance compared to subject-independent models?
3. What is the learning curve for users to effectively use the NEST system?
4. What are users' subjective experiences and acceptance of the system?
5. What are the practical limitations and error patterns observed in real-world usage?

### 1.3 Hypotheses
- H1: Subject-adapted models will achieve <15% WER after calibration
- H2: Users will achieve >80% satisfaction scores for system usability
- H3: Performance will improve by >20% between initial and final sessions
- H4: Cross-subject models will achieve <25% WER without adaptation

## 2. Participant Criteria

### 2.1 Inclusion Criteria
- Age 18-65 years
- Native English speaker or fluent (CEFR C1 or higher)
- Normal or corrected-to-normal vision
- Right-handed (for standardized electrode placement)
- No history of neurological disorders
- Able to provide informed consent
- Available for multiple sessions over 2 weeks

### 2.2 Exclusion Criteria
- History of epilepsy or seizures
- Current use of psychoactive medications
- Scalp conditions that prevent EEG electrode placement
- Claustrophobia or discomfort with EEG cap
- Pregnancy
- Prior participation in EEG-BCI studies (to avoid training effects)

### 2.3 Sample Size
- **Target**: 20 participants (12 for training data, 8 for testing)
- **Power Analysis**: Based on α=0.05, β=0.20, effect size d=0.8
- **Dropout Rate**: Recruit 25 participants to account for 20% attrition

## 3. Study Design

### 3.1 Study Type
- Repeated measures within-subjects design
- 4 sessions per participant over 2 weeks

### 3.2 Timeline

**Session 1 (90 minutes)**: Baseline & Training Data Collection
- Informed consent and screening (15 min)
- EEG cap setup and impedance check (20 min)
- Instructions and practice (10 min)
- Baseline data collection: reading task (30 min)
- Post-session questionnaire (15 min)

**Session 2 (60 minutes)**: Model Calibration
- EEG setup (15 min)
- Calibration data collection (30 min)
- Rest break (5 min)
- Model training (occurs offline)
- Initial testing (10 min)

**Session 3 (75 minutes)**: Performance Evaluation
- EEG setup (15 min)
- Text generation task: subject-specific model (25 min)
- Rest break (5 min)
- Text generation task: subject-independent model (25 min)
- Questionnaire (5 min)

**Session 4 (75 minutes)**: Final Evaluation & Feedback
- EEG setup (15 min)
- Free-form text generation (30 min)
- Cognitive load assessment (15 min)
- Final questionnaire and semi-structured interview (15 min)

### 3.3 Randomization
- Order of subject-specific vs. subject-independent model testing (Session 3)
- Order of text stimuli presented
- Counterbalancing across participants

## 4. Experimental Tasks

### 4.1 Task 1: Reading Task (Baseline Data Collection)
**Purpose**: Collect EEG data paired with known text (training data)

**Procedure**:
1. Participant reads sentences presented on screen
2. Each sentence displayed for 5-10 seconds
3. Participant reads silently while fixating on center
4. 90 sentences total from ZuCo-like stimuli
5. Breaks every 30 sentences (2 minutes)

**Stimuli**: 
- Sentences 8-20 words in length
- Mix of simple and complex syntax
- Controlled frequency and familiarity
- Drawn from standardized corpora

### 4.2 Task 2: Calibration Task
**Purpose**: Collect subject-specific data for model adaptation

**Procedure**:
1. Similar to reading task
2. 60 calibration sentences
3. Spans different linguistic constructs
4. Used to fine-tune model to participant

### 4.3 Task 3: Cued Text Generation
**Purpose**: Evaluate model performance with known targets

**Procedure**:
1. Target sentence displayed for 3 seconds
2. Participant mentally rehearses sentence
3. Sentence removed, participant "speaks" mentally
4. 40 test sentences per condition
5. WER/CER calculated against ground truth

### 4.4 Task 4: Free-Form Generation
**Purpose**: Evaluate real-world usability

**Procedure**:
1. Participant given open-ended prompts
2. E.g., "Describe your morning routine"
3. Generate 5-10 sentence responses
4. 6 prompts total
5. Qualitative analysis of outputs

## 5. Data Collection

### 5.1 EEG Recording
**Equipment**: 
- 32-channel active EEG system (e.g., g.tec g.Nautilus)
- Standard 10-20 electrode placement
- Sampling rate: 500 Hz
- Reference: average reference
- Ground: AFz

**Electrode Sites**: 
Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, POz, O1, O2 (28 sites + Ref + Gnd + EOG)

**Impedance**: <10 kΩ

**Filters**: 
- Hardware: 0.1-100 Hz bandpass
- Software: 0.5-50 Hz, 60 Hz notch

### 5.2 Eye Tracking (Optional)
- SR Research EyeLink 1000 (if available)
- 500 Hz sampling
- For fixation control and artifact detection

### 5.3 Behavioral Measures
- Task completion time
- Self-reported mental effort (NASA-TLX)
- Typing speed (baseline comparison)
- Error awareness (did participant notice errors?)

### 5.4 Physiological Measures
- EEG signal quality metrics
- Alpha/beta power (relaxation vs. concentration)

### 5.5 Subjective Measures

**System Usability Scale (SUS)**
- 10-item questionnaire
- 0-100 score
- Administered Sessions 3 & 4

**Communication Effectiveness**
- Custom 7-point Likert scales:
  - "I could express my thoughts accurately"
  - "The system understood my intentions"
  - "The speed was acceptable for communication"
  - "I felt in control of the system"
  - "I would use this system regularly"

**Cognitive Load (NASA-TLX)**
- Mental demand
- Physical demand
- Temporal demand
- Performance
- Effort
- Frustration

**Semi-Structured Interview** (Session 4)
- What worked well?
- What were the main challenges?
- How might the system be improved?
- Would you use this for daily communication?
- Comparison to other input methods

## 6. Data Analysis

### 6.1 Primary Outcomes
1. **Word Error Rate (WER)**: Primary metric
2. **Character Error Rate (CER)**: Secondary metric
3. **System Usability Score**: Subjective usability

### 6.2 Secondary Outcomes
1. BLEU score
2. Information transfer rate (ITR)
3. Cognitive load scores
4. Learning rate (WER improvement across sessions)
5. Subject adaptation benefit (adapted - baseline)

### 6.3 Statistical Analysis

**Repeated Measures ANOVA**:
- DV: WER
- IV: Model type (subject-specific vs. independent)
- Within-subject factor
- α = 0.05

**Paired t-tests**:
- Session 3 vs. Session 4 performance
- Adapted vs. baseline models

**Correlation Analysis**:
- Cognitive load vs. performance
- Signal quality vs. accuracy

**Qualitative Analysis**:
- Thematic coding of interview transcripts
- Error pattern categorization

### 6.4 Data Quality Control
- EEG signal quality checks (impedance, artifacts)
- Attention checks during tasks
- Exclusion criteria post-hoc:
  - >30% trials with artifacts
  - Participant not following instructions
  - Technical failures

## 7. Safety and Ethics

### 7.1 IRB Approval
- Protocol submitted to Institutional Review Board
- Approval required before participant recruitment
- Annual renewals as needed

### 7.2 Informed Consent
- Written consent obtained before participation
- Explanation of procedures, risks, benefits
- Right to withdraw at any time without penalty
- Data confidentiality and anonymization

### 7.3 Risks and Mitigation

**Minimal Risks**:
1. **Scalp irritation**: Rare, use hypoallergenic gel
2. **Fatigue**: Scheduled breaks, limit session length
3. **Eye strain**: Large text, adequate lighting
4. **Boredom**: Varied tasks, reasonable compensation

**Confidentiality**:
- Data de-identified (participant IDs)
- Secure storage (encrypted)
- Access limited to research team
- Aggregated reporting only

### 7.4 Compensation
- $20/hour (total $100 for 4 sessions + $100 completion bonus)
- Paid after each session
- Prorated if withdrawn early

### 7.5 Participant Rights
- Voluntary participation
- Right to withdraw any time
- Right to skip questions
- Access to own data upon request

## 8. Expected Outcomes

### 8.1 Scientific Contributions
1. Real-world NEST performance benchmarks
2. Subject adaptation effectiveness quantification
3. Usability insights for BCI design
4. Error analysis for system improvements

### 8.2 Publication Plan
- Primary paper: "NEST User Study Results" (NeurIPS/EMBC)
- Secondary: "Usability Factors in EEG-based BCIs" (HCI conference)

### 8.3 Data Sharing
- De-identified EEG data: OpenNeuro repository
- Behavioral data: OSF (Open Science Framework)
- Code: GitHub (already public)
- Results dashboard: Project website

## 9. Study Management

### 9.1 Roles and Responsibilities

**Principal Investigator**:
- Overall study oversight
- Data analysis and publication

**Research Assistant(s)**:
- Participant recruitment
- EEG cap setup and data collection
- Session administration

**Data Manager**:
- Data quality control
- Database management
- Statistical analysis support

### 9.2 Equipment and Resources
- EEG system with 32 channels
- Stimulus presentation computer (PsychoPy/MATLAB)
- Offline analysis workstation (GPU for model training)
- Participant compensation budget: $5,000
- Consumables (EEG gel, disposables): $500

### 9.3 Timeline
- **Month 1**: IRB submission and approval
- **Month 2**: Pilot testing (3 participants)
- **Month 3-4**: Main data collection (20 participants)
- **Month 5**: Data analysis
- **Month 6**: Manuscript preparation

## 10. Quality Assurance

### 10.1 Standard Operating Procedures (SOPs)
- EEG cap setup checklist
- Impedance testing protocol
- Task administration script
- Data backup procedures

### 10.2 Training
- All research assistants trained on:
  - EEG system operation
  - Informed consent procedures
  - Emergency protocols
  - Data collection standards

### 10.3 Monitoring
- Weekly team meetings
- Ongoing data quality checks
- Interim analyses (after 10 participants)

## 11. Contingency Planning

### 11.1 Technical Issues
- Backup EEG system available
- Data recovery procedures in place
- Re-scheduling protocol for technical failures

### 11.2 Recruitment Challenges
- Multiple recruitment channels (campus ads, online, participant pool)
- Flexible scheduling
- Referral bonuses

### 11.3 COVID-19 Considerations
- Sanitization protocols
- Disposable EEG caps if needed
- Masks during setup
- Screening questionnaire

## Appendices

### Appendix A: Informed Consent Form
[See separate document]

### Appendix B: Recruitment Flyer
[See separate document]

### Appendix C: Questionnaires
[See separate documents: SUS, NASA-TLX, Custom scales]

### Appendix D: Task Stimuli
[See stimuli database]

### Appendix E: Data Management Plan
[See separate document]

### Appendix F: Statistical Analysis Plan
[See separate document]

---

**Protocol Version History**:
- v1.0 (2026-02-15): Initial protocol

**Contact Information**:
- PI: [Name, Email, Phone]
- IRB Office: [Contact Info]
