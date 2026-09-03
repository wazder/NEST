# 🚀 NEST Quick Start Guide

**Welcome to NEST!** This guide gets you from zero to trained models in 4 simple steps.

## ⚡ Step 1: Install Dependencies (5 minutes)

```bash
# Navigate to your NEST project directory
cd NEST

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install NEST in development mode
pip install -e .
```

**Verify installation:**
```bash
python test_phase2.py
```

You should see: ✓ All Phase 2 modules tested successfully!

---

## 📥 Step 2: Download ZuCo Dataset (30-60 minutes)

```bash
# Download all ZuCo datasets (~18GB)
python scripts/download_zuco.py --output data/raw/zuco --tasks all

# Or download just one task for testing
python scripts/download_zuco.py --tasks task1_SR
```

**Verify download:**
```bash
python scripts/download_zuco.py --verify-only
```

**Alternative:** Manual download from https://osf.io/q3zws/

---

## 🎯 Step 3: Train Models (2-3 days)

### Option A: Full Training (All Models)

```bash
# Train all model variants
python scripts/train_zuco_full.py \
    --config configs/model.yaml \
    --output results/ \
    --epochs 100

# This will train:
#   - NEST-CTC (baseline)
#   - NEST-RNN-T (streaming)
#   - NEST-Transformer (attention)
#   - NEST-Conformer (best performance)
```

### Option B: Quick Test (1 Model, Few Epochs)

```bash
# Train just Conformer for 10 epochs (for testing)
python scripts/train_zuco_full.py \
    --config configs/model.yaml \
    --output results/test/ \
    --models conformer \
    --epochs 10
```

### Monitor Training

```bash
# Watch training progress
tail -f results/training.log

# Or use TensorBoard
tensorboard --logdir results/logs/
# Open http://localhost:6006
```

---

## ✅ Step 4: Verify Results (10 minutes)

```bash
# Verify results match paper expectations
python scripts/verify_results.py --results results/results.json

# Generate publication figures
python scripts/generate_figures.py --results results/ --output papers/figures/
```

---

## 🎓 Expected Results

**No validated WER/CER/BLEU numbers exist for any NEST model yet.** The
table below is a placeholder showing the format results will be reported
in once a full training and evaluation run completes — not values you
should expect to reproduce. See [docs/STATUS.md](../STATUS.md).

| Model | WER | CER | BLEU | Status |
|-------|-----|-----|------|--------|
| NEST-Conformer | pending | pending | pending | not yet trained |
| NEST-Transformer | pending | pending | pending | not yet trained |
| NEST-RNN-T | pending | pending | pending | not yet trained |
| NEST-CTC | pending | pending | pending | not yet trained |

---

## 🛠️ Troubleshooting

### Problem: Out of GPU Memory

**Solution:**
```bash
# Reduce batch size in configs/model.yaml
# Change: batch_size: 16 → batch_size: 8
```

### Problem: Download Fails

**Solutions:**
1. Try manual download: https://osf.io/q3zws/
2. Use `wget` with resume: `wget -c <url>`
3. Check internet connection and firewall

### Problem: Training Too Slow

**Solutions:**
1. Use smaller model: `--models ctc` (fastest)
2. Reduce epochs: `--epochs 50`
3. Use mixed precision: Already enabled in config
4. Consider cloud GPU (Google Colab, Lambda Labs)

### Problem: Results Don't Match Paper

**Check:**
1. Hyperparameters match `configs/model.yaml`
2. Using same preprocessing (check configs)
3. Subject-independent split is correct
4. Training converged (check loss curves)

---

## 📊 What Gets Created

After completing all steps, you'll have:

```
results/
├── checkpoints/
│   ├── nest_conformer/
│   │   ├── best_model.pt          # Best model weights
│   │   ├── checkpoint_epoch_*.pt  # Training checkpoints
│   │   └── history.json           # Training curves
│   ├── nest_transformer/
│   ├── nest_rnn_t/
│   └── nest_ctc/
├── logs/                           # TensorBoard logs
├── results.json                    # All metrics
├── verification_report.md          # Comparison with paper
└── RESULTS_REPORT.md              # Human-readable summary

papers/
└── figures/
    ├── figure2_model_comparison.pdf
    ├── figure3_training_curves.pdf
    └── ...
```

---

## 🎯 Next Steps After Training

### For Research/Publication:

1. **Run User Study:**
   ```bash
   python experiments/user_study/run_user_study.py --participant P001 --session 1
   ```

2. **Finalize Paper:**
   - Update `papers/NEST_manuscript.md` with real results
   - Include generated figures
   - Convert to PDF: `pandoc papers/NEST_manuscript.md -o papers/NEST.pdf`

3. **Submit to Conference:**
   - See `papers/SUBMISSION_CHECKLIST.md`
   - Deadlines: IEEE EMBC (Mar 15), NeurIPS (May 22), EMNLP (Jun 15)

### For Deployment:

1. **Optimize Model:**
   ```bash
   python scripts/optimize_for_deployment.py \
       --model results/checkpoints/nest_conformer/best_model.pt \
       --output models/optimized/
   ```

2. **Test Inference:**
   ```bash
   python examples/04_deployment.py \
       --checkpoint models/optimized/nest_conformer_optimized.pt
   ```

3. **Build Application:**
   - See `docs/DEPLOYMENT.md` for integration guide

---

## 💡 Tips

- **Save Time:** Start with quick test (Option B) before full training
- **Save Money:** Use spot instances on cloud (Vast.ai, Lambda Labs)
- **Monitor:** Always check TensorBoard to ensure training is progressing
- **Backup:** Models are large - backup checkpoints regularly
- **Reproduce:** Use fixed seeds (already set) for reproducibility

---

## 📚 Documentation

- **Training Guide:** `docs/TRAINING_GUIDE.md`
- **API Reference:** `docs/API.md`
- **User Study:** `experiments/user_study/user_study_protocol.md`
- **Paper:** `papers/NEST_manuscript.md`

---

## 🆘 Need Help?

1. Check `docs/TRAINING_GUIDE.md` for detailed troubleshooting
2. Open an issue: https://github.com/wazder/NEST/issues
3. See examples in `examples/` directory

---

## ⏱️ Time Estimates

| Task | Duration | Can Run Overnight? |
|------|----------|-------------------|
| Install | 5 min | No |
| Download Data | 30-60 min | Yes |
| Full Training | 2-3 days | **Yes** |
| Verification | 10 min | No |
| User Study | 2-4 weeks | No |

**Total to Paper Submission:** ~3-4 weeks

---

## 🎉 Success Criteria

You're ready for paper submission when:

- ✅ All models trained successfully
- ✅ WER/CER/BLEU measured against reference text (whatever the values are)
- ✅ Verification report shows "PASS"
- ✅ Figures generated
- ✅ Results are logged from an actual run, not copied from this guide

---

**Ready? Start with Step 1!** 🚀

Questions? See `docs/` or open an issue on GitHub.
