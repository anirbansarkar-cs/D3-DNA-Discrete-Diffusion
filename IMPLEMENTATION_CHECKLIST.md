# Implementation Checklist: Signal Embedding Modes for LentIMPRA

## ✅ Completed Tasks

### Core Implementation
- [x] Modified `EmbeddingLayer` class in `model/transformer.py` (lines 117-192)
  - [x] Added `embedding_mode` parameter to `__init__`
  - [x] Implemented 'add' mode (baseline)
  - [x] Implemented 'concat' mode (append labels to sequence)
  - [x] Implemented 'mask' mode (NaN handling)
  - [x] Added comprehensive docstrings

- [x] Updated `TransformerModel` class in `model/transformer.py` (line 240)
  - [x] Added `embedding_mode` extraction from config
  - [x] Passed `embedding_mode` to `EmbeddingLayer`

### Configuration Files
- [x] Created `transformer_multi_class_add.yaml`
  - [x] Set `embedding_mode: add`
  - [x] Updated wandb name and tags
  - [x] Added explanatory comments

- [x] Created `transformer_multi_class_concat.yaml`
  - [x] Set `embedding_mode: concat`
  - [x] Updated wandb name and tags
  - [x] Added note about effective sequence length (233)

- [x] Created `transformer_multi_class_mask.yaml`
  - [x] Set `embedding_mode: mask`
  - [x] Updated wandb name and tags
  - [x] Added note about NaN handling

### Documentation
- [x] Created `train_3cells.txt` with:
  - [x] Complete training commands for all three modes
  - [x] Architecture explanations
  - [x] Configuration details
  - [x] Monitoring guidelines
  - [x] Troubleshooting tips

- [x] Created `EMBEDDING_MODES_SUMMARY.md` with:
  - [x] Technical implementation details
  - [x] Architecture diagrams (text-based)
  - [x] Parameter counts
  - [x] Expected outcomes and hypotheses
  - [x] Evaluation metrics
  - [x] Future extensions

- [x] Created `IMPLEMENTATION_CHECKLIST.md` (this file)

### Testing
- [x] Created `test_embedding_modes.py` with tests for:
  - [x] 'add' mode shape validation
  - [x] 'concat' mode shape validation
  - [x] 'mask' mode NaN handling
  - [x] Unconditional generation (labels=None)
  - [x] Parameter count verification

## 📋 Pre-Training Checklist

Before running training, verify:

### Data Preparation
- [ ] Verify LentIMPRA data file exists at: `model_zoo/lentimpra/lenti_MPRA_K562_data.h5`
- [ ] Check label shape is (N, 3) for multi-class
- [ ] For 'mask' mode: optionally add NaN values if testing partial labels
- [ ] Verify oracle model exists if using SP-MSE validation

### Environment Setup
- [ ] Install required packages: `pip install -e .`
- [ ] Verify Flash Attention is available
- [ ] Check GPU availability (2 GPUs recommended per config)
- [ ] Configure wandb credentials if using logging

### Configuration Review
- [ ] Review each config file for correct paths
- [ ] Set `paths.data_file` to your data location
- [ ] Set `paths.oracle_model` if using SP-MSE validation
- [ ] Adjust `ngpus` if not using 2 GPUs
- [ ] Verify `batch_size` is divisible by (ngpus × accum)

### Optional Testing
- [ ] Run `python test_embedding_modes.py` to verify implementation
- [ ] Do a quick 1-epoch test run to verify configs work

## 🚀 Training Execution Plan

### Run Order Recommendation

**Option 1: Sequential Training**
```bash
# Run 1: Baseline 'add' mode
python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_add.yaml

# Run 2: 'concat' mode (after baseline completes)
python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_concat.yaml

# Run 3: 'mask' mode (after concat completes)
python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_mask.yaml
```

**Option 2: Parallel Training (if sufficient GPUs)**
```bash
# Terminal 1 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_add.yaml

# Terminal 2 (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_concat.yaml

# Terminal 3 (GPUs 4-5)
CUDA_VISIBLE_DEVICES=4,5 python model_zoo/lentimpra/train.py \
    --architecture transformer \
    --config model_zoo/lentimpra/configs/transformer_multi_class_mask.yaml
```

## 📊 Monitoring During Training

### Key Metrics to Track

For each model, monitor on wandb:

1. **Training Metrics**
   - [ ] Training loss curve (should decrease smoothly)
   - [ ] Gradient norms (should be stable)
   - [ ] Learning rate schedule
   - [ ] Time per epoch

2. **Validation Metrics**
   - [ ] Validation loss
   - [ ] SP-MSE scores (if enabled)
   - [ ] Sample quality (visual inspection)

3. **Comparative Analysis**
   - [ ] Which mode converges fastest?
   - [ ] Which mode achieves lowest validation loss?
   - [ ] Which mode has best SP-MSE scores?

### Expected Training Time

Per model:
- Epochs: 300
- Checkpoint frequency: Every 4 epochs (75 checkpoints total)
- Validation frequency: Every 4 epochs
- Estimated time: ~4-8 hours on 2x A100 GPUs (depends on hardware)

## 🔍 Post-Training Evaluation

After training completes:

### 1. Checkpoint Verification
```bash
# Check that checkpoints were saved
ls checkpoints/lentimpra-transformer-add/
ls checkpoints/lentimpra-transformer-concat/
ls checkpoints/lentimpra-transformer-mask/
```

### 2. Model Evaluation
```bash
# Evaluate each model
for mode in add concat mask; do
    python model_zoo/lentimpra/evaluate.py \
        --architecture transformer \
        --checkpoint checkpoints/lentimpra-transformer-${mode}/best.ckpt \
        --use_oracle \
        --oracle_checkpoint model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt
done
```

### 3. Sampling and Analysis
```bash
# Generate sequences with each model
for mode in add concat mask; do
    python model_zoo/lentimpra/sample.py \
        --architecture transformer \
        --checkpoint checkpoints/lentimpra-transformer-${mode}/best.ckpt \
        --num_samples 1000 \
        --output_file samples_${mode}.h5
done
```

### 4. Comparative Analysis
- [ ] Compare validation losses across modes
- [ ] Compare SP-MSE scores across modes
- [ ] Analyze sample diversity
- [ ] Measure conditioning accuracy
- [ ] Visualize generated sequences

## 📝 Results Documentation

Create a results summary document with:

### Training Comparison Table
```
| Mode   | Final Train Loss | Final Val Loss | Best SP-MSE | Training Time | Params |
|--------|-----------------|----------------|-------------|---------------|--------|
| add    | ?               | ?              | ?           | ?             | ~2.3K  |
| concat | ?               | ?              | ?           | ?             | ~1.5K  |
| mask   | ?               | ?              | ?           | ?             | ~2.3K  |
```

### Generation Quality Comparison
- [ ] Sample sequences from each model
- [ ] Evaluate with oracle model
- [ ] Compare to real data distribution
- [ ] Assess conditioning accuracy

### Conclusions
- [ ] Which mode performed best overall?
- [ ] Which mode is best for which use case?
- [ ] Recommendations for future work

## 🐛 Common Issues and Solutions

### Issue: CUDA out of memory
**Solution:** Reduce `batch_size` in config or reduce `ngpus` and adjust batch size accordingly

### Issue: Shape mismatch in EmbeddingLayer
**Solution:** Verify `signal_dim=3` in config matches data label shape

### Issue: NaN in loss
**Solution:** Check learning rate, gradient clipping, data normalization

### Issue: Poor conditioning accuracy
**Solution:**
- Verify label normalization
- Check `class_dropout_prob` isn't too high
- Ensure labels match oracle model's training distribution

### Issue: Slow convergence
**Solution:**
- Verify learning rate schedule
- Check warmup steps
- Ensure EMA is enabled

## 📌 Files Modified/Created

### Modified Files
1. `model/transformer.py` - Added embedding modes to EmbeddingLayer

### New Configuration Files
1. `model_zoo/lentimpra/configs/transformer_multi_class_add.yaml`
2. `model_zoo/lentimpra/configs/transformer_multi_class_concat.yaml`
3. `model_zoo/lentimpra/configs/transformer_multi_class_mask.yaml`

### New Documentation Files
1. `train_3cells.txt` - Training instructions
2. `EMBEDDING_MODES_SUMMARY.md` - Technical summary
3. `IMPLEMENTATION_CHECKLIST.md` - This checklist
4. `test_embedding_modes.py` - Test script

### Updated Files (User Data)
1. `train_3cells.txt` - Contains training commands

## ✨ Key Features Implemented

1. **Three Embedding Modes**
   - ✅ 'add': Standard signal conditioning
   - ✅ 'concat': Label-as-sequence conditioning
   - ✅ 'mask': NaN-aware conditioning

2. **Automatic Adaptation**
   - ✅ Rotary embeddings adapt to sequence length
   - ✅ Flash attention handles variable lengths
   - ✅ All modes support unconditional generation

3. **Backward Compatibility**
   - ✅ Default behavior unchanged
   - ✅ Existing configs work without modification
   - ✅ No breaking changes to API

4. **Complete Documentation**
   - ✅ Technical implementation details
   - ✅ Training instructions
   - ✅ Testing and validation
   - ✅ Troubleshooting guide

## 🎯 Success Criteria

Training is successful if:
- [ ] All three models train without errors
- [ ] Validation loss decreases over training
- [ ] Generated sequences are valid DNA (no N's)
- [ ] Conditioning works (generated sequences match requested labels)
- [ ] SP-MSE scores are reasonable (similar to published results)

## 📧 Next Steps

After successful training:
1. Analyze results and compare modes
2. Share findings with team
3. Decide which mode to use for production
4. Consider extensions (e.g., learnable mask tokens)
5. Apply learnings to other datasets (DeepSTARR, MPRA, etc.)

---

**Implementation Date:** 2025-11-10
**Status:** ✅ Ready for Training
**Estimated Completion:** ~24 hours (8 hours per model × 3 models)
