# Training Workflow

This page documents the complete training workflow for both v0.1 and P1 models, including step-by-step procedures and expected outputs.

## Prerequisites

Before starting training, ensure:
- ✅ Virtual environment is activated
- ✅ All dependencies are installed
- ✅ Internet connection available (for data download)
- ✅ Sufficient disk space (~500MB for outputs)

## v0.1 Model Training Workflow

### Step 1: Environment Preparation
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Verify environment
python --version  # Should show Python 3.12
pip list | grep tensorflow  # Verify TensorFlow installation
```

### Step 2: Execute Training
```bash
# Run the v0.1 model
python v0.1.py
```

### Step 3: Monitor Training Progress
Expected console output:
```
Loading data for CBA.AX...
[*********************100%***********************]  1 of 1 completed

2025-08-24 17:57:32: TensorFlow optimizations loaded
Training started...

Epoch 1/25
27/27 ━━━━━━━━━━━━━━━━━━━━ 2s 17ms/step - loss: 0.0565
Epoch 2/25  
27/27 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0092
...
Epoch 25/25
27/27 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 0.0045

Testing phase...
[*********************100%***********************]  1 of 1 completed
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
Prediction: [[468.90]]
```

### Step 4: Verify Results
- Training completes in ~35 seconds
- Final prediction displayed
- Matplotlib visualization window opens
- No files saved (limitation of v0.1)

## P1 Model Training Workflow

### Step 1: Environment Preparation
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Verify P1 files exist
ls train_p1.py test_p1.py p1.py parameters.py
```

### Step 2: Execute Training
```bash
# Train the P1 model
python train_p1.py
```

### Step 3: Monitor Training Progress
Expected console output:
```
Loading data for META...
[*********************100%***********************]  1 of 1 completed

2025-08-24 17:58:51: TensorFlow optimizations loaded
Model architecture initialized...

Epoch 1/25
15/15 ━━━━━━━━━━━━━━━━━━━━ 3s 112ms/step - loss: 0.0330 - val_loss: 0.0022
Epoch 1: val_loss improved from None to 0.00217, saving model to p1/results/...

Epoch 2/25
15/15 ━━━━━━━━━━━━━━━━━━━━ 2s 101ms/step - loss: 0.0030 - val_loss: 0.0021
Epoch 2: val_loss improved from 0.00217 to 0.00215, saving model to p1/results/...

...

Epoch 19/25
15/15 ━━━━━━━━━━━━━━━━━━━━ 1s 93ms/step - loss: 0.0020 - val_loss: 0.0016
Epoch 19: val_loss improved from 0.00161 to 0.00159, saving model to p1/results/...

...

Epoch 25/25
15/15 ━━━━━━━━━━━━━━━━━━━━ 1s 89ms/step - loss: 0.0018 - val_loss: 0.0016

Model saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256.keras
Data saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256_data.pkl
```

### Step 4: Verify Training Outputs
Check created directories and files:
```bash
# Verify directory structure
ls p1/
# Expected: data/ logs/ results/ model_checkpoints/ csv-results/

# Check model files
ls p1/model_checkpoints/
# Expected: .keras model file and .pkl data file

# Check training logs
ls p1/logs/
# Expected: TensorBoard log files
```

## P1 Model Testing Workflow

### Step 1: Execute Testing
```bash
# Test the trained P1 model
python test_p1.py
```

### Step 2: Monitor Testing Progress
Expected console output:
```
Loading saved model and data for 2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256...
Data loaded from: p1/model_checkpoints/...data.pkl
Model loaded from: p1/model_checkpoints/...keras

8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Future price after 15 days is 768.91$
huber loss: 0.0016128835268318653
Mean Absolute Error: 116.62753188573774
Accuracy score: 0.5690376569037657
Total buy profit: 1038.2820281982422
Total sell profit: 168.84672546386716
Total profit: 1207.1287536621094
Profit per trade: 5.050747923272424
```

### Step 3: Verify Testing Outputs
```bash
# Check CSV results
ls p1/csv-results/
# Expected: Results CSV file with detailed predictions

# Verify visualization files (if generated)
ls *.png *.jpg
# Expected: Prediction plots and analysis charts
```

## Workflow Troubleshooting

### Common Training Issues

**Issue**: Data download fails
```bash
# Solution: Check internet connection and retry
# Alternative: Use different time period or stock symbol
```

**Issue**: CUDA/GPU warnings
```bash
# Expected behavior: Model runs on CPU with optimizations
# No action needed - warnings are informational
```

**Issue**: Memory errors during training
```bash
# Solution: Reduce batch size in parameters.py
# Edit: BATCH_SIZE = 32  # Reduce from 64
```

**Issue**: Model checkpoint saving fails
```bash
# Solution: Ensure directory permissions
mkdir -p p1/model_checkpoints
chmod 755 p1/model_checkpoints
```

### Validation Steps

After each training session:

1. **Check Loss Convergence**
   - Training loss should decrease over epochs
   - Validation loss should not increase significantly (overfitting check)

2. **Verify File Outputs**
   - Model files saved in correct directories
   - Data files properly serialized
   - Log files generated for TensorBoard

3. **Test Model Loading**
   - Run test_p1.py to ensure model loads correctly
   - Verify predictions are generated
   - Check metrics calculation

## Performance Optimization

### Training Speed Optimization
```python
# In parameters.py, adjust these for faster training:
BATCH_SIZE = 128  # Increase batch size (if memory allows)
EPOCHS = 20       # Reduce epochs for faster iteration
```

### Memory Optimization
```python
# For memory-constrained systems:
BATCH_SIZE = 32   # Reduce batch size
UNITS = 128       # Reduce LSTM units
```

### Accuracy Optimization
```python
# For better accuracy (slower training):
EPOCHS = 50       # More training epochs
DROPOUT = 0.3     # Adjust dropout rate
SEQUENCE_LENGTH = 60  # Longer sequences
```

## Workflow Best Practices

### Before Training
1. ✅ Backup previous model files
2. ✅ Clear old log files if needed
3. ✅ Verify data connectivity
4. ✅ Check available disk space

### During Training
1. ✅ Monitor loss convergence
2. ✅ Watch for overfitting signs
3. ✅ Check system resource usage
4. ✅ Verify checkpoint saving

### After Training
1. ✅ Test model immediately
2. ✅ Document training parameters
3. ✅ Save training logs
4. ✅ Backup model files

## Next Steps

After successful training:
1. [Test Model Performance](Testing-Results.md)
2. [Analyze Results](Performance-Metrics.md)
3. [Compare Models](Model-Comparison.md)
4. [Review Insights](Insights-Recommendations.md)

---

*Training workflow documented: August 24, 2025*
