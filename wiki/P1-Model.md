# P1 Model Documentation

The P1 model represents the enhanced implementation with advanced features, comprehensive metrics, and superior performance.

## Model Overview

- **Files**: `train_p1.py`, `test_p1.py`, `p1.py`, `parameters.py`
- **Target Stock**: META (Meta Platforms Inc.)
- **Prediction Type**: 15-day ahead price prediction
- **Architecture**: 2-layer LSTM with high capacity
- **Data Source**: yfinance (5 years of data)

## Key Features

### Data Configuration
- **Training Data**: 5 years of historical data
- **Features Used**: 5 features (adjusted close, volume, open, high, low)
- **Preprocessing**: MinMaxScaler applied to each feature independently
- **Validation Split**: 20% for model validation

### Model Architecture

```python
# LSTM Layer 1: 256 units, return_sequences=True
# Dropout: 0.4
# LSTM Layer 2: 256 units
# Dropout: 0.4  
# Dense Output: 1 unit, linear activation
```

### Training Parameters
- **Sequence Length**: 50 days lookback
- **Prediction Horizon**: 15 days ahead
- **Batch Size**: 64
- **Epochs**: 25
- **Optimizer**: Adam
- **Loss Function**: Huber Loss
- **Validation Split**: 20%

## Execution Instructions

### Training the Model

```bash
# Ensure virtual environment is activated
.\venv\Scripts\activate

# Train the P1 model
python train_p1.py
```

### Testing the Model

```bash
# Test the trained P1 model
python test_p1.py
```

## Training Results

### Training Output
```
Loading data for META...
[*********************100%***********************]  1 of 1 completed

Epoch 1/25: loss: 0.0330 - val_loss: 0.0022 (solid initial performance)
Epoch 7/25: val_loss: 0.00198 (checkpoint saved)
Epoch 10/25: val_loss: 0.00166 (further improvement)
...
Epoch 19/25: val_loss: 0.00159 (best performance)

Model saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256.keras
Data saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256_data.pkl
```

### Training Metrics
- **Epochs**: 25
- **Training Loss**: 0.0018 (Huber)
- **Validation Loss**: 0.0016 (best: 0.00159)
- **Mean Absolute Error**: 0.0432
- **Training Time**: ~90 seconds

## Testing Results

### Performance Metrics
- **15-Day Prediction**: $768.91
- **Huber Loss**: 0.0016128835268318653
- **Mean Absolute Error**: $116.63
- **Accuracy**: 56.90%

### Profit Analysis
- **Buy Profit**: $1,038.28
- **Sell Profit**: $168.85
- **Total Profit**: $1,207.13
- **Profit per Trade**: $5.05
- **Number of Trades**: 239

## Advanced Features

### Model Persistence
```python
# Full model saving in .keras format
model_checkpoint_path = os.path.join("p1/model_checkpoints", model_name + ".keras")
save_model(model, model_checkpoint_path)

# Data persistence for inference
data_path = os.path.join("p1/model_checkpoints", model_name + "_data.pkl")
joblib.dump(data, data_path)
```

### Directory Structure
```
p1/
├── data/              # Raw data storage
├── logs/              # TensorBoard logs
├── results/           # Model weights (.h5)
├── model_checkpoints/ # Full models (.keras) and data (.pkl)
└── csv-results/       # Evaluation outputs
```

### Comprehensive Metrics
- Huber loss calculation
- Mean Absolute Error (MAE)
- Accuracy percentage
- Detailed profit analysis
- Buy/sell signal evaluation
- CSV export of results

## Model Improvements Over v0.1

### Technical Enhancements
1. **Multi-feature Input**: 5 features vs 1 feature
2. **Advanced Architecture**: 2×256 units vs 3×50 units
3. **Modern Loss Function**: Huber vs MSE
4. **Model Persistence**: Full model + data vs weights only
5. **Comprehensive Metrics**: Profit analysis vs basic prediction

### Code Quality Improvements
1. **Modular Design**: Separate training/testing scripts
2. **Parameter Management**: Centralized configuration
3. **Error Handling**: Robust data processing
4. **Logging**: TensorBoard integration
5. **Documentation**: Comprehensive code comments

## Configuration Parameters

Key parameters in `parameters.py`:
```python
SEQUENCE_LENGTH = 50
STEP_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 25
LAYERS = 2
UNITS = 256
DROPOUT = 0.4
LOSS = "huber"
OPTIMIZER = "adam"
```

## Troubleshooting

### Common Issues

**Issue**: Model loading fails in test_p1.py
```bash
# Solution: Ensure training completed successfully
python train_p1.py  # Re-run training if needed
```

**Issue**: CUDA/GPU errors
```bash
# Solution: Model runs on CPU by default
# No action needed - CPU optimization is automatic
```

**Issue**: Memory issues during training
```bash
# Solution: Reduce batch size in parameters.py
BATCH_SIZE = 32  # Reduce from 64
```

## Performance Comparison

| Metric | v0.1 Model | P1 Model | Winner |
|--------|------------|----------|---------|
| Accuracy | N/A | 56.90% | P1 |
| Profit | None | $1,207.13 | P1 |
| Features | 1 | 5 | P1 |
| Architecture | 3×50 | 2×256 | P1 |

## Next Steps

After testing P1 model:
1. [Compare with v0.1](Model-Comparison.md)
2. [Review Performance Metrics](Performance-Metrics.md)
3. [Analyze Results](Testing-Results.md)

---

*Model tested successfully on: August 24, 2025*
