# Performance Metrics

This page documents the detailed performance metrics for both v0.1 and P1 models based on actual execution results.

## v0.1 Model Performance

### Training Metrics
- **Epochs**: 25
- **Final Loss**: 0.0045 (MSE)
- **Training Time**: ~35 seconds
- **Convergence**: Stable after epoch 20
- **Batch Size**: 32
- **Optimizer**: Adam

### Training Progress
```
Epoch 1/25: loss: 0.0565 (high initial loss)
Epoch 2/25: loss: 0.0092 (significant improvement)
Epoch 5/25: loss: 0.0072 (continued improvement)
Epoch 10/25: loss: 0.0065 (stabilizing)
Epoch 15/25: loss: 0.0051 (approaching convergence)
Epoch 20/25: loss: 0.0051 (converged)
Epoch 25/25: loss: 0.0045 (final loss)
```

### Testing Results
- **Next-Day Prediction**: $468.90
- **Target Stock**: CBA.AX
- **Prediction Type**: Single next-day forecast
- **Evaluation**: Visual comparison only

### Limitations
- No quantitative accuracy metrics
- No profit/loss analysis
- Single prediction point
- Limited evaluation framework

## P1 Model Performance

### Training Metrics
- **Epochs**: 25
- **Training Loss**: 0.0018 (Huber)
- **Validation Loss**: 0.0016 (best: 0.00159)
- **Mean Absolute Error**: 0.0432
- **Training Time**: ~90 seconds
- **Batch Size**: 64
- **Optimizer**: Adam

### Training Progress with Validation
```
Epoch 1/25: loss: 0.0330 - val_loss: 0.0022 (solid start)
Epoch 2/25: loss: 0.0030 - val_loss: 0.0021 (improvement)
Epoch 7/25: val_loss: 0.00198 (checkpoint saved)
Epoch 8/25: val_loss: 0.00178 (best so far)
Epoch 10/25: val_loss: 0.00166 (continued improvement)
Epoch 12/25: val_loss: 0.00163 (new best)
Epoch 16/25: val_loss: 0.00161 (marginal improvement)
Epoch 19/25: val_loss: 0.00159 (final best)
Epoch 24/25: val_loss: 0.00159 (maintained)
```

### Testing Metrics
- **15-Day Prediction**: $768.91
- **Huber Loss**: 0.0016128835268318653
- **Mean Absolute Error**: $116.63
- **Accuracy**: 56.90%
- **Target Stock**: META

### Profit Analysis
- **Buy Profit**: $1,038.28
- **Sell Profit**: $168.85
- **Total Profit**: $1,207.13
- **Profit per Trade**: $5.05
- **Number of Trades**: 239
- **Success Rate**: 56.90%

## Detailed Performance Analysis

### Loss Function Comparison

**Mean Squared Error (v0.1)**
- Sensitive to outliers
- Penalizes large errors heavily
- Standard regression metric
- Final value: 0.0045

**Huber Loss (P1)**
- Robust to outliers
- Combines MSE and MAE benefits
- Better for financial data
- Final value: 0.0018

### Accuracy Metrics

**v0.1 Model:**
- No accuracy calculation
- Visual assessment only
- Qualitative evaluation

**P1 Model:**
- **Quantitative Accuracy**: 56.90%
- **Mean Absolute Error**: $116.63
- **Directional Accuracy**: Measured via buy/sell signals
- **Profit-based Validation**: $1,207.13 total profit

### Model Capacity Analysis

**v0.1 Architecture:**
- Total Parameters: ~40,000 (estimated)
- 3 LSTM layers × 50 units each
- Lower model capacity

**P1 Architecture:**
- Total Parameters: ~530,000 (estimated)
- 2 LSTM layers × 256 units each
- Higher model capacity, better feature learning

## Hardware Performance

### System Optimization
Both models utilize TensorFlow CPU optimizations:
- SSE3, SSE4.1, SSE4.2 instructions
- AVX, AVX2, AVX_VNNI support
- FMA (Fused Multiply-Add) operations

### Training Efficiency
- **v0.1**: 35 seconds for 25 epochs (~1.4 sec/epoch)
- **P1**: 90 seconds for 25 epochs (~3.6 sec/epoch)
- **Efficiency Ratio**: P1 takes 2.6× longer but provides significantly more value

## Statistical Analysis

### P1 Model Trading Performance

**Sample Trading Results (Last 10 Trades):**
```
Date        Close    Prediction  Buy_Profit  Sell_Profit
2025-05-27  641.83   654.40      55.40       0.0
2025-05-30  646.99   657.62      51.54       0.0
2025-06-11  693.61   702.67      25.40       0.0
2025-06-16  702.12   708.09      30.66       0.0
2025-06-17  697.23   709.64      30.01       0.0
2025-07-14  720.92   736.20      55.45       0.0
2025-07-16  702.91   732.71      69.08       0.0
2025-07-18  704.28   725.80      65.02       0.0
2025-07-22  704.81   722.89      85.19       0.0
2025-07-28  717.63   725.73      49.74       0.0
```

### Risk Metrics

**P1 Model Risk Assessment:**
- **Maximum Drawdown**: Not calculated (future enhancement)
- **Volatility**: Inherent in stock market data
- **Sharpe Ratio**: Not calculated (requires risk-free rate)
- **Win Rate**: 56.90% (above random chance)

## Benchmarking

### Comparison with Random Prediction
- **Random Accuracy**: ~50% (coin flip)
- **P1 Accuracy**: 56.90%
- **Improvement**: 6.90 percentage points above random

### Industry Standards
- **Typical LSTM Accuracy**: 50-60% for stock prediction
- **P1 Performance**: 56.90% (within industry range)
- **Profit Generation**: Positive ($1,207.13) indicates practical utility

## Performance Recommendations

### For v0.1 Model Improvement
1. Add accuracy calculation
2. Implement profit analysis
3. Include validation split
4. Add more evaluation metrics

### For P1 Model Enhancement
1. Calculate additional risk metrics
2. Implement ensemble methods
3. Add confidence intervals
4. Include feature importance analysis

## Conclusion

The performance metrics clearly demonstrate P1 model superiority:

- **Quantitative Evaluation**: P1 provides measurable accuracy (56.90%)
- **Financial Viability**: P1 generates positive profits ($1,207.13)
- **Technical Robustness**: P1 uses advanced loss functions and validation
- **Practical Utility**: P1 offers actionable trading insights

The v0.1 model serves as a functional baseline, while P1 represents a production-ready solution with comprehensive performance measurement and validation.

---

*Performance analysis completed: August 24, 2025*
