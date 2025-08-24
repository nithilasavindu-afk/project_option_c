# Testing Results

This page documents the comprehensive testing results for both v0.1 and P1 models, including screenshots and detailed analysis.

## v0.1 Model Testing Results

### Execution Summary
- **Command**: `python v0.1.py`
- **Execution Time**: ~35 seconds total
- **Status**: ✅ Successful completion
- **Output**: Single prediction value

### Test Results
```
Testing Phase Output:
[*********************100%***********************]  1 of 1 completed
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step 
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
Prediction: [[468.90]]
```

### Key Findings
- **Next-Day Prediction**: $468.90 for CBA.AX
- **Model Performance**: Stable prediction generation
- **Visualization**: Matplotlib chart showing actual vs predicted prices
- **Limitations**: No quantitative evaluation metrics

### Screenshots Needed
*Note: Add screenshots of the following:*
1. Console output during execution
2. Matplotlib visualization window
3. Training progress display

## P1 Model Testing Results

### Execution Summary
- **Training Command**: `python train_p1.py`
- **Testing Command**: `python test_p1.py`
- **Total Execution Time**: ~90 seconds training + ~10 seconds testing
- **Status**: ✅ Successful completion with comprehensive metrics

### Training Results
```
Training Completion Output:
Epoch 25/25
15/15 ━━━━━━━━━━━━━━━━━━━━ 1s 89ms/step - loss: 0.0018 - val_loss: 0.0016

Model saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256.keras
Data saved to: p1/model_checkpoints/2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256_data.pkl
```

### Testing Results
```
Testing Phase Output:
Loading saved model and data for 2025-08-24_META-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-50-step-15-layers-2-units-256...
Data loaded from p1/model_checkpoints/...data.pkl
Model loaded from p1/model_checkpoints/...keras

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

### Detailed Performance Metrics

#### Prediction Accuracy
- **15-Day Prediction**: $768.91 for META
- **Accuracy Score**: 56.90%
- **Mean Absolute Error**: $116.63
- **Huber Loss**: 0.00161

#### Financial Performance
- **Total Profit**: $1,207.13
- **Buy Profit**: $1,038.28
- **Sell Profit**: $168.85
- **Profit per Trade**: $5.05
- **Number of Trades**: 239

#### Sample Trading Results
```
Date        Close    High     Low      Open     Volume    Prediction  Buy_Profit  Sell_Profit
2025-05-27  641.83   642.59   632.26   634.92   9508400   654.40      55.40       0.0
2025-05-30  646.99   648.91   632.29   642.01   16241000  657.62      51.54       0.0
2025-06-11  693.61   708.32   691.46   703.17   9582500   702.67      25.40       0.0
2025-06-16  702.12   707.15   693.51   699.33   13720300  708.09      30.66       0.0
2025-06-17  697.23   705.97   696.06   702.00   10066100  709.64      30.01       0.0
2025-07-14  720.92   728.00   716.55   717.60   8939400   736.20      55.45       0.0
2025-07-16  702.91   713.97   699.27   713.37   13067600  732.71      69.08       0.0
2025-07-18  704.28   704.71   691.65   702.19   12779800  725.80      65.02       0.0
2025-07-22  704.81   716.60   701.41   716.19   8921100   722.89      85.19       0.0
2025-07-28  717.63   724.74   712.68   715.20   8715700   725.73      49.74       0.0
```

## File Outputs Generated

### P1 Model File Structure
```
p1/
├── data/
│   └── META_data.csv (raw stock data)
├── logs/
│   └── fit/ (TensorBoard logs)
├── results/
│   └── 2025-08-24_META-...weights.h5 (model weights)
├── model_checkpoints/
│   ├── 2025-08-24_META-...keras (full model)
│   └── 2025-08-24_META-...data.pkl (preprocessed data)
└── csv-results/
    └── predictions_results.csv (detailed results)
```

### Generated Files Analysis
1. **Model Files**: Successfully saved in .keras format
2. **Data Files**: Preprocessed data serialized with joblib
3. **Results CSV**: Comprehensive trading analysis
4. **TensorBoard Logs**: Training metrics for visualization

## Comparative Analysis

### Execution Reliability
| Aspect | v0.1 Model | P1 Model |
|--------|------------|----------|
| **Success Rate** | 100% | 100% |
| **Error Handling** | Basic | Robust |
| **Output Consistency** | Variable | Stable |
| **File Generation** | None | Comprehensive |

### Performance Validation
| Metric | v0.1 Model | P1 Model |
|--------|------------|----------|
| **Prediction Accuracy** | Not measured | 56.90% |
| **Financial Viability** | Unknown | $1,207.13 profit |
| **Risk Assessment** | Not available | Quantified |
| **Reproducibility** | Limited | Full |

## Testing Environment Details

### System Configuration
- **OS**: Windows 11
- **Python**: 3.12
- **TensorFlow**: 2.20.0
- **CPU Optimizations**: SSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX_VNNI, FMA
- **Memory Usage**: ~2GB during training

### Data Sources
- **v0.1**: CBA.AX from yfinance (3.5 years training data)
- **P1**: META from yfinance (5 years training data)
- **Network**: Stable internet connection required

## Validation Checklist

### v0.1 Model Validation
- ✅ Model trains without errors
- ✅ Prediction generated successfully
- ✅ Visualization displays correctly
- ❌ No quantitative metrics available
- ❌ No model persistence

### P1 Model Validation
- ✅ Training completes with validation
- ✅ Model saves successfully
- ✅ Testing loads model correctly
- ✅ Comprehensive metrics calculated
- ✅ Profit analysis generated
- ✅ CSV results exported
- ✅ All files created in correct directories

## Screenshots Documentation

*Note: The following screenshots should be captured and added:*

### v0.1 Model Screenshots
1. **Training Progress**: Console output showing epoch progression
2. **Final Prediction**: Terminal showing prediction result
3. **Visualization**: Matplotlib chart of actual vs predicted prices

### P1 Model Screenshots
1. **Training Progress**: Console showing validation loss improvements
2. **Model Saving**: Confirmation of file saves
3. **Testing Results**: Complete metrics output
4. **File Structure**: Directory listing showing generated files
5. **CSV Results**: Sample of exported trading analysis

## Troubleshooting Results

### Issues Encountered
1. **TensorFlow Warnings**: Informational only, no impact on functionality
2. **yfinance Deprecation**: Future warning about auto_adjust parameter
3. **Memory Usage**: Manageable within system limits

### Solutions Applied
1. **Warnings**: Documented as expected behavior
2. **Deprecation**: No immediate action required
3. **Performance**: Optimized batch sizes for available memory

## Next Steps

Based on testing results:
1. [Analyze Performance Metrics](Performance-Metrics.md)
2. [Compare Model Effectiveness](Model-Comparison.md)
3. [Review Insights](Insights-Recommendations.md)
4. [Document Weekly Progress](Weekly-Reports.md)

---

*Testing completed successfully: August 24, 2025*
