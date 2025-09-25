# Task 6: Ensemble Modeling

## Overview
Task 6 implements an ensemble forecasting approach combining multiple machine learning and statistical models for improved stock price prediction accuracy.

## Models Used
- **Deep Learning Models**: LSTM, GRU, RNN (adapted from Task 5)
- **Statistical Models**: ARIMA, SARIMA
- **Machine Learning**: Random Forest
- **Ensemble Method**: Simple averaging of all model predictions

## Files
- `ensemble_trainer.py`: Main training script
- `statistical_models.py`: ARIMA, SARIMA, Random Forest implementations
- `ensemble_methods.py`: Ensemble averaging logic
- `evaluate_task6.py`: Evaluation and visualization
- `data_utils.py`: Data processing utilities (copied from Task 5)
- `model_factory.py`: DL model factory (copied from Task 5)

## Requirements
- Python 3.8+
- TensorFlow
- statsmodels
- scikit-learn
- pandas, numpy, matplotlib

## Usage
```bash
cd task6
python ensemble_trainer.py --company META --multistep --k 5
```

## Output
- Individual model predictions and metrics
- Ensemble predictions
- Comparison plots and reports
- Saved models and results in `task6_results/`

## Report
The Task 6 Report (PDF) will summarize:
- Implementation details
- Experimental results with different configurations
- Performance comparisons
- References to research sources