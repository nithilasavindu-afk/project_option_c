# COS30018 - Stock Price Prediction using LSTM Neural Networks

**Course:** COS30018 - Intelligent Systems  
**Student:** Chiraath Madahapola  
**Student ID:** 104834009  
**Institution:** Swinburne University of Technology  
**Date:** August 2025

## Project Overview

This project implements and compares two LSTM-based neural network models for stock price prediction. The project demonstrates the evolution from a baseline implementation (v0.1) to an enhanced production-ready model (P1) with comprehensive evaluation metrics and trading analysis.

### Models Implemented

#### v0.1 Model (Baseline)
- **Architecture:** 3-layer LSTM with 50 units each
- **Features:** Single feature (closing price)
- **Prediction:** Next-day price forecasting
- **Target Stock:** META (Meta Platforms Inc.)
- **Training Period:** 3.5 years (2020-2023)

#### P1 Model (Enhanced)
- **Architecture:** 2-layer LSTM with 256 units each
- **Features:** Multi-feature input (OHLCV data)
- **Prediction:** 15-day ahead forecasting
- **Target Stock:** META (Meta Platforms Inc.)
- **Training Period:** 5 years of historical data

## Key Features

### v0.1 Model Capabilities
- ✅ Basic LSTM implementation for stock prediction
- ✅ yfinance data integration
- ✅ MinMaxScaler preprocessing
- ✅ Matplotlib visualization
- ✅ Simple next-day prediction

### P1 Model Advanced Features
- ✅ Multi-feature input processing (Open, High, Low, Close, Volume)
- ✅ Advanced model persistence (.keras format)
- ✅ Comprehensive evaluation metrics
- ✅ Profit/loss analysis with trading signals
- ✅ TensorBoard logging integration
- ✅ Modular architecture with separate training/testing
- ✅ Robust error handling and data validation
- ✅ CSV export of detailed results

## Performance Results

### v0.1 Model Performance
- **Training Time:** ~35 seconds
- **Final Loss:** 0.0045 (MSE)
- **Prediction:** $468.90 next-day forecast
- **Evaluation:** Visual comparison only

### P1 Model Performance
- **Training Time:** ~90 seconds
- **Accuracy:** 56.90%
- **15-Day Prediction:** $768.91
- **Total Profit:** $1,207.13
- **Profit per Trade:** $5.05
- **Mean Absolute Error:** $116.63
- **Huber Loss:** 0.0016

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- Git (for repository management)
- Internet connection (for data download)

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd project_option_c
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
```
numpy
pandas
matplotlib
tensorflow
scikit-learn
pandas-datareader
yfinance
joblib
```

## Usage Instructions

### Running the v0.1 Model

```bash
# Ensure virtual environment is activated
.\venv\Scripts\activate

# Run the baseline model
python v0.1.py
```

**Expected Output:**
- Training progress for 25 epochs
- Final prediction value
- Matplotlib visualization window

### Running the P1 Model

#### Training Phase:
```bash
# Train the P1 model
python train_p1.py
```

#### Testing Phase:
```bash
# Test the trained model
python test_p1.py
```

**Expected Output:**
- Comprehensive training metrics with validation
- Model and data persistence
- Detailed performance analysis
- Profit/loss calculations
- CSV results export

## Project Structure

```
project_option_c/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── COS30018_Task1_Report.md     # Comprehensive project report
├── v0.1.py                      # Baseline model implementation
├── train_p1.py                  # P1 model training script
├── test_p1.py                   # P1 model testing script
├── p1.py                        # P1 model core implementation
├── parameters.py                # Configuration parameters
├── p1/                          # P1 model outputs
│   ├── data/                    # Raw stock data
│   ├── logs/                    # TensorBoard logs
│   ├── results/                 # Model weights (.h5)
│   ├── model_checkpoints/       # Full models (.keras) and data
│   └── csv-results/             # Evaluation outputs
└── wiki/                        # Comprehensive documentation
    ├── Home.md                  # Project overview
    ├── Environment-Setup.md     # Setup instructions
    ├── Model-Comparison.md      # Detailed comparison
    ├── Performance-Metrics.md   # Results analysis
    └── [additional wiki pages]
```

## Model Comparison

| Feature | v0.1 Model | P1 Model | Winner |
|---------|------------|----------|---------|
| **Architecture** | 3×50 units | 2×256 units | P1 |
| **Input Features** | 1 (close price) | 5 (OHLCV) | P1 |
| **Loss Function** | MSE | Huber | P1 |
| **Evaluation** | Visual only | Comprehensive metrics | P1 |
| **Persistence** | None | Full model + data | P1 |
| **Profit Analysis** | None | $1,207.13 total | P1 |
| **Code Structure** | Monolithic | Modular | P1 |

## Technical Highlights

### Advanced Architecture (P1)
- **Efficient Design:** Fewer layers with higher capacity
- **Robust Loss Function:** Huber loss for financial data stability
- **Multi-Feature Processing:** Comprehensive market data utilization
- **Production Ready:** Full model persistence and deployment capability

### Comprehensive Evaluation
- **Quantitative Metrics:** Accuracy, MAE, profit analysis
- **Trading Simulation:** Buy/sell signal evaluation
- **Risk Assessment:** Performance across multiple time periods
- **Visualization:** Advanced plotting and analysis tools

## Results Summary

The P1 model demonstrates significant improvements over the baseline v0.1 implementation:

- **56.90% Prediction Accuracy** (above random baseline)
- **$1,207.13 Total Profit** from trading simulation
- **Superior Architecture** with efficient parameter utilization
- **Production-Ready Features** including model persistence and comprehensive metrics
- **Modular Design** enabling easy experimentation and deployment

## Configuration

Key parameters can be adjusted in `parameters.py`:

```python
SEQUENCE_LENGTH = 50      # Days of historical data
STEP_SIZE = 15           # Prediction horizon
BATCH_SIZE = 64          # Training batch size
EPOCHS = 25              # Training epochs
LAYERS = 2               # LSTM layers
UNITS = 256              # Units per layer
DROPOUT = 0.4            # Dropout rate
LOSS = "huber"           # Loss function
OPTIMIZER = "adam"       # Optimizer
```

## Hardware Requirements

- **CPU:** Modern multi-core processor (Intel/AMD)
- **RAM:** Minimum 8GB, recommended 16GB
- **Storage:** 2GB free space for dependencies and data
- **Network:** Internet connection for data download

**Note:** The models are optimized for CPU execution with TensorFlow optimizations (SSE, AVX, FMA instructions).

## Troubleshooting

### Common Issues

**Data Download Fails:**
- Check internet connection
- Verify yfinance package installation
- Try different time periods or stock symbols

**Memory Errors:**
- Reduce batch size in parameters.py
- Close other applications
- Consider using a machine with more RAM

**Import Errors:**
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip list`
- Reinstall requirements: `pip install -r requirements.txt`

## Future Enhancements

- Hyperparameter optimization using grid search
- Additional technical indicators integration
- Real-time prediction capabilities
- Multi-stock portfolio analysis
- Advanced ensemble methods
- Risk management features

## Documentation

Comprehensive documentation is available in the `wiki/` directory:
- Setup and configuration guides
- Detailed model analysis
- Performance comparisons
- Training workflows
- Results analysis

## License

This project is developed for academic purposes as part of COS30018 coursework at Swinburne University of Technology.

## Contact

**Student:** Chiraath Madahapola  
**Student ID:** 104834009  
**Course:** COS30018 - Intelligent Systems  
**Institution:** Swinburne University of Technology

---

*This project demonstrates the practical application of LSTM neural networks for financial time series prediction, showcasing the evolution from basic implementation to production-ready systems with comprehensive evaluation and analysis.*
