# COS30018 - Option C - Stock Prediction Models Wiki

**Student:** Chiraath Madahapola - 104834009
**Course:** COS30018 - Intelligent Systems
**Project:** Stock Price Prediction using LSTM Neural Networks
**Date:** August 31, 2025

## Project Overview

This wiki documents the complete implementation journey of LSTM-based stock prediction models, from baseline implementations to advanced data processing frameworks:

- **v0.1 Model**: Baseline implementation with basic functionality (Task 1)
- **P1 Model**: Enhanced implementation with advanced features and comprehensive metrics (Task 1)
- **Task 2 Enhanced Model**: Advanced data processing implementation with comprehensive enhancements

## Quick Navigation

### 📋 Project Documentation
- [Environment Setup](Environment-Setup.md) - Virtual environment configuration and dependencies
- [Model Comparison](Model-Comparison.md) - Detailed comparison between v0.1 and P1 models
- [Performance Metrics](Performance-Metrics.md) - Training and testing results
- [Task 2 Data Processing Report](Task2-Data-Processing-Report.md) - **NEW** Comprehensive Task 2 implementation report

### 🔧 Implementation Details
- [v0.1 Model](v0.1-Model.md) - Baseline model documentation (Task 1)
- [P1 Model](P1-Model.md) - Enhanced model documentation (Task 1)
- [Training Workflow](Training-Workflow.md) - Step-by-step training procedures
- [Testing Results](Testing-Results.md) - Evaluation outcomes and screenshots

### 📊 Results & Analysis
- [Weekly Reports](Weekly-Reports.md) - Progress tracking and submissions
- [Insights and Recommendations](Insights-Recommendations.md) - Key findings and future directions

## Project Structure

```
project_option_c/
├── task1/                    # Task 1 implementations (archived)
│   ├── p1/                   # P1 model files
│   │   ├── p1.py
│   │   ├── train_p1.py
│   │   ├── test_p1.py
│   │   └── parameters.py
│   └── v0.1/                 # v0.1 baseline model
│       └── v0.1.py
├── task2/                    # Task 2 enhanced data processing
│   ├── train.py              # Training script with data enhancements
│   ├── test.py               # Testing and evaluation framework
│   ├── model.py              # Class-based LSTM model architecture
│   ├── data_utils.py         # Data processing utilities
│   ├── data/                 # Local data storage (cached datasets)
│   ├── scalers/              # Persistent MinMaxScaler objects
│   ├── models/               # Trained model files (.h5 and .weights.h5)
│   ├── results/              # Generated plots and visualizations
│   └── README.md             # Task 2 documentation
├── p1/                       # Original P1 model outputs (preserved)
│   ├── data/                 # P1 training data
│   ├── logs/                 # TensorBoard logs
│   ├── results/              # Model weights and outputs
│   ├── model_checkpoints/    # Saved model checkpoints
│   └── csv-results/          # Performance metrics CSV
├── config.py                 # Global configuration settings
├── requirements.txt          # Python dependencies
├── COS30018_Task1_Report.md  # Task 1 comprehensive report
├── COS30018_Task2_Report.md  # Task 2 academic report
├── images/                   # Generated figures and plots
└── wiki/                     # Project documentation
```

## Key Achievements

### Task 1 Accomplishments
✅ **Environment Setup**: Successfully configured Python 3.12 virtual environment
✅ **Model Implementation**: Both v0.1 and P1 models functional
✅ **Performance Analysis**: P1 model shows superior results with 56.90% accuracy
✅ **Profit Analysis**: P1 model generates $1,207.13 total profit
✅ **Code Quality**: Modular, maintainable architecture implemented

### Task 2 Accomplishments
✅ **Enhanced Data Processing**: All 5 requirements fully implemented
✅ **Class-Based Architecture**: Object-oriented model design with StockPredictionModel class
✅ **Modular Framework**: Separate train.py, test.py, model.py, and data_utils.py files
✅ **Data Persistence**: Local caching and scaler storage with intelligent loading
✅ **Performance Excellence**: 5.71% MAPE with comprehensive error handling
✅ **Academic Documentation**: Professional report with technical analysis

## Assignment Requirements Status

### Task 1 Requirements
- [x] Environment setup with virtual environment
- [x] v0.1 and P1 model testing and execution
- [x] Performance comparison and analysis
- [x] GitHub repository setup
- [x] Wiki documentation creation
- [x] Task 1 Report completion (PDF ready)

### Task 2 Requirements
- [x] Function to specify start/end dates for dataset
- [x] NaN value handling with multiple strategies
- [x] Multiple data splitting methods (chronological/random)
- [x] Local data storage option implemented
- [x] Feature scaling with scaler persistence
- [x] META ticker usage when CBA.AX detected
- [x] Academic report generation

## Contact Information

**Project Leader**: Chiraath Madahapola  
**Student ID**: 104834009  
**Email**: [Contact via Canvas]  
**Repository**: [GitHub Repository Link]

---

*Last Updated: August 31, 2025 - Task 2 Enhanced Data Processing Implementation Complete*
