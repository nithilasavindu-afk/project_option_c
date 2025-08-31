# Task 2: Stock Price Prediction with Improved Data Processing

This folder contains the Task 2 implementation that builds upon the original v0.1.py code with enhanced data processing capabilities.

## Files Overview

### Core Files
- **`train.py`** - Training script with Task 2 enhanced data processing
- **`test.py`** - Testing script with Task 2 enhanced evaluation
- **`model.py`** - LSTM model architecture (same as v0.1)
- **`data_utils.py`** - Utility functions for data processing operations

### Generated Folders (created during execution)
- **`data/`** - Stores downloaded stock data locally
- **`scalers/`** - Stores saved MinMaxScaler objects
- **`models/`** - Stores trained model files

## Task 2 Requirements Implementation

### (a) Function to specify start and end dates for dataset
- Implemented in `DataProcessor.load_stock_data()` method
- Allows flexible date range specification
- Example: `load_stock_data('META', '2020-01-01', '2023-08-01')`

### (b) Handle NaN issues in data
- Implemented in `DataProcessor.handle_missing_data()` method
- Four methods available: 'drop', 'forward_fill', 'backward_fill', 'interpolate'
- Provides detailed reporting of missing values before and after handling

### (c) Different methods to split data into train/test
- Implemented in `DataProcessor.split_data()` method
- Two methods: 'date' (chronological) and 'random'
- Chronological maintains time order, random shuffles data

### (d) Option to store downloaded data locally
- Integrated into `DataProcessor.load_stock_data()` method
- Automatically saves data as CSV files in `data/` folder
- Checks for existing files before downloading to save time

### (e) Scale feature columns and store scalers
- Implemented in `DataProcessor.scale_data()` and `load_scaler()` methods
- Saves MinMaxScaler objects using pickle in `scalers/` folder
- Allows consistent scaling across training and testing phases

## How to Run

### Step 1: Train the model
```bash
python train.py
```

### Step 2: Test the model
```bash
python test.py
```

## Key Improvements from v0.1

1. **Modular Design**: Code is organized into reusable functions and classes
2. **Data Persistence**: Downloaded data is cached locally to avoid repeated API calls
3. **Flexible Data Handling**: Multiple options for handling missing data and splitting datasets
4. **Scaler Management**: Scalers are saved and can be reloaded for consistent preprocessing
5. **Better Error Handling**: More robust data processing with informative messages
6. **Performance Metrics**: Comprehensive evaluation including MAE, RMSE, and MAPE

## Model Architecture

The LSTM model maintains the same architecture as v0.1:
- 3 LSTM layers (50 units each)
- Dropout layers (0.2) for regularization
- Dense output layer
- Adam optimizer with mean squared error loss

## Expected Output

When running the training and testing scripts:
1. Data loading and preprocessing steps
2. Model training progress
3. Performance metrics (MAE, RMSE, MAPE)
4. Visualization plots
5. Next-day price prediction
6. Summary of implemented requirements

## Dependencies

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn
- yfinance

## Notes

- The code maintains the same prediction accuracy as v0.1 while adding enhanced data processing
- All file paths are relative and will work from the task2 directory
- The implementation follows the original v0.1 structure and style
- Comments are educational and explain the reasoning behind each step
