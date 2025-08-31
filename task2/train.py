# File: train.py
# Task 2: Training Script with Enhanced Data Processing
# Building upon v0.1.py with the 5 required improvements

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from model import StockPredictionModel
from data_utils import DataProcessor

# Configuration - same as v0.1
COMPANY = 'META'
PREDICTION_DAYS = 60
PRICE_VALUE = "Close"

# Task 2 Requirement (a): Specify date ranges
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'

def main():
    """
    Main training function with Task 2 enhancements
    """
    print("Stock Price Prediction - Training")
    print("Building upon v0.1 with enhanced data processing")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Task 2 Requirements (a) & (d): Load data with date specification and local storage
    print(f"\nLoading training data for {COMPANY}...")
    print(f"Date range: {TRAIN_START} to {TRAIN_END}")
    data = processor.load_stock_data(COMPANY, TRAIN_START, TRAIN_END, save_local=True)
    
    print(f"Original data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Task 2 Requirement (b): Handle missing data
    print(f"\nHandling missing data...")
    data = processor.handle_missing_data(data, method='drop')
    
    # Task 2 Requirement (e): Scale data and save scaler
    print(f"\nScaling data...")
    scaled_data, scaler = processor.scale_data(
        data, PRICE_VALUE, 
        save_scaler=True, 
        scaler_name=f"{COMPANY}_{TRAIN_START}_{TRAIN_END}"
    )
    
    # Prepare training sequences (same approach as v0.1)
    print(f"\nPreparing training sequences...")
    x_data, y_data = processor.prepare_sequences(scaled_data, PREDICTION_DAYS)
    print(f"Sequence data shape: X={x_data.shape}, y={y_data.shape}")
    
    # Task 2 Requirement (c): Split data using different methods
    print(f"\nSplitting data...")
    x_train, x_val, y_train, y_val = processor.split_data(
        x_data, y_data, test_size=0.2, method='date'
    )
    
    # Build model using class-based approach (like P1 model)
    print(f"\nBuilding LSTM model...")
    model = StockPredictionModel(
        sequence_length=PREDICTION_DAYS,
        n_features=1,
        units=50,
        n_layers=3,
        dropout=0.2
    )

    # Build and show model summary
    model.build_model()
    print("Model summary:")
    model.get_model_summary()

    # Train model (same parameters as v0.1)
    print(f"\nTraining model...")
    history = model.train_model(
        x_train, y_train,
        X_val=x_val, y_val=y_val,
        epochs=25,
        batch_size=32,
        verbose=1
    )

    # Save model weights (like P1 model)
    model_path = f"models/{COMPANY}_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save_model_weights(model_path)
    
    # Save training plots (like P1 model)
    print(f"\nSaving training plots...")
    results_dir = "results"
    model.save_training_plots(save_dir=results_dir)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Model weights saved to: {model_path.replace('.h5', '.weights.h5')}")
    print(f"Training plots saved to: {results_dir}/training_history.png")
    print(f"Scaler saved for future use")
    
    print(f"\nTask 2 requirements implemented:")
    print("(a) Date range specification")
    print("(b) Missing data handling")
    print("(c) Data splitting methods")
    print("(d) Local data storage")
    print("(e) Feature scaling and scaler storage")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
