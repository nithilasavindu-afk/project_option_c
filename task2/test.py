# File: test.py
# Task 2: Testing Script with Enhanced Data Processing
# Building upon v0.1.py testing approach

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from model import StockPredictionModel
from data_utils import DataProcessor, calculate_metrics

# Configuration - same as v0.1
COMPANY = 'META'
PREDICTION_DAYS = 60
PRICE_VALUE = "Close"

# Date ranges
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

def main():
    """
    Main testing function with Task 2 enhancements
    """
    print("Stock Price Prediction - Testing")
    print("Building upon v0.1 with enhanced data processing")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load saved model using class-based approach
    model_path = f"models/{COMPANY}_model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first to train the model")
        return

    print(f"\nLoading trained model...")
    model = StockPredictionModel(
        sequence_length=PREDICTION_DAYS,
        n_features=1,
        units=50,
        n_layers=3,
        dropout=0.2
    )
    model.load_model_weights(model_path)
    
    # Load saved scaler
    scaler_name = f"{COMPANY}_{TRAIN_START}_{TRAIN_END}"
    scaler = processor.load_scaler(scaler_name)
    if scaler is None:
        print("Error: Scaler not found. Please run train.py first.")
        return
    
    # Task 2 Requirements (a) & (d): Load test data
    print(f"\nLoading test data for {COMPANY}...")
    print(f"Date range: {TEST_START} to {TEST_END}")
    
    # Load training data (needed for prediction preparation)
    train_data = processor.load_stock_data(COMPANY, TRAIN_START, TRAIN_END, save_local=True)
    train_data = processor.handle_missing_data(train_data, method='drop')
    
    # Load test data
    test_data = processor.load_stock_data(COMPANY, TEST_START, TEST_END, save_local=True)
    test_data = processor.handle_missing_data(test_data, method='drop')
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Prepare test data for prediction (following v0.1 approach)
    print(f"\nPreparing test data for prediction...")
    x_test = processor.prepare_test_data(
        train_data, test_data, scaler, PREDICTION_DAYS, PRICE_VALUE
    )
    
    # Make predictions
    print(f"\nMaking predictions...")
    predicted_prices_scaled = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
    
    # Get actual prices
    actual_prices = test_data[PRICE_VALUE].values
    
    # Calculate performance metrics
    print(f"\nEvaluating performance...")
    metrics = calculate_metrics(actual_prices, predicted_prices.flatten())
    
    print("Performance Metrics:")
    print(f"MAE: ${metrics['MAE']:.2f}")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    
    # Plot and save results (like P1 model)
    print(f"\nPlotting and saving results...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price", linewidth=2)
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
    plt.title(f"{COMPANY} Stock Price Prediction - Task 2")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Stock Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f"{COMPANY}_prediction_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Prediction plot saved to {plot_path}")
    plt.show()
    
    # Predict next day (following v0.1 approach)
    print(f"\nPredicting next day...")
    
    # Get last sequence for prediction
    total_data = pd.concat((train_data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)
    last_sequence = total_data[-PREDICTION_DAYS:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = last_sequence_scaled.reshape(1, PREDICTION_DAYS, 1)
    
    # Make next day prediction
    next_day_scaled = model.predict(last_sequence_scaled)
    next_day_price = scaler.inverse_transform(next_day_scaled)
    
    current_price = test_data[PRICE_VALUE].iloc[-1]
    predicted_price = next_day_price[0][0]
    price_change = predicted_price - current_price
    percent_change = (price_change / current_price) * 100
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted next day price: ${predicted_price:.2f}")
    print(f"Expected change: ${price_change:.2f} ({percent_change:+.2f}%)")
    
    direction = "increase" if price_change > 0 else "decrease"
    print(f"Prediction: Price expected to {direction}")
    
    print(f"\nTesting completed.")
    print(f"Stock: {COMPANY}")
    print(f"Test period: {TEST_START} to {TEST_END}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    return metrics, predicted_prices, actual_prices

if __name__ == "__main__":
    results = main()
