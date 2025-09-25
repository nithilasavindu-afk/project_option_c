# File: data_utils.py
# Task 5: Data Processing Utilities for Multivariate and Multistep Predictions
# Enhanced utility functions for advanced time series handling

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

class DataProcessor:
    """
    Class to handle all data processing operations for Task 5
    Enhanced for multivariate and multistep predictions
    """

    def __init__(self, data_dir="data", scaler_dir="scalers"):
        self.data_dir = data_dir
        self.scaler_dir = scaler_dir

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)

    def load_stock_data(self, company, start_date, end_date, save_local=True):
        """
        Task 2 Requirement (a): Load stock data with specified date range
        Also implements Requirement (d): Option to store data locally
        """
        filename = f"{self.data_dir}/{company}_{start_date}_{end_date}.csv"

        # Check if data exists locally first
        if save_local and os.path.exists(filename):
            print(f"Loading existing data from {filename}")
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            # Ensure numeric columns are properly typed
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            print(f"Downloading {company} data from {start_date} to {end_date}")
            data = yf.download(company, start=start_date, end=end_date)

            if save_local:
                data.to_csv(filename)
                print(f"Data saved to {filename}")

        return data

    def handle_missing_data(self, data, method='drop'):
        """
        Task 2 Requirement (b): Handle NaN issues in data
        """
        print(f"Data shape before cleaning: {data.shape}")
        missing_count = data.isnull().sum().sum()
        print(f"Missing values found: {missing_count}")

        if missing_count == 0:
            print("No missing values detected")
            return data

        if method == 'drop':
            cleaned_data = data.dropna()
            print("Missing values dropped")
        elif method == 'forward_fill':
            cleaned_data = data.fillna(method='ffill')
            print("Missing values forward filled")
        elif method == 'backward_fill':
            cleaned_data = data.fillna(method='bfill')
            print("Missing values backward filled")
        elif method == 'interpolate':
            cleaned_data = data.interpolate()
            print("Missing values interpolated")
        else:
            print(f"Unknown method '{method}', using drop instead")
            cleaned_data = data.dropna()

        print(f"Data shape after cleaning: {cleaned_data.shape}")
        return cleaned_data

    def split_data(self, x_data, y_data, test_size=0.2, method='date', random_state=42):
        """
        Task 2 Requirement (c): Different methods to split data into train/test
        """
        print(f"Splitting data using '{method}' method...")

        if method == 'date':
            # Chronological split - maintains time order
            split_idx = int(len(x_data) * (1 - test_size))
            x_train = x_data[:split_idx]
            x_test = x_data[split_idx:]
            y_train = y_data[:split_idx]
            y_test = y_data[split_idx:]

        elif method == 'random':
            # Random split - shuffles the data
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=test_size, random_state=random_state
            )

        else:
            print(f"Unknown split method '{method}', using chronological split")
            split_idx = int(len(x_data) * (1 - test_size))
            x_train = x_data[:split_idx]
            x_test = x_data[split_idx:]
            y_train = y_data[:split_idx]
            y_test = y_data[split_idx:]

        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_test)}")

        return x_train, x_test, y_train, y_test

    def scale_data(self, data, column_name, save_scaler=True, scaler_name=None):
        """
        Task 2 Requirement (e): Scale feature columns and store scalers
        """
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Reshape data for scaling
        data_values = data[column_name].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data_values)

        # Save scaler if requested
        if save_scaler and scaler_name:
            scaler_path = f"{self.scaler_dir}/{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to {scaler_path}")

        return scaled_data, scaler

    def scale_data_multivariate(self, data, feature_columns, save_scaler=True, scaler_name=None):
        """
        Scale multivariate data (multiple features) for Task 5
        """
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale all specified features
        feature_data = data[feature_columns].values
        scaled_data = scaler.fit_transform(feature_data)

        # Save scaler if requested
        if save_scaler and scaler_name:
            scaler_path = f"{self.scaler_dir}/{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Multivariate scaler saved to {scaler_path}")

        return scaled_data, scaler

    def load_scaler(self, scaler_name):
        """
        Load a previously saved scaler
        """
        scaler_path = f"{self.scaler_dir}/{scaler_name}_scaler.pkl"

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")
            return scaler
        else:
            print(f"Scaler file not found: {scaler_path}")
            return None

    def prepare_multistep_sequences(self, scaled_data, sequence_length, prediction_horizon=7):
        """
        Prepare data sequences for multistep prediction (Task 5)
        Creates targets as sequences of k future values instead of single values
        """
        x_data = []
        y_data = []

        # Convert to 1D array if needed
        if scaled_data.ndim > 1:
            scaled_data = scaled_data[:, 0]

        # Create sequences with k-step lookahead targets
        for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
            x_data.append(scaled_data[i-sequence_length:i])
            y_data.append(scaled_data[i:i+prediction_horizon])  # k future values

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Reshape for LSTM input (samples, time steps, features)
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

        return x_data, y_data

    def prepare_test_data(self, train_data, test_data, scaler, sequence_length, price_column):
        """
        Prepare test data for prediction (following v0.1 approach)
        """
        # Combine train and test data
        total_data = pd.concat((train_data[price_column], test_data[price_column]), axis=0)

        # Get the inputs needed for prediction
        model_inputs = total_data[len(total_data) - len(test_data) - sequence_length:].values
        model_inputs = model_inputs.reshape(-1, 1)

        # Scale the inputs
        model_inputs = scaler.transform(model_inputs)

        # Create test sequences
        x_test = []
        for i in range(sequence_length, len(model_inputs)):
            x_test.append(model_inputs[i - sequence_length:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_test

    def prepare_test_multistep_data(self, train_data, test_data, scaler, sequence_length, prediction_horizon, price_column):
        """
        Prepare test data for multistep prediction
        Returns sequences and corresponding multistep targets
        """
        # Combine train and test data
        total_data = pd.concat((train_data[price_column], test_data[price_column]), axis=0)

        # Get the inputs needed for prediction
        model_inputs = total_data[len(total_data) - len(test_data) - sequence_length:].values
        model_inputs = model_inputs.reshape(-1, 1)

        # Scale the inputs
        model_inputs = scaler.transform(model_inputs)

        # Create test sequences and targets
        x_test = []
        y_test = []

        for i in range(sequence_length, len(model_inputs) - prediction_horizon + 1):
            x_test.append(model_inputs[i - sequence_length:i, 0])
            y_test.append(model_inputs[i:i + prediction_horizon, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = np.array(y_test)

        return x_test, y_test

    def prepare_test_multivariate_data(self, train_data, test_data, feature_columns, sequence_length):
        """
        Prepare test data for multivariate single-step prediction
        """
        # Combine train and test data for each feature
        total_data = {}
        for col in feature_columns:
            total_data[col] = pd.concat((train_data[col], test_data[col]), axis=0)

        # Create combined dataframe
        combined_df = pd.DataFrame(total_data)

        # Get the portion needed for test sequences
        test_start_idx = len(train_data) - sequence_length
        model_inputs = combined_df.iloc[test_start_idx:].values

        # Create test sequences
        x_test = []
        y_test = []

        for i in range(sequence_length, len(model_inputs)):
            x_test.append(model_inputs[i - sequence_length:i])  # All features
            y_test.append(model_inputs[i, feature_columns.index('Close')])  # Target is Close price

        x_test = np.array(x_test)  # shape: (samples, seq_len, num_features)
        y_test = np.array(y_test)  # shape: (samples,)

        return x_test, y_test

    def prepare_test_multivariate_multistep_data(self, train_data, test_data, feature_columns, sequence_length, prediction_horizon):
        """
        Prepare test data for multivariate multistep prediction
        """
        # Combine train and test data directly as DataFrames
        combined_df = pd.concat([train_data[feature_columns], test_data[feature_columns]], axis=0)

        # Get the portion needed for test sequences
        test_start_idx = len(train_data) - sequence_length
        model_inputs = combined_df.iloc[test_start_idx:].values

        # Create test sequences and multistep targets
        x_test = []
        y_test = []

        for i in range(sequence_length, len(model_inputs) - prediction_horizon + 1):
            x_test.append(model_inputs[i - sequence_length:i])  # All features
            # Target: k future Close prices
            y_test.append(model_inputs[i:i + prediction_horizon, feature_columns.index('Close')])

        x_test = np.array(x_test)  # shape: (samples, seq_len, num_features)
        y_test = np.array(y_test)  # shape: (samples, k)

        return x_test, y_test

    def prepare_multivariate_sequences(self, data, feature_columns, sequence_length, prediction_horizon=1):
        """
        Prepare multivariate sequences for Task 5
        Uses multiple features (Open, High, Low, Close, Volume, Adj Close) as input
        data can be either pandas DataFrame (unscaled) or numpy array (scaled)
        """
        if isinstance(data, np.ndarray):
            # Already scaled data - assume feature_columns is list of indices
            feature_data = data
            close_idx = feature_columns.index('Close') if isinstance(feature_columns[0], str) else feature_columns.index(3)  # Close is typically index 3
        else:
            # Pandas DataFrame - select columns
            feature_data = data[feature_columns].values
            close_idx = feature_columns.index('Close')

        x_data = []
        y_data = []

        # Create sequences
        for i in range(sequence_length, len(feature_data) - prediction_horizon + 1):
            # Input: sequence of multiple features
            x_data.append(feature_data[i-sequence_length:i])  # shape: (seq_len, num_features)

            # Target: future Close prices (can be single or multistep)
            if prediction_horizon == 1:
                y_data.append(feature_data[i, close_idx])  # Single value
            else:
                # Multistep: k future Close prices
                close_prices = feature_data[i:i+prediction_horizon, close_idx]
                y_data.append(close_prices)

        # Convert to numpy arrays
        x_data = np.array(x_data)  # shape: (samples, seq_len, num_features)
        y_data = np.array(y_data)  # shape: (samples,) or (samples, k) for multistep

        return x_data, y_data

def calculate_metrics(actual, predicted):
    """
    Calculate basic performance metrics
    """
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    # Calculate percentage error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def print_data_info(data, title="Data Information"):
    """
    Print basic information about the dataset
    """
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Columns: {list(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")