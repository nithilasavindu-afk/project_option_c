"""
Task C.6 - Statistical and ML Models for Ensemble
Implements ARIMA, SARIMA, and Random Forest models.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(series):
    """Check if series is stationary using ADF test."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series)
    return result[1] < 0.05  # p-value < 0.05 means stationary

def make_stationary(series, max_diff=2):
    """Make series stationary by differencing."""
    diff_series = series.copy()
    for i in range(max_diff):
        if check_stationarity(diff_series):
            return diff_series, i
        diff_series = diff_series.diff().dropna()
    return diff_series, max_diff

def train_arima(data, order=(5,1,0), val_size=None):
    """Train ARIMA model."""
    if val_size is None:
        val_size = int(len(data) * 0.2)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    
    try:
        # Make data stationary if needed
        stationary_data, d = make_stationary(pd.Series(train_data))
        order = (order[0], d, order[2])  # Update d based on differencing

        model = ARIMA(stationary_data, order=order)
        model_fit = model.fit()

        # Forecast for validation
        forecast = model_fit.forecast(steps=val_size)

        # Inverse differencing if needed
        if d > 0:
            forecast = np.cumsum(forecast) + train_data[-d]

        # Calculate metrics
        mae = mean_absolute_error(val_data, forecast)
        rmse = np.sqrt(mean_squared_error(val_data, forecast))
        mape = np.mean(np.abs((val_data - forecast) / val_data)) * 100

        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        return forecast, metrics
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        return np.zeros(val_size), {'MAE': 0, 'RMSE': 0, 'MAPE': 0}

def train_sarima(data, order=(1,1,1,1,1,1,7), val_size=None):
    """Train SARIMA model."""
    if val_size is None:
        val_size = int(len(data) * 0.2)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    
    try:
        # Make data stationary if needed
        stationary_data, d = make_stationary(pd.Series(train_data))
        order = (order[0], d, order[2], order[3], order[4], order[5], order[6])

        model = SARIMAX(stationary_data, order=order[:3], seasonal_order=order[3:])
        model_fit = model.fit(disp=False)

        # Forecast
        forecast = model_fit.forecast(steps=val_size)

        # Inverse differencing
        if d > 0:
            forecast = np.cumsum(forecast) + train_data[-d]

        mae = mean_absolute_error(val_data, forecast)
        rmse = np.sqrt(mean_squared_error(val_data, forecast))
        mape = np.mean(np.abs((val_data - forecast) / val_data)) * 100

        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        return forecast, metrics
    except Exception as e:
        print(f"SARIMA training failed: {e}")
        return np.zeros(val_size), {'MAE': 0, 'RMSE': 0, 'MAPE': 0}

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=10):
    """Train Random Forest model."""
    try:
        # For RF, use flattened sequences as features
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train_flat, y_train)

        predictions = model.predict(X_val_flat)

        # Assuming y_val is scaled, but for metrics we need actual scale
        # This is a simplification; in practice, inverse transform
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100

        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        return predictions, metrics
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        return np.zeros(len(y_val)), {'MAE': 0, 'RMSE': 0, 'MAPE': 0}