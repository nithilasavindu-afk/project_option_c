"""
Configuration Parameters for Task 2 - Enhanced Stock Price Prediction
COS30018 - Option C - Task 2: Data Processing 1

This module builds upon the existing parameters.py from Task 1 P1 model,
extending it with Task 2 enhanced data processing capabilities.
Based on task1/p1/parameters.py
"""

import os
import time
from tensorflow.keras.layers import LSTM

# =============================================================================
# CORE PARAMETERS (from P1 model - task1/p1/parameters.py)
# =============================================================================

# Window size or the sequence length
N_STEPS = 50

# Lookup step, 1 is the next day, 15 is 15 days ahead
LOOKUP_STEP = 15

# Whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# Whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# Whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# Test ratio size, 0.2 is 20%
TEST_SIZE = 0.2

# Features to use (keeping P1 model feature names but handling yfinance column mapping)
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Date configuration
date_now = time.strftime("%Y-%m-%d")

# =============================================================================
# MODEL PARAMETERS (from P1 model - task1/p1/parameters.py)
# =============================================================================

N_LAYERS = 2
CELL = LSTM  # LSTM cell
UNITS = 256  # 256 LSTM neurons
DROPOUT = 0.4  # 40% dropout
BIDIRECTIONAL = False  # Whether to use bidirectional RNNs

# =============================================================================
# TRAINING PARAMETERS (from P1 model - task1/p1/parameters.py)
# =============================================================================

# Loss function - using huber loss like P1 model
LOSS = "huber"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 25

# Meta stock market (from P1 model)
ticker = "META"

# =============================================================================
# TASK 2 ENHANCED DATA PROCESSING PARAMETERS
# =============================================================================

# Data date range configuration - Task 2 requirement (a)
DEFAULT_START_DATE = None  # Will use yfinance period="5y" like P1 model
DEFAULT_END_DATE = None    # Current date if None

# Data storage configuration - Task 2 requirement (d)
SAVE_DATA_LOCALLY = True
DATA_DIRECTORY = "task2_data"
SCALERS_DIRECTORY = "task2_scalers"

# NaN handling strategy - Task 2 requirement (b)
NAN_STRATEGY = "drop"  # Options: "drop", "forward_fill", "backward_fill", "interpolate"

# Stock symbol logic - Task 2 requirement: Use META if CBA.AX is present
def get_ticker_symbol(requested_ticker="META"):
    """
    Determine which ticker to use based on Task 2 requirements.
    If CBA.AX is requested or present, use META instead.
    """
    if requested_ticker == "CBA.AX":
        print(f"CBA.AX detected, switching to META as per Task 2 requirements")
        return "META"
    return requested_ticker

# =============================================================================
# FILE PATHS AND NAMING (adapted from P1 model)
# =============================================================================

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, SCALERS_DIRECTORY, "task2_results", "task2_logs"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Data file naming (Task 2 enhancement)
ticker_data_filename = os.path.join(DATA_DIRECTORY, f"{ticker}_{date_now}.csv")
scalers_filename = os.path.join(SCALERS_DIRECTORY, f"{ticker}_scalers_{date_now}.pkl")

# Model naming (same structure as P1 model)
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" \
             f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-" \
             f"layers-{N_LAYERS}-units-{UNITS}"

if BIDIRECTIONAL:
    model_name += "-b"

# Model file paths
model_weights_path = os.path.join("task2_results", f"{model_name}.weights.h5")
model_checkpoint_path = os.path.join("task2_results", f"{model_name}_checkpoint.h5")
tensorboard_log_dir = os.path.join("task2_logs", model_name)

# =============================================================================
# FILE PATHS AND NAMING
# =============================================================================

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, SCALERS_DIRECTORY, "task2_results", "task2_logs"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Data file naming
ticker_data_filename = os.path.join(DATA_DIRECTORY, f"{ticker}_{date_now}.csv")
scalers_filename = os.path.join(SCALERS_DIRECTORY, f"{ticker}_scalers_{date_now}.pkl")

# Model naming - make it unique based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" \
             f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-" \
             f"layers-{N_LAYERS}-units-{UNITS}"

if BIDIRECTIONAL:
    model_name += "-b"

# Model file paths
model_weights_path = os.path.join("task2_results", f"{model_name}.weights.h5")
model_checkpoint_path = os.path.join("task2_results", f"{model_name}_checkpoint.h5")
tensorboard_log_dir = os.path.join("task2_logs", model_name)

# =============================================================================
# TASK 2 DISPLAY AND LOGGING PARAMETERS
# =============================================================================

# Plotting configuration
FIGURE_SIZE = (12, 8)
SAVE_PLOTS = True
PLOTS_DIRECTORY = "task2_plots"

if not os.path.exists(PLOTS_DIRECTORY):
    os.makedirs(PLOTS_DIRECTORY)

print(f"Task 2 Configuration loaded - Building upon P1 model")
print(f"Ticker: {ticker}")
print(f"Model: {model_name}")
print(f"Data will be saved to: {DATA_DIRECTORY}")
print(f"Scalers will be saved to: {SCALERS_DIRECTORY}")
