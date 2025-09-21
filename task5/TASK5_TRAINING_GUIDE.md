# Task 5 - Multivariate Multistep Stock Price Prediction
## Training and Testing Guide

This guide provides step-by-step instructions for training and evaluating the Task 5 implementation with multivariate and multistep prediction capabilities.

## 🚀 Quick Start - Interactive Mode

**Easiest way to get started:**
```bash
cd D:\project_option_C\project_option_c\task5
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py
```

This will launch an **interactive menu** where you can select all options by number:
- Prediction mode (single-step vs multistep)
- k value for multistep prediction
- Features (univariate vs multivariate)
- Model type (LSTM, GRU, RNN)
- Training parameters (epochs, layers, units, dropout)
- Device selection (GPU/CPU)
- Training mode (single run vs grid search)

## Prerequisites

Make sure you're in the virtual environment:
```bash
& D:/project_option_C/.venv/Scripts/python.exe --version
```

## Interactive Menu Options

### Prediction Mode
1. **Single-step** - Predict next 1 value
2. **Multistep** - Predict next k values (then choose k: 3, 5, 7, 10, or custom)

### Features
1. **Univariate** - Close price only
2. **Multivariate** - OHLCV features (Open, High, Low, Close, Volume)

### Model Types
1. **LSTM** - Long Short-Term Memory
2. **GRU** - Gated Recurrent Unit
3. **RNN** - Simple Recurrent Neural Network

### Training Parameters
- **Epochs**: 1-100
- **Layers**: 1-5
- **Units**: 16-256
- **Dropout**: 0.0-0.5

### Device Selection
1. **GPU** - If available (faster)
2. **CPU** - Always available

### Training Mode
1. **Single training run**
2. **Grid search** - Test multiple configurations automatically

## Interactive Menu Walkthrough

When you run `train_task5.py` without arguments, you'll see:

```
🎯 TASK 5 - Interactive Training Configuration
============================================================

📊 PREDICTION MODE:
1. Single-step (predict next 1 value)
2. Multistep (predict next k values)
Select mode (1-2):
```

**Example selections for Task 5 main requirement:**
1. Choose: `2` (Multistep)
2. Choose: `2` (k=5)
3. Choose: `2` (Multivariate)
4. Choose: `1` (LSTM)
5. Enter: `10` (epochs)
6. Enter: `3` (layers)
7. Enter: `64` (units)
8. Enter: `0.2` (dropout)
9. Choose: `2` (CPU)
10. Choose: `1` (Single run)

This gives you the exact Task 5 requirement: **Multistep (k=5) + Multivariate + LSTM**

## Available Options

### Two Ways to Run Training:

#### 1. Interactive Mode (Recommended for beginners)
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py
```
- No command-line arguments needed
- Menu-driven selection of all options
- Perfect for testing different configurations

#### 2. Command-Line Mode (Advanced users)
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --help
```

**Key Task 5 Features:**
- `--multistep`: Enable multistep prediction
- `--k K`: Prediction horizon (k steps ahead)
- `--multivariate`: Use OHLCV features instead of just Close
- `--device {gpu,cpu}`: Hardware selection
- `--grid`: Run hyperparameter grid search

## CLI User Input Methods

### Interactive Mode (Easiest)
```bash
# Run without arguments for menu-driven setup
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py
```
**What it does:** Guides you through selections by number for all options.

### Command-Line Mode (Advanced)
```bash
# Full command with all parameters
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --multistep --k 5 --multivariate --model lstm --epochs 10 --device cpu --out_root results

# Get help for all options
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --help
```

### Quick Examples
```bash
# Task 5 main requirement (Multistep + Multivariate + LSTM)
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --multistep --k 5 --multivariate --epochs 10

# Compare different models
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --multistep --k 5 --multivariate --model gru --epochs 10
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --multistep --k 5 --multivariate --model rnn --epochs 10

# Grid search for best parameters
& D:/project_option_C/.venv/Scripts/python.exe train_task5.py --grid --multistep --k 5 --multivariate --epochs 5
```

## Step-by-Step Testing Guide

### Step 1: Basic Training Tests

#### 1.1 Univariate Single-step (Baseline)
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --epochs 5 --device cpu --out_root univariate_single
```

#### 1.2 Univariate Multistep (k=3)
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multistep --k 3 --epochs 5 --device cpu --out_root univariate_multistep
```

#### 1.3 Multivariate Single-step
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multivariate --epochs 5 --device cpu --out_root multivariate_single
```

#### 1.4 Multivariate Multistep (k=5) - Main Task 5 Requirement
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multistep --k 5 --multivariate --epochs 10 --device cpu --out_root multivariate_multistep
```

### Step 2: Hyperparameter Grid Search

#### 2.1 Quick Grid Search (Recommended)
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --grid --multistep --k 5 --multivariate --epochs 5 --device cpu
```

#### 2.2 Custom Grid Search
```bash
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --grid --multistep --k 5 --multivariate --models lstm gru --layers_list 2 3 --units_list 32 64 --dropouts 0.1 0.2 --epochs 5 --device cpu
```

### Step 3: Evaluate Trained Models

#### 3.1 Evaluate Specific Model
```bash
# Basic evaluation command
& D:/project_option_C/.venv/Scripts/python.exe evaluate_task5.py --model_path MODEL_PATH --multistep --k 5 --multivariate --out_dir evaluation_results

# Example with actual model path
& D:/project_option_C/.venv/Scripts/python.exe evaluate_task5.py --model_path task5_results/META_lstm_20250921-XXXXXX/META_lstm.keras --multistep --k 5 --multivariate --out_dir evaluation_results
```

#### 3.2 Auto-select Best Model (after grid search)
```bash
& D:/project_option_C/.venv/Scripts/python.exe evaluate_task5.py --auto_select_best --multistep --k 5 --multivariate --out_dir best_model_eval
```

#### 3.3 Evaluation Options
- `--model_path`: Path to trained .keras model file
- `--multistep`: Enable multistep evaluation mode
- `--k`: Number of steps to predict (must match training)
- `--multivariate`: Use multivariate features (must match training)
- `--out_dir`: Output directory for results
- `--auto_select_best`: Automatically find best model from grid search

#### 3.4 Evaluation Output Files
- `metrics.json` - Performance metrics (MAE, RMSE, MAPE)
- `predictions.csv` - Actual vs predicted values
- `test_fit.png` - Test data visualization

### Step 4: Compare Different Approaches

#### 4.1 Test Different k Values
```bash
# Test k=3, k=5, k=7
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multistep --k 3 --multivariate --epochs 5 --device cpu --out_root k3_test
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multistep --k 7 --multivariate --epochs 5 --device cpu --out_root k7_test
```

#### 4.2 Test Different Models
```bash
# Compare LSTM vs GRU vs RNN
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --model lstm --multistep --k 5 --multivariate --epochs 5 --device cpu --out_root lstm_test
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --model gru --multistep --k 5 --multivariate --epochs 5 --device cpu --out_root gru_test
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --model rnn --multistep --k 5 --multivariate --epochs 5 --device cpu --out_root rnn_test
```

### Step 5: Check Results

#### 5.1 View Training Results
```bash
# Check the experiments CSV after grid search
type task5_results\task5_experiments.csv
```

#### 5.2 View Evaluation Metrics
```bash
# Check evaluation results
type evaluation_results\metrics.json
```

## Recommended Testing Sequence

1. **Start with:** Multivariate Multistep (k=5) - your main Task 5 requirement
2. **Run:** Grid search to find best hyperparameters
3. **Evaluate:** Best model on test data
4. **Compare:** Different k values and model types

## Example Complete Workflow

```bash
# 1. Train main model
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --multistep --k 5 --multivariate --epochs 10 --device cpu

# 2. Run grid search for optimization
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/train_task5.py --grid --multistep --k 5 --multivariate --epochs 5 --device cpu

# 3. Evaluate best model
& D:/project_option_C/.venv/Scripts/python.exe D:/project_option_C/project_option_c/task5/evaluate_task5.py --auto_select_best --multistep --k 5 --multivariate --out_dir final_evaluation
```

## Task 5 Requirements Verification

✅ **Requirement 1**: Multistep prediction function (k future steps)
✅ **Requirement 2**: Multivariate prediction function (OHLCV features)
✅ **Requirement 3**: Combined multivariate multistep function
✅ **Bonus**: GPU/CPU selection, auto model selection, user-specified k

## Output Files

Each training run creates:
- `config.json` - Training configuration
- `metrics.json` - Performance metrics (MAE, RMSE, MAPE)
- `predictions.csv` - Model predictions vs actual values
- `training_history.png` - Training loss curves
- `val_fit.png` - Validation predictions plot
- `best.weights.h5` - Best model weights
- `*.keras` - Complete model file
- `model_summary.txt` - Model architecture

## Troubleshooting

### Virtual Environment Issues
```bash
# Activate virtual environment
& D:/project_option_C/.venv/Scripts/activate

# Or run directly with full path
& D:/project_option_C/.venv/Scripts/python.exe script.py
```

### GPU/CPU Selection
```bash
# Force CPU (if GPU issues)
--device cpu

# Try GPU (if available)
--device gpu
```

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Reduce sequence length: `--seq_len 30`
- Use fewer epochs: `--epochs 5`

## Performance Expectations

Typical results for multivariate multistep (k=5):
- **MAE**: 10-15 (price units)
- **RMSE**: 13-20 (price units)
- **MAPE**: 5-8% (percentage error)

Lower values indicate better performance.</content>
<parameter name="filePath">d:\project_option_C\project_option_c\task5\TASK5_TRAINING_GUIDE.md