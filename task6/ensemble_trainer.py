"""
Task C.6 - Machine Learning 3: Ensemble Modeling
Main trainer script for ensemble forecasting with 6 models: LSTM, GRU, RNN, ARIMA, SARIMA, Random Forest.
Combines predictions using simple averaging.
"""

import os, sys
# Fix: make sure task6 is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import argparse, json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import utilities
from data_utils import DataProcessor, calculate_metrics
from model_factory import build_model, model_summary_str
from statistical_models import train_arima, train_sarima, train_random_forest
from ensemble_methods import ensemble_average, ensemble_weighted_average
from evaluate_task6 import evaluate_ensemble

def parse_args():
    p = argparse.ArgumentParser(description="Task C.6 Ensemble Trainer")
    # Data
    p.add_argument("--company", default="META")
    p.add_argument("--price_col", default="Close")
    p.add_argument("--train_start", default="2020-01-01")
    p.add_argument("--train_end", default="2023-08-01")
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--split_method", choices=["date", "random"], default="date")
    p.add_argument("--test_size", type=float, default=0.2)

    # DL Models (from Task 5)
    p.add_argument("--multistep", action="store_true", help="Enable multistep prediction")
    p.add_argument("--k", type=int, default=5, help="Prediction horizon for multistep")
    p.add_argument("--multivariate", action="store_true", help="Use multivariate features")

    # DL Hyperparams
    p.add_argument("--models", nargs="+", default=["lstm", "gru", "rnn"])
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")

    # Statistical Models
    p.add_argument("--arima_order", nargs="+", type=int, default=[5,1,0])
    p.add_argument("--sarima_order", nargs="+", type=int, default=[1,1,1,1,1,1,7])
    p.add_argument("--rf_n_estimators", type=int, default=100)
    p.add_argument("--rf_max_depth", type=int, default=10)

    # Output
    p.add_argument("--out_root", default="task6_results")
    return p.parse_args()

def setup_device(device_preference):
    """Setup TensorFlow for GPU/CPU."""
    import tensorflow as tf
    if device_preference == "gpu":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Using GPU: {gpus}")
                return "GPU"
            except RuntimeError as e:
                print(f"❌ GPU setup failed: {e}, using CPU")
        else:
            print("❌ No GPU, using CPU")
    tf.config.set_visible_devices([], 'GPU')
    print("✅ Using CPU")
    return "CPU"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_and_prepare(args):
    dp = DataProcessor()
    data = dp.load_stock_data(args.company, args.train_start, args.train_end, save_local=True)
    data = dp.handle_missing_data(data, method="drop")

    # For DL models: prepare sequences
    if args.multivariate:
        features = ["Open", "High", "Low", "Close", "Volume"]
        scaled, scaler = dp.scale_data_multivariate(data, features, save_scaler=True,
            scaler_name=f"{args.company}_{args.train_start}_{args.train_end}_multivariate")
        if args.multistep:
            X, y = dp.prepare_multivariate_sequences(scaled, features, args.seq_len, args.k)
            input_shape = (args.seq_len, len(features))
        else:
            X, y = dp.prepare_multivariate_sequences(scaled, features, args.seq_len, 1)
            input_shape = (args.seq_len, len(features))
    else:
        scaled, scaler = dp.scale_data(data, args.price_col, save_scaler=True,
            scaler_name=f"{args.company}_{args.train_start}_{args.train_end}")
        if args.multistep:
            X, y = dp.prepare_multistep_sequences(scaled, args.seq_len, args.k)
            input_shape = (args.seq_len, 1)
        else:
            X, y = dp.prepare_multistep_sequences(scaled, args.seq_len, 1)
            input_shape = (args.seq_len, 1)

    X_tr, X_val, y_tr, y_val = dp.split_data(X, y, test_size=args.test_size, method=args.split_method)

    # For statistical models: raw time series
    raw_data = data[args.price_col].values
    return dp, scaler, (X_tr, X_val, y_tr, y_val), data, input_shape, raw_data

def train_dl_model(args, model_name, X_tr, y_tr, X_val, y_val, input_shape, scaler, run_dir: Path):
    """Train a single DL model."""
    ensure_dir(run_dir)
    model = build_model(model_name, input_shape, units=args.units, n_layers=args.layers,
                       dropout=args.dropout, prediction_horizon=(args.k if args.multistep else 1))

    cbs = [tf.keras.callbacks.ModelCheckpoint(str(run_dir / "best.weights.h5"), save_weights_only=True,
                                             monitor="val_loss", save_best_only=True)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs,
                     batch_size=args.batch_size, verbose=0, callbacks=cbs)

    if (run_dir / "best.weights.h5").exists():
        model.load_weights(str(run_dir / "best.weights.h5"))

    pred_val = model.predict(X_val, verbose=0)

    # Handle predictions shape
    if not args.multistep:
        pred_val_flat = pred_val.ravel()
        true_val_flat = y_val.ravel()
    else:
        # For multistep, take the last prediction step
        pred_val_flat = pred_val[:, -1]
        true_val_flat = y_val[:, -1]

    # Inverse transform
    if args.multivariate:
        # Create dummy array for inverse transform (5 features: OHLCV)
        pred_dummy = np.zeros((len(pred_val_flat), 5))
        pred_dummy[:, 3] = pred_val_flat  # Close is index 3
        true_dummy = np.zeros((len(true_val_flat), 5))
        true_dummy[:, 3] = true_val_flat
        pred_val_inv = scaler.inverse_transform(pred_dummy)[:, 3]
        true_val_inv = scaler.inverse_transform(true_dummy)[:, 3]
    else:
        pred_val_inv = scaler.inverse_transform(pred_val_flat.reshape(-1, 1)).ravel()
        true_val_inv = scaler.inverse_transform(true_val_flat.reshape(-1, 1)).ravel()

    metrics = calculate_metrics(true_val_inv, pred_val_inv)

    # Save
    model.save(run_dir / f"{args.company}_{model_name}.keras")
    pd.DataFrame({"actual": true_val_inv, "predicted": pred_val_inv}).to_csv(run_dir/"predictions.csv", index=False)
    with open(run_dir/"metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    return pred_val_inv, true_val_inv, metrics

def main():
    args = parse_args()
    device_used = setup_device(args.device)

    random.seed(42); np.random.seed(42); tf.random.set_seed(42)

    dp, scaler, (X_tr, X_val, y_tr, y_val), data, input_shape, raw_data = load_and_prepare(args)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_root); ensure_dir(root)

    # Train DL models
    dl_predictions = {}
    dl_metrics = {}
    true_vals = None
    for model_name in args.models:
        run_dir = root / f"{args.company}_{model_name}_{stamp}"
        pred, true, mets = train_dl_model(args, model_name, X_tr, y_tr, X_val, y_val, input_shape, scaler, run_dir)
        dl_predictions[model_name] = pred
        dl_metrics[model_name] = mets
        if true_vals is None:
            true_vals = true

    # Train statistical models
    val_size = len(y_val)
    arima_pred, arima_mets = train_arima(raw_data, args.arima_order, val_size)
    sarima_pred, sarima_mets = train_sarima(raw_data, args.sarima_order, val_size)

    # Train Random Forest (using features from DL if multivariate, else simple)
    rf_pred_scaled, rf_mets = train_random_forest(X_tr, y_tr, X_val, y_val, args.rf_n_estimators, args.rf_max_depth)
    # Inverse transform RF predictions
    if args.multivariate:
        rf_pred_dummy = np.zeros((len(rf_pred_scaled), 5))
        rf_pred_dummy[:, 3] = rf_pred_scaled
        rf_pred = scaler.inverse_transform(rf_pred_dummy)[:, 3]
    else:
        rf_pred = scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()

    # Collect all predictions and metrics
    all_preds = [dl_predictions['lstm'], dl_predictions['gru'], dl_predictions['rnn'], arima_pred, sarima_pred, rf_pred]
    all_mets = [dl_metrics['lstm'], dl_metrics['gru'], dl_metrics['rnn'], arima_mets, sarima_mets, rf_mets]

    # Compute weights based on inverse MAE (higher weight for lower error)
    maes = [mets['MAE'] for mets in all_mets]
    weights = [1.0 / mae for mae in maes]
    weights = [w / sum(weights) for w in weights]  # Normalize

    # Ensemble with weighted average
    ensemble_pred = ensemble_weighted_average(all_preds, weights)
    ensemble_mets = calculate_metrics(true_vals, ensemble_pred)

    # Save ensemble results
    ensemble_dir = root / f"ensemble_{stamp}"
    ensure_dir(ensemble_dir)
    pd.DataFrame({"actual": true, "ensemble_predicted": ensemble_pred}).to_csv(ensemble_dir/"predictions.csv", index=False)
    with open(ensemble_dir/"metrics.json", "w") as f: json.dump(ensemble_mets, f, indent=2)

    # Summary
    print("Task 6 Ensemble Results:")
    print(f"DL Models: {args.models}")
    print(f"Ensemble MAE: {ensemble_mets['MAE']:.4f}, RMSE: {ensemble_mets['RMSE']:.4f}")

if __name__ == "__main__":
    import tensorflow as tf
    main()