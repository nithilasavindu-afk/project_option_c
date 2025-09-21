"""
Task C.5 - Evaluate trained model on test data
Enhanced with automatic best model selection from experiments.
"""

import os, sys
# --- Fix: make sure task5 is on sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from data_utils import DataProcessor, calculate_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--company", default="META")
    p.add_argument("--model_path", help="Path to specific model file (.keras)")
    p.add_argument("--auto_select_best", action="store_true",
                   help="Automatically select best model from task5_experiments.csv")
    p.add_argument("--experiments_csv", default="task5_results/task5_experiments.csv",
                   help="Path to experiments CSV file")
    p.add_argument("--metric", choices=["MAE", "RMSE", "MAPE"], default="MAE",
                   help="Metric to use for selecting best model (lower is better)")
    p.add_argument("--price_col", default="Close")
    p.add_argument("--train_start", default="2020-01-01")
    p.add_argument("--train_end", default="2023-08-01")
    p.add_argument("--test_start", default="2023-08-02")
    p.add_argument("--test_end", default="2024-07-02")
    p.add_argument("--seq_len", type=int, default=60)

    # Task 5 specific: multistep and multivariate
    p.add_argument("--multistep", action="store_true", help="Enable multistep prediction")
    p.add_argument("--k", type=int, default=5, help="Prediction horizon for multistep (k steps ahead)")
    p.add_argument("--multivariate", action="store_true", help="Use multivariate features (OHLCV) instead of just Close")

    p.add_argument("--out_dir", default="task5_test_eval")
    return p.parse_args()

def find_best_model(experiments_csv, metric):
    """Find the best model based on the specified metric."""
    if not Path(experiments_csv).exists():
        raise FileNotFoundError(f"Experiments CSV not found: {experiments_csv}")

    df = pd.read_csv(experiments_csv)

    # Find row with minimum value for the specified metric
    best_idx = df[metric].idxmin()
    best_row = df.loc[best_idx]

    model_dir = best_row['run_dir']
    model_path = Path(model_dir) / f"{best_row['company']}_{best_row['model']}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model file not found: {model_path}")

    print(f"✅ Selected best model based on {metric}:")
    print(f"   Model: {best_row['model']} (L{best_row['layers']}, U{best_row['units']}, D{best_row['dropout']})")
    print(f"   {metric}: {best_row[metric]:.4f}")
    print(f"   Path: {model_path}")

    return str(model_path)

def main():
    args = parse_args()

    # Determine model path
    if args.auto_select_best:
        if args.model_path:
            print("⚠️  Warning: --model_path ignored when using --auto_select_best")
        model_path = find_best_model(args.experiments_csv, args.metric)
    elif args.model_path:
        model_path = args.model_path
    else:
        raise ValueError("Must specify either --model_path or --auto_select_best")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dp = DataProcessor()
    model = tf.keras.models.load_model(model_path)

    # Load appropriate scaler
    if args.multivariate:
        scaler_name = f"{args.company}_{args.train_start}_{args.train_end}_multivariate"
    else:
        scaler_name = f"{args.company}_{args.train_start}_{args.train_end}"
    scaler = dp.load_scaler(scaler_name)
    if scaler is None:
        raise FileNotFoundError("Scaler not found. Re-run training first.")

    # Load/clean data
    train = dp.handle_missing_data(
        dp.load_stock_data(args.company, args.train_start, args.train_end, save_local=True)
    )
    test = dp.handle_missing_data(
        dp.load_stock_data(args.company, args.test_start, args.test_end, save_local=True)
    )

    # Prepare test data based on mode
    if args.multivariate:
        features = ["Open", "High", "Low", "Close", "Volume"]
        if args.multistep:
            x_test, y_test = dp.prepare_test_multivariate_multistep_data(train, test, features, args.seq_len, args.k)
        else:
            x_test, y_test = dp.prepare_test_multivariate_data(train, test, features, args.seq_len)
        
        # For multivariate models, scale the test inputs (model was trained on scaled data)
        x_test_scaled = np.zeros_like(x_test, dtype=float)
        for i in range(x_test.shape[0]):  # For each sequence
            for j in range(x_test.shape[1]):  # For each time step
                # Scale each time step (5 features) using the multivariate scaler
                x_test_scaled[i, j] = scaler.transform(x_test[i, j].reshape(1, -1))
        x_test = x_test_scaled
    else:
        if args.multistep:
            x_test, y_test = dp.prepare_test_multistep_data(train, test, scaler, args.seq_len, args.k, args.price_col)
        else:
            x_test = dp.prepare_test_data(train, test, scaler, args.seq_len, args.price_col)
            y_test = test[args.price_col].to_numpy(dtype=float).reshape(-1)

    # ---- PREDICTION AND METRICS ----
    pred_scaled = model.predict(x_test, verbose=0)

    if args.multistep:
        # Multistep predictions: (samples, k)
        if args.multivariate:
            # For multivariate multistep, y_test is already prepared (unscaled)
            actual = y_test[:, -1]  # Last step of each sequence
            pred = pred_scaled[:, -1]  # Last step prediction (scaled)
        else:
            # For univariate multistep, y_test is prepared as sequences
            actual = y_test[:, -1]  # Last step of each sequence
            pred = pred_scaled[:, -1]  # Last step prediction

        # Inverse transform predictions only (actuals are already unscaled for multivariate)
        if not args.multivariate:
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
            actual = scaler.inverse_transform(actual.reshape(-1, 1)).reshape(-1)
        # For multivariate, inverse transform predictions only (actuals are unscaled)
        else:
            # Create dummy array for inverse transform of predictions
            pred_dummy = np.zeros((len(pred), 5))  # 5 features: Open, High, Low, Close, Volume
            pred_dummy[:, 3] = pred  # Close is index 3
            # Inverse transform and extract Close column
            pred = scaler.inverse_transform(pred_dummy)[:, 3]
            # actual is already unscaled from y_test
    else:
        # Single-step predictions
        if args.multivariate:
            # y_test is prepared as single values (unscaled)
            actual = y_test
            pred = pred_scaled.reshape(-1)  # Predictions are scaled
            # For multivariate, inverse transform predictions only
            pred_dummy = np.zeros((len(pred), 5))  # 5 features
            pred_dummy[:, 3] = pred  # Close is index 3
            # Inverse transform and extract Close column
            pred = scaler.inverse_transform(pred_dummy)[:, 3]
            # actual is already unscaled from y_test
        else:
            # Original univariate single-step
            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
            actual = test[args.price_col].to_numpy(dtype=float).reshape(-1)

            # Align lengths defensively
            n = min(len(actual), len(pred))
            actual = actual[-n:]
            pred = pred[-n:]

    # Calculate metrics
    metrics = calculate_metrics(actual, pred)
    with open(out / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Save predictions CSV
    pd.DataFrame({"actual": actual, "predicted": pred}).to_csv(out / "predictions.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", linewidth=2)
    plt.plot(pred, label="Predicted", linewidth=2)

    # Build title with mode information
    mode_parts = []
    if args.multistep:
        mode_parts.append(f"Multistep (k={args.k})")
    else:
        mode_parts.append("Single-step")
    if args.multivariate:
        mode_parts.append("Multivariate")
    else:
        mode_parts.append("Univariate")

    title_suffix = " (Best Model)" if args.auto_select_best else ""
    plt.title(f"{args.company} Test Fit ({args.test_start}→{args.test_end}) {' + '.join(mode_parts)}{title_suffix}")
    plt.xlabel("Time"); plt.ylabel("Price ($)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "test_fit.png", dpi=200); plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out}")

if __name__ == "__main__":
    main()