"""
Task C.4 - Evaluate trained model on test data
"""

import os, sys
# --- Fix: make sure task2 is on sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "task2"))

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from data_utils import DataProcessor, calculate_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--company", default="META")
    p.add_argument("--model_path", required=True)
    p.add_argument("--price_col", default="Close")
    p.add_argument("--train_start", default="2020-01-01")
    p.add_argument("--train_end", default="2023-08-01")
    p.add_argument("--test_start", default="2023-08-02")
    p.add_argument("--test_end", default="2024-07-02")
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--out_dir", default="task4_test_eval")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dp = DataProcessor()
    model = tf.keras.models.load_model(args.model_path)

    # Load scaler saved during training
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

    # Prepare X_test sequences
    x_test = dp.prepare_test_data(train, test, scaler, args.seq_len, args.price_col)

    # ---- FIX START: ensure 1-D arrays and align lengths ----
    # Predict -> inverse transform -> flatten to 1-D
    pred_scaled = model.predict(x_test, verbose=0)              # (N, 1) or (N,)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)) # ensure 2-D for scaler
    pred = np.asarray(pred, dtype=float).reshape(-1)            # -> 1-D

    # Actual prices from test set -> 1-D
    actual = test[args.price_col].to_numpy(dtype=float).reshape(-1)

    # Align lengths defensively (in case of off-by-one)
    n = min(len(actual), len(pred))
    actual = actual[-n:]
    pred = pred[-n:]
    # ---- FIX END ----

    # Metrics
    metrics = calculate_metrics(actual, pred)
    with open(out / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Save predictions CSV
    pd.DataFrame({"actual": actual, "predicted": pred}).to_csv(out / "predictions.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", linewidth=2)
    plt.plot(pred, label="Predicted", linewidth=2)
    plt.title(f"{args.company} Test Fit ({args.test_start}→{args.test_end})")
    plt.xlabel("Time"); plt.ylabel("Price ($)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "test_fit.png", dpi=200); plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out}")

if __name__ == "__main__":
    main()
