"""
Task C.4 - Machine Learning 1
Training runner with CLI for running experiments.
"""

import os, sys
# --- Fix: make sure task2 is on sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "task2"))

import argparse, json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from data_utils import DataProcessor, calculate_metrics
from model_factory import build_model, model_summary_str

def parse_args():
    p = argparse.ArgumentParser(description="Task C.4 Trainer")
    # Data
    p.add_argument("--company", default="META")
    p.add_argument("--price_col", default="Close")
    p.add_argument("--train_start", default="2020-01-01")
    p.add_argument("--train_end", default="2023-08-01")
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--split_method", choices=["date", "random"], default="date")
    p.add_argument("--test_size", type=float, default=0.2)

    # Model / training
    p.add_argument("--model", choices=["lstm", "gru", "rnn"], default="lstm")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--recurrent_dropout", type=float, default=0.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--dense_units", type=int, default=0)
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "rmsprop"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)

    # Callbacks
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--reduce_lr", action="store_true")

    # Grid
    p.add_argument("--grid", action="store_true")
    p.add_argument("--models", nargs="+", default=["lstm", "gru", "rnn"])
    p.add_argument("--layers_list", nargs="+", type=int, default=[2, 3])
    p.add_argument("--units_list", nargs="+", type=int, default=[32, 64])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.1, 0.2])

    # Output
    p.add_argument("--out_root", default="task4_results")
    return p.parse_args()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_and_prepare(args):
    dp = DataProcessor()
    data = dp.load_stock_data(args.company, args.train_start, args.train_end, save_local=True)
    data = dp.handle_missing_data(data, method="drop")
    scaled, scaler = dp.scale_data(
        data, args.price_col,
        save_scaler=True,
        scaler_name=f"{args.company}_{args.train_start}_{args.train_end}"
    )
    X, y = dp.prepare_sequences(scaled, args.seq_len)
    X_tr, X_val, y_tr, y_val = dp.split_data(X, y, test_size=args.test_size, method=args.split_method)
    return dp, scaler, (X_tr, X_val, y_tr, y_val), data

def train_once(args, run_dir: Path):
    ensure_dir(run_dir)
    dp, scaler, (X_tr, X_val, y_tr, y_val), data = load_and_prepare(args)

    model = build_model(
        args.model, (args.seq_len, 1),
        units=args.units, n_layers=args.layers,
        dropout=args.dropout, bidirectional=args.bidirectional,
        recurrent_dropout=args.recurrent_dropout,
        l2=(args.l2 if args.l2 > 0 else None),
        dense_units=(args.dense_units if args.dense_units > 0 else None),
        loss=args.loss, optimizer=args.optimizer, learning_rate=args.lr
    )

    # Callbacks
    cbs = []
    if args.early_stop:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True))
    if args.reduce_lr:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), verbose=1))
    ckpt_path = run_dir / "best.weights.h5"
    cbs.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path), save_weights_only=True,
        monitor="val_loss", save_best_only=True))

    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch_size,
                     verbose=1, callbacks=cbs)

    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))

    # Predictions
    pred_val = model.predict(X_val, verbose=0).reshape(-1, 1)
    true_val = y_val.reshape(-1, 1)
    pred_val = scaler.inverse_transform(pred_val).ravel()
    true_val = scaler.inverse_transform(true_val).ravel()
    metrics = calculate_metrics(true_val, pred_val)

    # Save artifacts
    model.save(run_dir / f"{args.company}_{args.model}.keras")
    pd.DataFrame({"actual": true_val, "predicted": pred_val}).to_csv(run_dir/"predictions.csv", index=False)
    with open(run_dir/"metrics.json", "w") as f: json.dump({k: float(v) for k,v in metrics.items()}, f, indent=2)
    with open(run_dir/"model_summary.txt", "w") as f: f.write(model_summary_str(model))
    with open(run_dir/"config.json", "w") as f: json.dump(vars(args), f, indent=2)

    # Training history plot
    plt.figure(); plt.plot(hist.history["loss"], label="train")
    if "val_loss" in hist.history: plt.plot(hist.history["val_loss"], label="val")
    plt.legend(); plt.title("Training History"); plt.savefig(run_dir/"training_history.png", dpi=200); plt.close()

    # Validation fit plot
    plt.figure(); plt.plot(true_val, label="Actual"); plt.plot(pred_val, label="Predicted")
    plt.legend(); plt.title(f"{args.company} Validation — {args.model.upper()}")
    plt.savefig(run_dir/"val_fit.png", dpi=200); plt.close()

    return dict(metrics=metrics, run_dir=str(run_dir))

def run_grid(args):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_root); ensure_dir(root)
    consolidated = root/"task4_experiments.csv"
    if not consolidated.exists():
        with open(consolidated, "w") as f:
            f.write("stamp,company,model,layers,units,dropout,MAE,RMSE,MAPE,run_dir\n")
    for m in args.models:
        for L in args.layers_list:
            for U in args.units_list:
                for D in args.dropouts:
                    cfg = argparse.Namespace(**vars(args)); cfg.model=m; cfg.layers=L; cfg.units=U; cfg.dropout=D
                    run_dir = Path(args.out_root)/f"{args.company}_{m}_L{L}_U{U}_D{D}_{stamp}"
                    res = train_once(cfg, run_dir)
                    line = [stamp,args.company,m,str(L),str(U),str(D),
                            f"{res['metrics']['MAE']:.4f}",f"{res['metrics']['RMSE']:.4f}",
                            f"{res['metrics']['MAPE']:.4f}",res["run_dir"]]
                    with open(consolidated,"a") as f: f.write(",".join(line)+"\n")
    print(f"[Grid] Results saved to {consolidated}")

def main():
    args = parse_args()
    random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    if args.grid: run_grid(args)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.out_root)/f"{args.company}_{args.model}_{stamp}"
        res = train_once(args, run_dir)
        print(json.dumps(res["metrics"], indent=2))
        print(f"Artifacts saved to: {run_dir}")

if __name__=="__main__": main()
