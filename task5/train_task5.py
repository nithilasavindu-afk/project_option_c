"""
Task C.5 - Machine Learning 2
Training runner with CLI for running experiments.
Enhanced for multivariate and multistep predictions with GPU/CPU selection.
"""

import os, sys
# --- Fix: make sure task5 is on sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import argparse, json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from data_utils import DataProcessor, calculate_metrics
from model_factory import build_model, model_summary_str

def interactive_menu():
    """Interactive menu for easy configuration selection."""
    print("\n" + "="*60)
    print("🎯 TASK 5 - Interactive Training Configuration")
    print("="*60)

    # Prediction Mode
    print("\n📊 PREDICTION MODE:")
    print("1. Single-step (predict next 1 value)")
    print("2. Multistep (predict next k values)")
    while True:
        try:
            mode_choice = input("Select mode (1-2): ").strip()
            if mode_choice == "1":
                multistep = False
                k = 1
                break
            elif mode_choice == "2":
                multistep = True
                print("\n🔮 MULTISTEP HORIZON (k):")
                print("1. k=3 (3 steps ahead)")
                print("2. k=5 (5 steps ahead)")
                print("3. k=7 (7 steps ahead)")
                print("4. k=10 (10 steps ahead)")
                print("5. Custom k value")
                while True:
                    k_choice = input("Select k value (1-5): ").strip()
                    if k_choice == "1":
                        k = 3
                        break
                    elif k_choice == "2":
                        k = 5
                        break
                    elif k_choice == "3":
                        k = 7
                        break
                    elif k_choice == "4":
                        k = 10
                        break
                    elif k_choice == "5":
                        while True:
                            try:
                                k = int(input("Enter custom k value (2-20): ").strip())
                                if 2 <= k <= 20:
                                    break
                                else:
                                    print("❌ k must be between 2-20")
                            except ValueError:
                                print("❌ Please enter a valid number")
                        break
                    else:
                        print("❌ Invalid choice. Please select 1-5.")
                break
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Features
    print("\n📈 FEATURES:")
    print("1. Univariate (Close price only)")
    print("2. Multivariate (OHLCV features)")
    while True:
        try:
            feature_choice = input("Select features (1-2): ").strip()
            if feature_choice == "1":
                multivariate = False
                break
            elif feature_choice == "2":
                multivariate = True
                break
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Model Type
    print("\n🤖 MODEL TYPE:")
    print("1. LSTM (Long Short-Term Memory)")
    print("2. GRU (Gated Recurrent Unit)")
    print("3. RNN (Simple Recurrent Neural Network)")
    while True:
        try:
            model_choice = input("Select model (1-3): ").strip()
            if model_choice == "1":
                model = "lstm"
                break
            elif model_choice == "2":
                model = "gru"
                break
            elif model_choice == "3":
                model = "rnn"
                break
            else:
                print("❌ Invalid choice. Please select 1-3.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Training Parameters
    print("\n⚙️  TRAINING PARAMETERS:")

    # Epochs
    while True:
        try:
            epochs = int(input("Enter number of epochs (1-100): ").strip())
            if 1 <= epochs <= 100:
                break
            else:
                print("❌ Epochs must be between 1-100")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Layers
    while True:
        try:
            layers = int(input("Enter number of layers (1-5): ").strip())
            if 1 <= layers <= 5:
                break
            else:
                print("❌ Layers must be between 1-5")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Units
    while True:
        try:
            units = int(input("Enter units per layer (16-256): ").strip())
            if 16 <= units <= 256:
                break
            else:
                print("❌ Units must be between 16-256")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Dropout
    while True:
        try:
            dropout = float(input("Enter dropout rate (0.0-0.5): ").strip())
            if 0.0 <= dropout <= 0.5:
                break
            else:
                print("❌ Dropout must be between 0.0-0.5")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Device
    print("\n💻 DEVICE:")
    print("1. GPU (if available, faster)")
    print("2. CPU (always available)")
    while True:
        try:
            device_choice = input("Select device (1-2): ").strip()
            if device_choice == "1":
                device = "gpu"
                break
            elif device_choice == "2":
                device = "cpu"
                break
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Grid Search Option
    print("\n🔍 TRAINING MODE:")
    print("1. Single training run")
    print("2. Grid search (test multiple configurations)")
    while True:
        try:
            grid_choice = input("Select mode (1-2): ").strip()
            if grid_choice == "1":
                grid = False
                break
            elif grid_choice == "2":
                grid = True
                break
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    # Summary
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY:")
    print("="*60)
    print(f"Mode: {'Multistep' if multistep else 'Single-step'}")
    if multistep:
        print(f"k value: {k}")
    print(f"Features: {'Multivariate (OHLCV)' if multivariate else 'Univariate (Close)'}")
    print(f"Model: {model.upper()}")
    print(f"Layers: {layers}, Units: {units}, Dropout: {dropout}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device.upper()}")
    print(f"Training: {'Grid Search' if grid else 'Single Run'}")

    # Confirm
    while True:
        try:
            confirm = input("\n🚀 Start training? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                break
            elif confirm in ['n', 'no']:
                print("❌ Training cancelled.")
                exit(0)
            else:
                print("❌ Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)

    return {
        'multistep': multistep,
        'k': k,
        'multivariate': multivariate,
        'model': model,
        'epochs': epochs,
        'layers': layers,
        'units': units,
        'dropout': dropout,
        'device': device,
        'grid': grid
    }

def parse_args():
    p = argparse.ArgumentParser(description="Task C.5 Trainer")
    # Data
    p.add_argument("--company", default="META")
    p.add_argument("--price_col", default="Close")
    p.add_argument("--train_start", default="2020-01-01")
    p.add_argument("--train_end", default="2023-08-01")
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--split_method", choices=["date", "random"], default="date")
    p.add_argument("--test_size", type=float, default=0.2)

    # Task 5 specific: multistep and multivariate
    p.add_argument("--multistep", action="store_true", help="Enable multistep prediction")
    p.add_argument("--k", type=int, default=5, help="Prediction horizon for multistep (k steps ahead)")
    p.add_argument("--multivariate", action="store_true", help="Use multivariate features (OHLCV) instead of just Close")

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

    # Device selection
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                   help="Device to use for training (gpu preferred, falls back to cpu)")

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
    p.add_argument("--out_root", default="task5_results")
    return p.parse_args()

def setup_device(device_preference):
    """Setup TensorFlow to use GPU or CPU based on preference and availability."""
    if device_preference == "gpu":
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Using GPU: {gpus}")
                return "GPU"
            except RuntimeError as e:
                print(f"❌ GPU setup failed: {e}")
                print("🔄 Falling back to CPU")
        else:
            print("❌ No GPU devices found, using CPU")
    else:
        print("🔄 Using CPU as requested")

    # CPU-only setup
    tf.config.set_visible_devices([], 'GPU')  # Hide GPUs
    print("✅ Using CPU")
    return "CPU"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_and_prepare(args):
    dp = DataProcessor()
    data = dp.load_stock_data(args.company, args.train_start, args.train_end, save_local=True)
    data = dp.handle_missing_data(data, method="drop")

    if args.multivariate:
        # Use all OHLCV features for multivariate prediction
        features = ["Open", "High", "Low", "Close", "Volume"]
        scaled, scaler = dp.scale_data_multivariate(
            data, features,
            save_scaler=True,
            scaler_name=f"{args.company}_{args.train_start}_{args.train_end}_multivariate"
        )
        if args.multistep:
            # Multivariate multistep
            features = ["Open", "High", "Low", "Close", "Volume"]
            X, y = dp.prepare_multivariate_sequences(scaled, features, args.seq_len, args.k)
            input_shape = (args.seq_len, len(features))
            prediction_horizon = args.k
        else:
            # Multivariate single-step
            features = ["Open", "High", "Low", "Close", "Volume"]
            X, y = dp.prepare_multivariate_sequences(scaled, features, args.seq_len, prediction_horizon=1)
            input_shape = (args.seq_len, len(features))
            prediction_horizon = 1
    else:
        # Univariate (traditional)
        scaled, scaler = dp.scale_data(
            data, args.price_col,
            save_scaler=True,
            scaler_name=f"{args.company}_{args.train_start}_{args.train_end}"
        )
        if args.multistep:
            # Univariate multistep
            X, y = dp.prepare_multistep_sequences(scaled, args.seq_len, args.k)
            input_shape = (args.seq_len, 1)
            prediction_horizon = args.k
        else:
            # Univariate single-step (original)
            X, y = dp.prepare_multistep_sequences(scaled, args.seq_len, prediction_horizon=1)
            input_shape = (args.seq_len, 1)
            prediction_horizon = 1

    X_tr, X_val, y_tr, y_val = dp.split_data(X, y, test_size=args.test_size, method=args.split_method)
    return dp, scaler, (X_tr, X_val, y_tr, y_val), data, input_shape, prediction_horizon

def train_once(args, run_dir: Path):
    ensure_dir(run_dir)
    dp, scaler, (X_tr, X_val, y_tr, y_val), data, input_shape, prediction_horizon = load_and_prepare(args)

    model = build_model(
        args.model, input_shape,
        units=args.units, n_layers=args.layers,
        dropout=args.dropout, bidirectional=args.bidirectional,
        recurrent_dropout=args.recurrent_dropout,
        l2=(args.l2 if args.l2 > 0 else None),
        dense_units=(args.dense_units if args.dense_units > 0 else None),
        loss=args.loss, optimizer=args.optimizer, learning_rate=args.lr,
        prediction_horizon=prediction_horizon  # NEW: For multistep
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
    pred_val = model.predict(X_val, verbose=0)
    true_val = y_val

    if prediction_horizon > 1:
        # Multistep: predictions are (samples, k), targets are (samples, k)
        # For metrics, we'll use the last step prediction vs last step target
        pred_val_flat = pred_val[:, -1].reshape(-1, 1)  # Last step of each sequence
        true_val_flat = true_val[:, -1].reshape(-1, 1)
        if args.multivariate:
            # For multivariate, create dummy array for inverse transform
            pred_dummy = np.zeros((len(pred_val_flat), 5))  # 5 features
            pred_dummy[:, 3] = pred_val_flat.ravel()  # Close is index 3
            true_dummy = np.zeros((len(true_val_flat), 5))
            true_dummy[:, 3] = true_val_flat.ravel()
            pred_val_plot = scaler.inverse_transform(pred_dummy)[:, 3]  # Extract Close
            true_val_plot = scaler.inverse_transform(true_dummy)[:, 3]
        else:
            pred_val_plot = scaler.inverse_transform(pred_val_flat).ravel()
            true_val_plot = scaler.inverse_transform(true_val_flat).ravel()
    else:
        # Single-step: predictions are (samples, 1), targets are (samples, 1)
        pred_val_flat = pred_val.reshape(-1, 1)
        true_val_flat = true_val.reshape(-1, 1)
        if args.multivariate:
            # For multivariate, create dummy array for inverse transform
            pred_dummy = np.zeros((len(pred_val_flat), 5))  # 5 features
            pred_dummy[:, 3] = pred_val_flat.ravel()  # Close is index 3
            true_dummy = np.zeros((len(true_val_flat), 5))
            true_dummy[:, 3] = true_val_flat.ravel()
            pred_val_plot = scaler.inverse_transform(pred_dummy)[:, 3]  # Extract Close
            true_val_plot = scaler.inverse_transform(true_dummy)[:, 3]
        else:
            pred_val_plot = scaler.inverse_transform(pred_val_flat).ravel()
            true_val_plot = scaler.inverse_transform(true_val_flat).ravel()

    metrics = calculate_metrics(true_val_plot, pred_val_plot)

    # Save artifacts
    model.save(run_dir / f"{args.company}_{args.model}.keras")

    if prediction_horizon > 1:
        # For multistep, save all prediction steps
        pred_df = pd.DataFrame(pred_val, columns=[f"pred_step_{i+1}" for i in range(prediction_horizon)])
        true_df = pd.DataFrame(true_val, columns=[f"true_step_{i+1}" for i in range(prediction_horizon)])
        combined_df = pd.concat([true_df, pred_df], axis=1)
        combined_df.to_csv(run_dir/"predictions.csv", index=False)
    else:
        # Single-step
        pd.DataFrame({"actual": true_val_plot, "predicted": pred_val_plot}).to_csv(run_dir/"predictions.csv", index=False)
    with open(run_dir/"metrics.json", "w") as f: json.dump({k: float(v) for k,v in metrics.items()}, f, indent=2)
    with open(run_dir/"model_summary.txt", "w", encoding="utf-8") as f: f.write(model_summary_str(model))
    with open(run_dir/"config.json", "w") as f: json.dump(vars(args), f, indent=2)

    # Training history plot
    plt.figure(); plt.plot(hist.history["loss"], label="train")
    if "val_loss" in hist.history: plt.plot(hist.history["val_loss"], label="val")
    plt.legend(); plt.title("Training History"); plt.savefig(run_dir/"training_history.png", dpi=200); plt.close()

    # Validation fit plot
    plt.figure(); plt.plot(true_val_plot, label="Actual"); plt.plot(pred_val_plot, label="Predicted")
    title_suffix = f"Multistep (k={prediction_horizon})" if prediction_horizon > 1 else "Single-step"
    feature_suffix = "Multivariate" if args.multivariate else "Univariate"
    plt.legend(); plt.title(f"{args.company} Validation — {args.model.upper()} {feature_suffix} {title_suffix}")
    plt.savefig(run_dir/"val_fit.png", dpi=200); plt.close()

    return dict(metrics=metrics, run_dir=str(run_dir))

def run_grid(args):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_root); ensure_dir(root)
    consolidated = root/"task5_experiments.csv"
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
    # Check if any arguments provided (excluding script name)
    if len(sys.argv) == 1:
        # Interactive mode
        print("🤖 No command-line arguments provided. Starting interactive mode...")
        config = interactive_menu()

        # Create args namespace from interactive config
        args = argparse.Namespace(
            company="META",
            price_col="Close",
            train_start="2020-01-01",
            train_end="2023-08-01",
            seq_len=60,
            split_method="date",
            test_size=0.2,
            multistep=config['multistep'],
            k=config['k'],
            multivariate=config['multivariate'],
            model=config['model'],
            layers=config['layers'],
            units=config['units'],
            dropout=config['dropout'],
            bidirectional=False,
            recurrent_dropout=0.0,
            l2=0.0,
            dense_units=0,
            loss="mse",
            optimizer="adam",
            lr=1e-3,
            epochs=config['epochs'],
            batch_size=32,
            device=config['device'],
            early_stop=False,
            patience=5,
            reduce_lr=False,
            grid=config['grid'],
            models=["lstm", "gru", "rnn"],
            layers_list=[2, 3],
            units_list=[32, 64],
            dropouts=[0.1, 0.2],
            out_root="task5_results"
        )
    else:
        # Command-line mode
        args = parse_args()

    # Setup device (GPU preferred, CPU fallback)
    device_used = setup_device(args.device)

    random.seed(42); np.random.seed(42); tf.random.set_seed(42)

    # Print Task 5 mode information
    mode_info = []
    if args.multistep:
        mode_info.append(f"Multistep (k={args.k})")
    else:
        mode_info.append("Single-step")
    if args.multivariate:
        mode_info.append("Multivariate (OHLCV)")
    else:
        mode_info.append("Univariate (Close only)")
    print(f"🔧 Task 5 Mode: {' + '.join(mode_info)}")

    if args.grid: run_grid(args)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.out_root)/f"{args.company}_{args.model}_{stamp}"
        res = train_once(args, run_dir)
        print(f"Training completed on {device_used}")
        print(json.dumps(res["metrics"], indent=2))
        print(f"Artifacts saved to: {run_dir}")

if __name__=="__main__": main()