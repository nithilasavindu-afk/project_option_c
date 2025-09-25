"""
Task C.6 - Evaluation and Visualization
Evaluates individual models and ensemble, generates plots and reports.
"""

import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

def plot_predictions(actual, predictions_dict, ensemble_pred, save_path):
    """Plot actual vs predicted for all models and ensemble."""
    plt.figure(figsize=(15, 10))

    # Plot actual
    plt.plot(actual, label='Actual', linewidth=2, color='black')

    # Plot individual models
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    for i, (model_name, pred) in enumerate(predictions_dict.items()):
        plt.plot(pred, label=f'{model_name.upper()}', alpha=0.7, color=colors[i % len(colors)])

    # Plot ensemble
    plt.plot(ensemble_pred, label='Ensemble', linewidth=3, color='red', linestyle='--')

    plt.legend()
    plt.title('Task 6: Ensemble Model Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path):
    """Plot bar chart comparing MAE, RMSE, MAPE across models."""
    models = list(metrics_dict.keys())
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    mape_values = [metrics_dict[m]['MAPE'] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MAE
    axes[0].bar(models, mae_values, color='skyblue')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)

    # RMSE
    axes[1].bar(models, rmse_values, color='lightgreen')
    axes[1].set_title('Root Mean Square Error (RMSE)')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)

    # MAPE
    axes[2].bar(models, mape_values, color='lightcoral')
    axes[2].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results_dir, metrics_dict, ensemble_metrics):
    """Generate a summary report."""
    report_path = results_dir / "task6_report.txt"

    with open(report_path, 'w') as f:
        f.write("Task 6: Ensemble Modeling Report\n")
        f.write("=" * 40 + "\n\n")

        f.write("Individual Model Performance:\n")
        f.write("-" * 30 + "\n")
        for model, mets in metrics_dict.items():
            f.write(f"{model.upper()}:\n")
            f.write(".4f")
            f.write(".4f")
            f.write(".2f")
            f.write("\n")

        f.write("\nEnsemble Performance:\n")
        f.write("-" * 20 + "\n")
        f.write(".4f")
        f.write(".4f")
        f.write(".2f")

        f.write("\n\nConclusion:\n")
        f.write("The ensemble combines predictions from LSTM, GRU, RNN, ARIMA, SARIMA, and Random Forest.\n")
        f.write("Simple averaging is used to combine predictions, potentially improving accuracy over individual models.\n")

    print(f"Report saved to {report_path}")

def evaluate_ensemble(actual, predictions_dict, ensemble_pred, results_dir):
    """Main evaluation function."""
    # Collect all metrics
    all_metrics = predictions_dict.copy()
    all_metrics['Ensemble'] = ensemble_pred  # Wait, no, ensemble_pred is the predictions, not metrics

    # Actually, we need metrics for ensemble too
    # Assuming calculate_metrics is imported
    from data_utils import calculate_metrics
    ensemble_metrics = calculate_metrics(actual, ensemble_pred)

    # Plot predictions
    plot_predictions(actual, predictions_dict, ensemble_pred, results_dir / "predictions_comparison.png")

    # Plot metrics (need to compute metrics for each)
    metrics_dict = {}
    for model, pred in predictions_dict.items():
        metrics_dict[model] = calculate_metrics(actual, pred)
    metrics_dict['Ensemble'] = ensemble_metrics

    plot_metrics_comparison(metrics_dict, results_dir / "metrics_comparison.png")

    # Generate report
    generate_report(results_dir, metrics_dict, ensemble_metrics)

    return metrics_dict, ensemble_metrics

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pandas as pd
    import json

    parser = argparse.ArgumentParser(description="Evaluate Task 6 Ensemble Results")
    parser.add_argument("--results_dir", default="task6_results", help="Directory containing results")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results directory {results_root} not found.")
        exit(1)

    # Find the latest ensemble folder
    ensemble_dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("ensemble_")]
    if not ensemble_dirs:
        print("No ensemble results found.")
        exit(1)
    latest_dir = max(ensemble_dirs, key=lambda d: d.stat().st_mtime)

    print(f"Evaluating results from: {latest_dir}")

    # Load data
    pred_df = pd.read_csv(latest_dir / "predictions.csv")
    actual = pred_df["actual"].values
    ensemble_pred = pred_df["ensemble_predicted"].values

    # Load individual model predictions (assuming saved separately, but for now, skip or load from subdirs)
    # For simplicity, since individual preds are in subdirs, we can load them
    predictions_dict = {}
    for model_dir in results_root.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("ensemble_"):
            model_name = model_dir.name.split("_")[1]  # e.g., META_lstm_... -> lstm
            if (model_dir / "predictions.csv").exists():
                model_pred_df = pd.read_csv(model_dir / "predictions.csv")
                predictions_dict[model_name] = model_pred_df["predicted"].values

    if not predictions_dict:
        print("No individual model predictions found. Run trainer first.")
        exit(1)

    # Evaluate
    metrics, ens_metrics = evaluate_ensemble(actual, predictions_dict, ensemble_pred, latest_dir)

    print("Evaluation complete. Check plots and report in results folder.")
    print(f"Ensemble MAE: {ens_metrics['MAE']:.4f}, RMSE: {ens_metrics['RMSE']:.4f}")