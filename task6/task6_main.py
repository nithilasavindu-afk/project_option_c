"""
Task 6 - Complete Workflow Script
Runs training, ensembling, and evaluation in one go.
Asks user for epochs input.
"""

import subprocess
import sys
import os

def main():
    print("Task 6: Ensemble Modeling Workflow")
    print("=" * 40)

    # Get user input for epochs
    try:
        epochs = int(input("Enter number of epochs for training (default 25): ") or 25)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Using default epochs = 25")
        epochs = 25

    print(f"Training with {epochs} epochs...")

    # Run training
    train_cmd = [
        "python", "ensemble_trainer.py",
        "--company", "META",
        "--models", "lstm", "gru", "rnn",
        "--epochs", str(epochs)
    ]

    try:
        result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(e.stderr)
        return

    # Run evaluation
    print("\nRunning evaluation...")
    eval_cmd = ["python", "evaluate_task6.py"]

    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        print(e.stderr)
        return

    print("\nWorkflow complete! Check task6_results/ for outputs.")

if __name__ == "__main__":
    main()