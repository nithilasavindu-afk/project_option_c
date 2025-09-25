"""
Task C.6 - Ensemble Methods
Implements ensemble averaging for combining model predictions.
"""

import numpy as np

def ensemble_average(predictions_list):
    """
    Simple averaging of predictions from multiple models.

    Args:
        predictions_list: List of numpy arrays, each containing predictions from one model

    Returns:
        Averaged predictions as numpy array
    """
    if not predictions_list:
        raise ValueError("No predictions provided")

    # Ensure all predictions have the same length
    lengths = [len(pred) for pred in predictions_list]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        predictions_list = [pred[:min_len] for pred in predictions_list]

    # Stack and average
    stacked = np.stack(predictions_list, axis=0)
    averaged = np.mean(stacked, axis=0)

    return averaged

def ensemble_weighted_average(predictions_list, weights=None):
    """
    Weighted averaging of predictions.

    Args:
        predictions_list: List of prediction arrays
        weights: List of weights for each model (must sum to 1)

    Returns:
        Weighted average predictions
    """
    if weights is None:
        # Equal weights
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    if len(weights) != len(predictions_list):
        raise ValueError("Number of weights must match number of predictions")

    # Ensure same length
    lengths = [len(pred) for pred in predictions_list]
    min_len = min(lengths)
    predictions_list = [pred[:min_len] for pred in predictions_list]

    # Weighted average
    weighted_sum = np.zeros(min_len)
    for pred, weight in zip(predictions_list, weights):
        weighted_sum += pred * weight

    return weighted_sum