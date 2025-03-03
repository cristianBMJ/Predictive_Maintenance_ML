# model_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model performance using RMSE and R² score.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.

    Returns:
    dict: A dictionary containing RMSE and R² score.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'R2 Score': r2
    }

def plot_predictions(y_true, y_pred):
    """
    Plot the true values against the predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    plt.show()
