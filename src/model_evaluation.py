# model_evaluation.py


import joblib
import os
import json 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import torch
import mlflow.pytorch 

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

# save local model 
def save_model(model, rmse, model_version="1.0.0", y_true=None, y_pred=None, name="x"):
    """
    Save the model if it has a better RMSE than the previous model.

    Parameters:
    model: The trained model to save.
    rmse (float): The RMSE of the current model.
    model_version (str): The version of the model.
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values.
    name (str): Name identifier for the model.
    """
    previous_rmse = float('inf')  # Initialize to a high value
    model_filename = f"models/model_{name}_v{model_version}.joblib"
    metadata_filename = f"models/model_{name}_v{model_version}_metadata.json"

    # Ensure the "models" directory exists
    os.makedirs("models", exist_ok=True)

    # Check if a previous model exists and load metadata
    if os.path.exists(metadata_filename):
        with open(metadata_filename, "r") as f:
            previous_metadata = json.load(f)
            previous_rmse = previous_metadata.get("metrics", {}).get("rmse", float('inf'))

    # Save the model if the current RMSE is better
    if rmse < previous_rmse:
        joblib.dump(model, model_filename)
        
        # Save metadata
        metadata = {
            "version": model_version,
            "name": name,
            "metrics": {
                "rmse": rmse,
                "r2_score": r2_score(y_true, y_pred) if y_true is not None and y_pred is not None else None
            }
        }
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f)

        print(f"✅ Model saved as {model_filename} with RMSE: {rmse}")
    else:
        print(f"❌ Model not saved. Current RMSE: {rmse} is not better than previous RMSE: {previous_rmse}.")

# save model in mlflow
def save_model_mlflow(model, rmse, model_version="1.0.0", y_true=None, y_pred=None, name="x", input_example= None):
    """
    Save the model using MLflow if it has a better RMSE than the previous version.

    Parameters:
    model: The trained model to save.
    rmse (float): The RMSE of the current model.
    model_version (str): The version of the model.
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values.
    name (str): Name identifier for the model.
    """
    # mlflow.set_experiment(f"{name}_model_tracking")

    with mlflow.start_run( run_name=name):
        # Log metrics
        mlflow.set_tag("model_name", name)
        mlflow.log_param("version", model_version)
       
        
        if y_true is not None and y_pred is not None:
            r2 = r2_score(y_true, y_pred)
            mlflow.log_metrics(
                {"r2-score": r2,
                "rmse": rmse
                }
            )

        
        # Log model
        if isinstance(model, torch.nn.Module):  # Check if the model is a PyTorch model
            mlflow.pytorch.log_model(model,
                                      artifact_path="model",
                                      input_example=input_example,
                                      registered_model_name=name)
        else:
            mlflow.sklearn.log_model(model,
                                     artifact_path="model",
                                     input_example=input_example,
                                     registered_model_name=name)
#        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

        print(f"✅ Model saved in MLflow with RMSE: {rmse}")

