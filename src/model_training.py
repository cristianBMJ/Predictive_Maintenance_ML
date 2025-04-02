# model_training.py

import sys
import os
import logging

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow   
import mlflow.pytorch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import load_and_process_data  # Import the function

from models.neural_model import SimpleModel, RMSELoss

from model_evaluation import save_model, save_model_mlflow

data = load_and_process_data()


import os
print("Current Working Directory:", os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    """
    A class to train machine learning models for predicting turbine energy yield.

    Attributes:
        data_path (str): Path to the CSV file containing the training data.
        data (DataFrame): Loaded data from the CSV file.
        X (DataFrame): Features for training.
        y (Series): Target variable for training.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.
    """


    def __init__(self, data_path, target = 'TEY', name=''):
        """
        Initializes the ModelTrainer with the data path.

        Parameters:
            data_path (str): Path to the CSV file containing the training data.
        """
        logging.info("Initializing ModelTrainer with data path: %s", data_path)
        self.data = pd.read_csv(data_path)
        self.X = self.data.drop(columns=[target])
        self.y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.name = name 

    def train_random_forest(self):
        """
        Trains a Random Forest model and evaluates its performance.
        """
        logging.info("Training Random Forest model.")
        model = RandomForestRegressor(n_estimators=100, )
        model.fit(self.X_train, self.y_train)
        self.evaluate(model, name="RandomForestRegressor")
      

    def train_xgboost(self):
        """
        Trains a XGBRegressor model and evaluates its performance.
        """        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, )
        model.fit(self.X_train, self.y_train)
        self.evaluate(model, name="XGBRegressor")

    def evaluate(self, model, name):
        logging.info("Evaluating model: %s", name)
        y_pred = model.predict(self.X_test)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        logging.info('RMSE: %s', rmse)
        logging.info('R2 Score: %s', r2)
        print('RMSE: %s', rmse)
        print('R2 Score: %s', r2)
        # mlflow.log_metric(f"RMSE {name} ", rmse)
        # mlflow.log_metric(f"R2 Score {name} ", r2)
        input_example = self.X_train.loc[[0]]
        save_model(model,  rmse, y_true = self.y_test, y_pred =  y_pred, name=name ) # save model local
        save_model_mlflow(model,  rmse, y_true = self.y_test, y_pred = y_pred, name=name, input_example=input_example )


    def evaluate_nn(self, model, name):
        # Evaluate
        with torch.no_grad():
            y_pred = model(self.X_test).numpy()
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        print('RMSE:', rmse )
        print('R2 Score:', r2 )
        
        save_model_mlflow(model,  rmse, y_true = self.y_test, y_pred = y_pred, name=name,)


    def train_neural_network(self):

        # Separate features (X) and target (y) for train and test
        self.X_train = self.X_train.values  # Convert DataFrame to numpy array
        self.y_train = self.y_train.values
        self.X_test = self.X_test.values
        self.y_test = self.y_test.values

        # Convert numpy arrays to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

        # Initialize model
        model = SimpleModel(self.X_train.shape[1])  # Ensure SimpleModel is imported
        criterion = RMSELoss()  # Ensure RMSELoss is imported
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer


        # DataLoader
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            for batch_X, batch_y in train_loader:  # Iterate over batches
                optimizer.zero_grad()
                y_pred = model(batch_X)  # Forward pass
                loss = criterion(y_pred, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Log model with MLflow
        self.evaluate_nn(model, name="SimpleModel" )

# Usage
if __name__ == "__main__":
    # Set the tracking URI to the SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set the experiment name
    mlflow.set_experiment("models_include_metric")

    # mlflow.start_run()
    trainer = ModelTrainer("data/processed_data.csv")
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.train_neural_network()

    mlflow.end_run()
    print('\n Done')
#/home/cris/workaplace/Predictive_Maintenance_ML/data/processed_data.csv

