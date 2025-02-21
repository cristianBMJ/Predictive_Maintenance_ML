# model_training.py

import mlflow
import mlflow.pytorch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_preprocessing import load_and_process_data  # Import the function


data = load_and_process_data()

import os
print("Current Working Directory:", os.getcwd())

class ModelTrainer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = self.data.drop(columns=['TEY'])
        self.y = self.data['TEY']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.evaluate(model)

    def train_xgboost(self):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.evaluate(model)

    def evaluate(self, model):
        y_pred = model.predict(self.X_test)
        print('RMSE:', mean_squared_error(self.y_test, y_pred, squared=False))
        print('R2 Score:', r2_score(self.y_test, y_pred))

    def train_neural_network(self):
        # ... existing neural network training code ...
        # Initialize model
        model = SimpleModel(self.X_train.shape[1])
        # ... existing training loop ...
        # Log model with MLflow
        mlflow.pytorch.log_model(model, "neural_network_model")

# Usage
if __name__ == "__main__":
    mlflow.start_run()
    trainer = ModelTrainer("data/processed_data.csv")
    trainer.train_random_forest()
    trainer.train_xgboost()
    # trainer.train_neural_network()
    mlflow.end_run()
    print('\n Done')
#/home/cris/workaplace/Predictive_Maintenance_ML/data/processed_data.csv