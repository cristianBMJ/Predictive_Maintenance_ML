# app.py 
from flask import Flask, request, jsonify
import joblib
import mlflow
import mlflow.pyfunc


app = Flask(__name__)

# Load your model (make sure the path is correct)
model_local = joblib.load("models/model_XGBRegressor_v1.0.0.joblib")

def load_model():
    # Set the tracking URI to the MLflow server
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Update with your tracking URI if needed
    # Get the best model based on the lowest RMSE
    runs = mlflow.search_runs( 
        order_by=["metrics.rmse ASC"], 
        experiment_names=["models_include_metric"],

        )
    best_run = runs.iloc[0]

    model_uri = f"runs:/{best_run.run_id}/model"  
    print(model_uri)
    # Adjust the artifact path if necessary
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model()
print('Model load')

@app.route('/')
def home():
    return "Welcome to the Predictive Maintenance API. Use the /predict endpoint to make predictions from mlflow model."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)  # Add this line to log incoming data
    features = data['features']
    print("print features:",  features)
       # Create a dictionary with the required feature names
    feature_names = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO', 'NOX', 'Year', 'TAT_TIT_Ratio']
    input_data = {name: int(value) if name == 'Year' else value for name, value in zip(feature_names, features)}

    #input_data = {name: value for name,value in zip(feature_names, features) }

    prediction = model.predict([input_data])
    return jsonify({'prediction': float(prediction[0])})  # Convert to float
if __name__ == '__main__':
    app.run( port=5001)