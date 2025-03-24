# app.py 
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your model (make sure the path is correct)
model = joblib.load("models/model_XGBRegressor_v1.0.0.joblib")

@app.route('/')
def home():
    return "Welcome to the Predictive Maintenance API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)  # Add this line to log incoming data
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': float(prediction[0])})  # Convert to float
if __name__ == '__main__':
    app.run( port=5001)