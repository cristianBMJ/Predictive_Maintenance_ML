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
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)