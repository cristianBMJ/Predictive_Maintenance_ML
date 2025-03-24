import streamlit as st
import requests

st.title("Predictive Maintenance Model")

# Initialize a list to hold the features
features = []

# Input fields for features
for i in range(1, 13):
    feature_value = st.number_input(f"Feature {i}")  # Get input for each feature
    features.append(feature_value)  # Append the input value to the features list

if st.button("Predict"):
    # Prepare the data for the API
    response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
    
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Error in prediction")