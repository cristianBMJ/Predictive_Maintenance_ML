# streamlit_app.py
import streamlit as st
import requests

st.title("Predictive Maintenance Model")

# Input fields for features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# Add more input fields as needed

if st.button("Predict"):
    # Prepare the data for the API
    features = [feature1, feature2]  # Add all features here
    response = requests.post("http://127.0.0.1:5000/predict", json={'features': features})
    
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Error in prediction")