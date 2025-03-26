import streamlit as st
import requests
import pandas as pd  # Import pandas to handle data


def get_random_row(data):
    random_row = data.sample(n=1)  # Get a random row from the dataset
    features = random_row.values.flatten().tolist()  # Flatten the row to a list
    return features

st.title("Predictive Maintenance Model")


# Load your dataset here
df= pd.read_csv("/home/cris/workaplace/Predictive_Maintenance_ML/data/processed_data.csv")  # Update with your dataset path

data = df.drop( columns='TEY')

option = st.selectbox("Choose input method:", ["Random Input", "Manual Input", "Select Row", "List" ])  # Added "Random Input"
    
if option == "Random Input":  # New option for random input

    st.button("Get random row:")

    features = get_random_row(data)
    st.write("Randomly selected features:", features)  # Display the selected features

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction")

elif option == "Select Row":
    # Implement logic to select a row from a dataset
    # For example, using st.selectbox or st.dataframe
    pass  # Replace with actual implementation

elif option == "List":
    # Initialize a list to hold the features
    features = []

    # Input fields for features
    for i in range(1, 13):
        feature_value = st.number_input(f"Feature {i}")  # Get input for each feature
        features.append(feature_value)  # Append the input value to the features list

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction")

elif option == "Manual Input":
    features = st.text_input("Enter 12 float features as comma-separated values:")
    st.write(features) 
    st.write(features.split(',')[0].strip()) 

    if st.button("Predict"):
        features = [float(x.strip()) for x in features.split(',')]
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction:")
        st.write(f"Prediction: {prediction}")