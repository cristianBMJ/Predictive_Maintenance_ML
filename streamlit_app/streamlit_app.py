import streamlit as st
import requests

st.title("Predictive Maintenance Model")

option = st.selectbox("Choose input method:", ["Manual Input", "Select Row", "List"])
    
if option == "Manual Input":
    features = st.text_input("Enter 12 float features as comma-separated values:")
    st.write( features ) 
    st.write( features.split(',')[0].strip() ) 

    if st.button("Predict"):
        features = [float(x.strip() ) for x in features.split( ',') ]

        # Call your prediction function here
        #prediction = model.predict([features_list])
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction:")
        st.write(f"Prediction: {prediction}")

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
        # Prepare the data for the API
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction")