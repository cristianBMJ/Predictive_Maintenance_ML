import streamlit as st
import requests
import pandas as pd  # Import pandas to handle data
import numpy as np

def get_random_row(df):
    
    random_row = df.sample(n=1)  # Get a random row from the dataset
    target = random_row['TEY']
    random_row = random_row.drop( columns='TEY')
    features = random_row.values.flatten().tolist()  # Flatten the row to a list
    target_value = target.iloc[0]    # Flatten the row to a list

    return features, target_value

feature_names = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO', 'NOX', 'Year', 'TAT_TIT_Ratio']


st.title("Predict Turbine Energy Yield (TEY) v1.0")


# Load your dataset here
df= pd.read_csv("/home/cris/workaplace/Predictive_Maintenance_ML/data/processed_data_evaluation.csv")  # Update with your dataset path


option = st.selectbox("Choose input method:", ["Input From Dataset", "Manual Input", "Select Row", "List" ])  # Added "Random Input"
    
if option == "Input From Dataset":  # New option for random input

    st.button("Get random sample:")
    
    features, y_real = get_random_row(df)

    st.write(y_real )

    input_data = {name: int(value)  if name == 'Year' 
                  else value 
                  for name, value in zip(feature_names, features)}


    df = pd.DataFrame(list(input_data.items()), columns=["Features", "Value"])
    df.set_index( "Features", inplace=True)
    st.table(df)

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:5001/predict", json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction TEY: {np.around(prediction, 2)}")

            st.success(f"\n Error: { np.around( np.abs((prediction - y_real )) * 100 / y_real , 3)} %")

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
            st.success(f"Prediction: {np.around( prediction, 2)}")
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
            st.success(f"Prediction: {np.around( prediction, 2) }")
        else:
            st.error("Error in prediction:")
