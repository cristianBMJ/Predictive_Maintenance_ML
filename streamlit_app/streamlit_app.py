import streamlit as st
import requests
import pandas as pd  # Import pandas to handle data
import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store true and predicted values
if 'true_values' not in st.session_state:
    st.session_state.true_values = []
if 'predicted_values' not in st.session_state:
    st.session_state.predicted_values = []


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

                        # Aggregate true and predicted values
            st.session_state.true_values.append(y_real)
            st.session_state.predicted_values.append(prediction)


            # Plotting true vs predicted value (single point)
            st.subheader("True vs Predicted Value")
            plt.figure(figsize=(4, 4))
            plt.scatter(y_real, prediction, color='blue', label='Prediction')
            plt.plot([y_real, y_real], [y_real, prediction], 'r--', lw=2, label='Error')
            plt.plot([y_real], [y_real], 'go', label='True Value')
            plt.title('True vs Predicted TEY')
            plt.xlabel('True TEY')
            plt.ylabel('Predicted TEY')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            # Plotting aggregated true vs predicted values
            st.subheader("Aggregated True vs Predicted Values")
            true_vals = st.session_state.true_values
            pred_vals = st.session_state.predicted_values

            plt.figure(figsize=(7, 3))
            plt.scatter(true_vals, pred_vals, alpha=0.6, label='Predicted')
            min_val = min(min(true_vals), min(pred_vals))
            max_val = max(max(true_vals), max(pred_vals))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y = x)')
            plt.title('Aggregated True vs Predicted TEY')
            plt.xlabel('True TEY')
            plt.ylabel('Predicted TEY')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)

            st.write("*Aggredate new prediction push botton predict" ) 


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
