import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('regression_model.h5')

# Load the saved scalers and encoders
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)


# Define the Streamlit app
st.title('Salary Prediction')

# Create input fields for the features
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
gender = st.selectbox('Gender', ['Female', 'Male'])
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
exited = st.selectbox('Exited', [0, 1])  # Add input for 'Exited'

# Create a button to trigger prediction
if st.button('Predict Salary'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Gender': [gender],
        'Geography': [geography],
        'Exited': [exited]  # Include 'Exited' in the input data
    })

    # Encode Gender
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

    # One-Hot Encode Geography
    possible_geographies = ['France', 'Germany', 'Spain']
    geo_encoded_df = pd.DataFrame(0, index=input_data.index, columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])
    selected_geo = input_data['Geography'].iloc[0]
    if selected_geo in possible_geographies:
        if selected_geo == 'France':
            geo_encoded_df['Geography_France'] = 1
        elif selected_geo == 'Germany':
            geo_encoded_df['Geography_Germany'] = 1
        elif selected_geo == 'Spain':
            geo_encoded_df['Geography_Spain'] = 1

    input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Ensure correct column order (important for the model)
    feature_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited', 'Geography_France', 'Geography_Germany', 'Geography_Spain'] # Add 'Exited' to feature order
    input_data = input_data[feature_order]

    # Scale the numerical features
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)[0][0]

    # Display the prediction
    st.success(f'Predicted Estimated Salary: ${prediction:.2f}')