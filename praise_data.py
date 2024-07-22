import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('Expresso_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Streamlit app title
st.title('Churn Prediction App')

# Input fields for the features
def user_input_features():
    feature_dict = {}
    for column in model.feature_names_in_:
        if column in label_encoders:
            options = label_encoders[column].classes_
            feature_dict[column] = st.selectbox(f'{column}', options)
        else:
            feature_dict[column] = st.number_input(f'{column}')
    return pd.DataFrame(feature_dict, index=[0])

# Get user input
input_df = user_input_features()

# Encode the categorical features
for column in label_encoders:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f'The predicted class is: {prediction[0]}')
