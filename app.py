import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st

#import model
model = load_model('model.h5')

#import ohe
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

#import labelencoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

# import scaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')

# user input
st.write('INPUT VARIABLES')
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('BAlance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is active member', [0,1])

#convert input to dict
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

#scaling
input_data_scaled = scaler.transform(input_data)

# prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(prediction_proba)
# show straemlit app
if prediction_proba >0.5:
    st.write('the customer is likely to churn.')
else:
    st.write('the customer is not likely to churn.')