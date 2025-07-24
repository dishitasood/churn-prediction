# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of X --> 'Age', 'Gender', 'Tensure', 'MonthlyCharges'

import streamlit as st
import numpy as np
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("Customer Churn Prediction")

st.divider()

st.write("Please enter the values and hit the predict button to get a prediction.")

st.divider()

age = st.number_input("Age", min_value=10, max_value=100, value=30)

gender = st.selectbox("Select the gender", ['Male', 'Female'])

tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)

monthly_charge = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

st.divider()

predict_button = st.button("Predict")

if predict_button:
    gender_selected = 1 if gender == 'Female' else 0

    X = [age, gender_selected, tenure, monthly_charge]
    X1 = np.array(X)

    X_array = scaler.transform([X1])
    prediction = model.predict(X_array.reshape(1, -1))

    predicted = "Churn" if prediction == 1 else "Not Churn"
    st.write(f"The customer is predicted to: {predicted}")

else:
    st.write("Please enter the values and hit the predict button to get a prediction.")
