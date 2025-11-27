import streamlit as st
import numpy as np
import joblib

# Load models
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Kidney Stone Prediction - Results Page")

# Check if inputs exist
if "inputs" in st.session_state:
    inputs = st.session_state["inputs"]

    # Display entered values
    st.subheader("Entered Values")
    st.write(inputs)

    # Convert to array
    input_data = np.array([[inputs["Urine Specific Gravity"],
                            inputs["Urine pH"],
                            inputs["Osmolality"],
                            inputs["Conductivity"],
                            inputs["Urea"],
                            inputs["Calcium"]]])
    input_scaled = scaler.transform(input_data)

    # Predictions
    rf_pred = rf_model.predict(input_scaled)[0]
    lr_pred = lr_model.predict(input_scaled)[0]

    rf_result = "High Risk of Kidney Stone" if rf_pred == 1 else "Low Risk"
    lr_result = "High Risk of Kidney Stone" if lr_pred == 1 else "Low Risk"

    st.subheader("Predictions")
    st.success(f"Random Forest: {rf_result}")
    st.success(f"Logistic Regression: {lr_result}")
else:
    st.warning("No input values found. Please go back to the Input Page.")
