# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load and train models once
# -----------------------------
@st.cache_resource
def train_models():
    # Load dataset
    df = pd.read_csv("kidney-stone-dataset.csv")

    # Detect target column automatically (last column assumed if not named)
    if "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]  # assume last column is target

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    return scaler, rf_model, lr_model

scaler, rf_model, lr_model = train_models()

# -----------------------------
# Page control
# -----------------------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# -----------------------------
# Input Page
# -----------------------------
if not st.session_state.show_results:
    st.title("Kidney Stone Prediction - Input Page")
    st.write("Developed by CHANDRASEKARAN S & Team")

    gravity = st.slider("Urine Specific Gravity", 1.005, 1.035, step=0.001)
    ph = st.slider("Urine pH", 4.5, 8.0, step=0.1)
    osmo = st.slider("Osmolality", 100, 1300)
    cond = st.slider("Conductivity", 5.0, 40.0)
    urea = st.slider("Urea (mg/dL)", 10, 650)
    calc = st.slider("Calcium (mg/dL)", 0.1, 15.0)

    if st.button("Predict"):
        st.session_state.inputs = {
            "Urine Specific Gravity": gravity,
            "Urine pH": ph,
            "Osmolality": osmo,
            "Conductivity": cond,
            "Urea": urea,
            "Calcium": calc
        }
        st.session_state.show_results = True
        st.rerun()   # ✅ updated

# -----------------------------
# Results Page
# -----------------------------
else:
    st.title("Kidney Stone Prediction - Results Page")

    inputs = st.session_state.inputs
    st.subheader("Entered Values")
    st.write(inputs)

    input_data = np.array([[inputs["Urine Specific Gravity"],
                            inputs["Urine pH"],
                            inputs["Osmolality"],
                            inputs["Conductivity"],
                            inputs["Urea"],
                            inputs["Calcium"]]])
    input_scaled = scaler.transform(input_data)

    rf_pred = rf_model.predict(input_scaled)[0]
    lr_pred = lr_model.predict(input_scaled)[0]

    rf_result = "High Risk of Kidney Stone" if rf_pred == 1 else "Low Risk"
    lr_result = "High Risk of Kidney Stone" if lr_pred == 1 else "Low Risk"

    st.subheader("Predictions")
    st.success(f"Random Forest: {rf_result}")
    st.success(f"Logistic Regression: {lr_result}")

    if st.button("Back to Input Page"):
        st.session_state.show_results = False
        st.rerun()   # ✅ updated
