# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load and train models
# -----------------------------
df = pd.read_csv("kidney-stone-dataset.csv")
df.drop(columns=df.columns[0], inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Kidney Stone Prediction App")
st.write("Developed by CHANDRASEKARAN S & Team")

# Input fields
gravity = st.slider("Urine Specific Gravity", 1.005, 1.035, step=0.001)
ph = st.slider("Urine pH", 4.5, 8.0, step=0.1)
osmo = st.slider("Osmolality", 100, 1300)
cond = st.slider("Conductivity", 5.0, 40.0)
urea = st.slider("Urea (mg/dL)", 10, 650)
calc = st.slider("Calcium (mg/dL)", 0.1, 15.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[gravity, ph, osmo, cond, urea, calc]])
    input_scaled = scaler.transform(input_data)

    # Predictions
    rf_pred = rf_model.predict(input_scaled)[0]
    lr_pred = lr_model.predict(input_scaled)[0]

    rf_result = "High Risk of Kidney Stone" if rf_pred == 1 else "Low Risk"
    lr_result = "High Risk of Kidney Stone" if lr_pred == 1 else "Low Risk"

    # -----------------------------
    # Results Page
    # -----------------------------
    st.subheader("Entered Values")
    st.write({
        "Urine Specific Gravity": gravity,
        "Urine pH": ph,
        "Osmolality": osmo,
        "Conductivity": cond,
        "Urea": urea,
        "Calcium": calc
    })

    st.subheader("Predictions")
    st.success(f"Random Forest: {rf_result}")
    st.success(f"Logistic Regression: {lr_result}")
