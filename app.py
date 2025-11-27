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
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Enter Input", "Show Results"])

# -----------------------------
# Page 1: Input collection
# -----------------------------
if page == "Enter Input":
    st.title("Kidney Stone Prediction - Input Page")
    st.write("Developed by CHANDRASEKARAN S & Team")

    # Model choice
    st.session_state.model_choice = st.selectbox(
        "Choose Model", ["Random Forest", "Logistic Regression"]
    )

    # Input fields
    st.session_state.gravity = st.slider("Urine Specific Gravity", 1.005, 1.035, step=0.001)
    st.session_state.ph = st.slider("Urine pH", 4.5, 8.0, step=0.1)
    st.session_state.osmo = st.slider("Osmolality", 100, 1300)
    st.session_state.cond = st.slider("Conductivity", 5.0, 40.0)
    st.session_state.urea = st.slider("Urea (mg/dL)", 10, 650)
    st.session_state.calc = st.slider("Calcium (mg/dL)", 0.1, 15.0)

    st.info("Go to the 'Show Results' page to see your prediction.")

# -----------------------------
# Page 2: Display input + output
# -----------------------------
elif page == "Show Results":
    st.title("Kidney Stone Prediction - Results Page")

    # Display entered values
    st.subheader("Entered Input Values")
    st.write(f"Urine Specific Gravity: {st.session_state.gravity}")
    st.write(f"Urine pH: {st.session_state.ph}")
    st.write(f"Osmolality: {st.session_state.osmo}")
    st.write(f"Conductivity: {st.session_state.cond}")
    st.write(f"Urea: {st.session_state.urea}")
    st.write(f"Calcium: {st.session_state.calc}")
    st.write(f"Chosen Model: {st.session_state.model_choice}")

    # Prepare input for prediction
    input_data = np.array([[st.session_state.gravity,
                            st.session_state.ph,
                            st.session_state.osmo,
                            st.session_state.cond,
                            st.session_state.urea,
                            st.session_state.calc]])
    input_scaled = scaler.transform(input_data)

    # Predict
    if st.session_state.model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)
    else:
        prediction = lr_model.predict(input_scaled)

    result = "High Risk of Kidney Stone" if prediction[0] == 1 else "Low Risk"
    st.success(f"Prediction Result: {result}")
