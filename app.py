import streamlit as st
import numpy as np

st.title("Kidney Stone Prediction - Input Page")

# Input fields
gravity = st.slider("Urine Specific Gravity", 1.005, 1.035, step=0.001)
ph = st.slider("Urine pH", 4.5, 8.0, step=0.1)
osmo = st.slider("Osmolality", 100, 1300)
cond = st.slider("Conductivity", 5.0, 40.0)
urea = st.slider("Urea (mg/dL)", 10, 650)
calc = st.slider("Calcium (mg/dL)", 0.1, 15.0)

# Save inputs in session_state
if st.button("Go to Results Page"):
    st.session_state["inputs"] = {
        "Urine Specific Gravity": gravity,
        "Urine pH": ph,
        "Osmolality": osmo,
        "Conductivity": cond,
        "Urea": urea,
        "Calcium": calc
    }
    st.switch_page("pages/results.py")   # navigates to results page
