from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page control
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"   # options: input, results, evaluation

# -----------------------------
# Input Page
# -----------------------------
if st.session_state.page == "input":
    st.title("Kidney Stone Prediction Model")
    st.write("Developed by CHANDRASEKARAN S & Team")

    gravity = st.slider("Urine Specific Gravity", 1.005, 1.035, value=1.005, step=0.001)
    ph = st.slider("Urine pH", 4.5, 8.0, value=4.5, step=0.1)
    osmo = st.slider("Osmolality", 100, 1300, value=100)
    cond = st.slider("Conductivity", 5.0, 40.0, value=5.0)
    urea = st.slider("Urea (mg/dL)", 10, 650, value=10)
    calc = st.slider("Calcium (mg/dL)", 0.1, 15.0, value=0.1)

    default_values = {"gravity":1.005,"ph":4.5,"osmo":100,"cond":5.0,"urea":10,"calc":0.1}

    if st.button("Predict"):
        inputs = {"gravity":gravity,"ph":ph,"osmo":osmo,"cond":cond,"urea":urea,"calc":calc}
        if inputs == default_values:
            st.warning("Please input your levels")
        else:
            st.session_state.inputs = inputs
            st.session_state.page = "results"
            st.rerun()

    if st.button("Evaluate Models"):
        st.session_state.page = "evaluation"
        st.rerun()

# -----------------------------
# Results Page
# -----------------------------
elif st.session_state.page == "results":
    st.title("Kidney Stone Prediction Model")
    st.write("Developed by CHANDRASEKARAN S & Team")
    inputs = st.session_state.inputs

    entered_values = pd.DataFrame({
        "Parameter":["Urine Specific Gravity","Urine pH","Osmolality","Conductivity","Urea","Calcium"],
        "Value":[inputs["gravity"],inputs["ph"],inputs["osmo"],inputs["cond"],inputs["urea"],inputs["calc"]]
    })
    st.table(entered_values)

    input_data = np.array([[inputs["gravity"],inputs["ph"],inputs["osmo"],inputs["cond"],inputs["urea"],inputs["calc"]]])
    input_scaled = scaler.transform(input_data)

    rf_pred = rf_model.predict(input_scaled)[0]
    lr_pred = lr_model.predict(input_scaled)[0]

    rf_result = "High Risk of Kidney Stone" if rf_pred == 1 else "Low Risk"
    lr_result = "High Risk of Kidney Stone" if lr_pred == 1 else "Low Risk"

    st.subheader("Predictions")
    st.success(f"Random Forest: {rf_result}")
    st.success(f"Logistic Regression: {lr_result}")

    if st.button("Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()

    if st.button("Evaluate Models"):
        st.session_state.page = "evaluation"
        st.rerun()

# -----------------------------
# Evaluation Page
# -----------------------------
elif st.session_state.page == "evaluation":
    st.title("Model Evaluation")
    st.write("Performance metrics on test data")

    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    st.subheader("Accuracy")
    st.write(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
    st.write(f"Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")

    st.subheader("Classification Report")
    st.text("Random Forest:\n" + classification_report(y_test, y_pred_rf))
    st.text("Logistic Regression:\n" + classification_report(y_test, y_pred_lr))

    st.subheader("Confusion Matrix")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Random Forest Confusion Matrix")
    st.pyplot(fig)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fig, ax = plt.subplots()
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title("Logistic Regression Confusion Matrix")
    st.pyplot(fig)

    st.subheader("ROC Curve")
    rf_probs = rf_model.predict_proba(X_test)[:,1]
    lr_probs = lr_model.predict_proba(X_test)[:,1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)

    plt.figure()
    plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_score(y_test, rf_probs):.2f})")
    plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={roc_auc_score(y_test, lr_probs):.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    if st.button("Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()
