import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= CONFIG =================
st.set_page_config(page_title="Early Stage Diabetes Prediction", layout="centered")

DATA_PATH = "data/diabetes.csv"
MODEL_DIR = "model"
ACC_LOG = "accuracy_log.csv"
THRESHOLD = 0.35

FEATURES = [
    "Age","Gender","Polyuria","Polydipsia","SuddenWeightLoss",
    "Polyphagia","VisualBlurring","Obesity","DelayedHealing","Irritability"
]

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["🩺 Prediction", "📊 Model Performance", "🔒 Privacy Policy"])

# ================= UTILITIES =================
def build_dnn(input_dim, seed):
    np.random.seed(seed)
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def save_accuracy(acc):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": acc
    }])
    if os.path.exists(ACC_LOG):
        row.to_csv(ACC_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(ACC_LOG, index=False)

def generate_pdf(patient, prob, result):
    file = "Patient_Diabetes_Report.pdf"
    c = canvas.Canvas(file, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(300, 800, "Early Stage Diabetes Medical Report")

    c.setFont("Helvetica", 11)
    y = 760
    for k, v in patient.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    c.drawString(50, y-10, f"Prediction Result: {result}")
    c.drawString(50, y-30, f"Risk Probability: {prob*100:.2f}%")

    c.drawString(50, 200, "⚠ AI-assisted prediction.")
    c.drawString(50, 180, "Consult a certified doctor.")

    c.save()
    return file

# ================= TRAIN MODEL IF NEEDED =================
if not os.path.exists(f"{MODEL_DIR}/dnn1.h5"):
    st.info("Training Ensemble DNN models...")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["class"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    os.makedirs(MODEL_DIR, exist_ok=True)

    for i, seed in enumerate([42, 99], start=1):
        model = build_dnn(Xs.shape[1], seed)
        es = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(Xs, y, epochs=100, batch_size=16,
                  validation_split=0.2, callbacks=[es], verbose=0)
        model.save(f"{MODEL_DIR}/dnn{i}.h5")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# ================= LOAD MODELS =================
dnn1 = load_model(f"{MODEL_DIR}/dnn1.h5")
dnn2 = load_model(f"{MODEL_DIR}/dnn2.h5")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

# ============================================================
# ===================== TAB 1: PREDICTION ====================
# ============================================================
with tab1:
    st.markdown("### 🩺 Early Stage Diabetes Prediction")

    with st.form("patient_form"):
        Age = st.number_input("Age", 1, 120, 55)
        Gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            Polyuria = st.selectbox("Polyuria", [0, 1])
            Polydipsia = st.selectbox("Polydipsia", [0, 1])
            SuddenWeightLoss = st.selectbox("SuddenWeightLoss", [0, 1])
            Polyphagia = st.selectbox("Polyphagia", [0, 1])
        with col2:
            VisualBlurring = st.selectbox("VisualBlurring", [0, 1])
            Obesity = st.selectbox("Obesity", [0, 1])
            DelayedHealing = st.selectbox("DelayedHealing", [0, 1])
            Irritability = st.selectbox("Irritability", [0, 1])

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        gender_val = 1 if Gender == "Male" else 0
        X_single = pd.DataFrame([[Age, gender_val, Polyuria, Polydipsia,
                                  SuddenWeightLoss, Polyphagia, VisualBlurring,
                                  Obesity, DelayedHealing, Irritability]],
                                columns=FEATURES)

        Xs = scaler.transform(X_single)
        prob = (dnn1.predict(Xs)[0][0] + dnn2.predict(Xs)[0][0]) / 2
        pred = 1 if prob >= THRESHOLD else 0

        st.metric("Risk Probability", f"{prob*100:.2f}%")
        st.success("Low Risk" if pred == 0 else "High Risk")

# ============================================================
# ================= TAB 2: MODEL PERFORMANCE =================
# ============================================================
with tab2:
    st.subheader("📊 Model Performance")

    df = pd.read_csv(DATA_PATH)
    X = scaler.transform(df[FEATURES])
    y = df["class"]

    y_prob = (dnn1.predict(X).ravel() + dnn2.predict(X).ravel()) / 2
    y_pred = (y_prob >= THRESHOLD).astype(int)

    acc = accuracy_score(y, y_pred)
    save_accuracy(acc)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    st.pyplot(fig)

# ============================================================
# ================= TAB 3: PRIVACY POLICY ====================
# ============================================================
with tab3:
    st.title("🔒 Privacy Policy")
    st.markdown("""
**Early Stage Diabetes Prediction App**  
Effective Date: 25 February 2026

### Information We Collect
- Age
- Gender
- Clinical symptoms
- Prediction results

No personal identifiers like name or phone are collected.

### Usage
Used only for:
- AI-based prediction
- Research purposes

### Data Storage
- No permanent storage
- No third-party sharing

### Medical Disclaimer
This app provides AI-assisted predictions and does NOT replace professional medical advice.

### Contact
📧 sajjanvsl@gmail.com
""")
