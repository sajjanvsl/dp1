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

# ================= SIDEBAR NAVIGATION =================
page = st.sidebar.radio("Navigation", ["Prediction App", "Privacy Policy"])

# ================= PRIVACY PAGE =================
if page == "Privacy Policy":
    st.title("Privacy Policy")
    st.write("Early Stage Diabetes Prediction App")
    st.write("Effective Date: 25 Feb 2026")

    st.markdown("""
### 1. Information We Collect
- Age
- Gender
- Symptom inputs
- Prediction results

We do NOT collect personal identifiers like name, phone, email, or Aadhaar.

### 2. Usage of Information
Data is used only for:
- AI-based diabetes prediction
- Academic research
- Model improvement

### 3. Data Storage
- No permanent personal data storage
- Anonymous accuracy logs only
- No third-party data sharing

### 4. Medical Disclaimer
This app provides AI-assisted predictions and does NOT replace professional medical advice. Always consult a certified doctor.

### 5. Security
Reasonable safeguards are implemented to protect data.

### 6. Children's Privacy
This app is not intended for children under 13 years.

### 7. Contact
Email: sajjanvsl@gmail.com  
Dept. of Computer Science  
Karnataka State Akkamahadevi Women's University, Vijayapur
    """)
    st.stop()

# ================= MAIN APP =================
DATA_PATH = "data/diabetes.csv"
MODEL_DIR = "model"
ACC_LOG = "accuracy_log.csv"
THRESHOLD = 0.35

FEATURES = [
    "Age","Gender","Polyuria","Polydipsia","SuddenWeightLoss",
    "Polyphagia","VisualBlurring","Obesity","DelayedHealing","Irritability"
]

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

    c.drawString(50, 200, "⚠ This is an AI-assisted prediction.")
    c.drawString(50, 180, "Consult a certified doctor for confirmation.")
    c.save()
    return file

# ================= TRAIN MODEL =================
if not os.path.exists(f"{MODEL_DIR}/dnn1.h5"):
    st.info("Training models...")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["class"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    os.makedirs(MODEL_DIR, exist_ok=True)

    for i, seed in enumerate([42, 99], start=1):
        model = build_dnn(Xs.shape[1], seed)
        es = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(Xs, y, epochs=50, batch_size=16,
                  validation_split=0.2, callbacks=[es], verbose=0)
        model.save(f"{MODEL_DIR}/dnn{i}.h5")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# ================= LOAD =================
dnn1 = load_model(f"{MODEL_DIR}/dnn1.h5")
dnn2 = load_model(f"{MODEL_DIR}/dnn2.h5")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

# ================= UI =================
st.markdown("""
<div style="text-align:center; border:2px solid #4CAF50; border-radius:12px; padding:10px">
<h3>🩺 Early Stage Diabetes Prediction</h3>
</div>
""", unsafe_allow_html=True)

st.subheader("Patient Information")

with st.form("form"):
    Age = st.number_input("Age", 1, 120, 55)
    Gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    Polyuria = st.selectbox("Polyuria", [0, 1])
    Polydipsia = st.selectbox("Polydipsia", [0, 1])
    SuddenWeightLoss = st.selectbox("SuddenWeightLoss", [0, 1])
    Polyphagia = st.selectbox("Polyphagia", [0, 1])
    VisualBlurring = st.selectbox("VisualBlurring", [0, 1])
    Obesity = st.selectbox("Obesity", [0, 1])
    DelayedHealing = st.selectbox("DelayedHealing", [0, 1])
    Irritability = st.selectbox("Irritability", [0, 1])
    submit = st.form_submit_button("Predict")

if submit:
    gender_val = 1 if Gender == "Male" else 0

    X_single = pd.DataFrame([[Age, gender_val, Polyuria, Polydipsia,
                              SuddenWeightLoss, Polyphagia,
                              VisualBlurring, Obesity,
                              DelayedHealing, Irritability]],
                            columns=FEATURES)

    Xs = scaler.transform(X_single)
    prob = (dnn1.predict(Xs)[0][0] + dnn2.predict(Xs)[0][0]) / 2

    st.metric("Risk Probability", f"{prob*100:.2f}%")
    st.success("Prediction Complete")

st.markdown("---")
st.caption("© Early Stage Diabetes Prediction Research Project")
