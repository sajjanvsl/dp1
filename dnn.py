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

# ===== TOP NAVIGATION =====
tab1, tab2 = st.tabs(["🩺 Prediction System", "🔐 Privacy Policy"])

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

    c.drawString(50, 200, "This is an AI-assisted prediction.")
    c.drawString(50, 180, "Consult a certified doctor for confirmation.")

    c.drawString(50, 150, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    c.save()
    return file


# ================= TRAIN MODEL IF NEEDED =================
if not os.path.exists(f"{MODEL_DIR}/dnn1.h5"):

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


# ==========================================================
# ===================== TAB 1 ==============================
# ==========================================================
with tab1:

    # ================= HEADER =================
    st.markdown("""
    <div style="text-align:center; padding:10px; border-radius:15px; border:2px solid #4CAF50">
    <h3>🩺 Early Stage Diabetes Prediction</h3>
    <p>Deep Neural Network Based Clinical System</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🧍 Patient Information")

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

        colb1, colb2 = st.columns(2)
        submit = colb1.form_submit_button("🔍 Predict")
        reset = colb2.form_submit_button("🔄 Reset")

    if reset:
        st.rerun()

    # ================= PREDICTION =================
    if submit:

        gender_val = 1 if Gender == "Male" else 0

        X_single = pd.DataFrame([[
            Age, gender_val, Polyuria, Polydipsia, SuddenWeightLoss,
            Polyphagia, VisualBlurring, Obesity, DelayedHealing, Irritability
        ]], columns=FEATURES)

        Xs = scaler.transform(X_single)

        p1 = dnn1.predict(Xs)[0][0]
        p2 = dnn2.predict(Xs)[0][0]
        prob = (p1 + p2) / 2

        pred = 1 if prob >= THRESHOLD else 0
        result = "DIABETIC" if pred == 1 else "NON-DIABETIC"

        st.subheader("🧾 Prediction Result")
        st.metric("Risk Probability", f"{prob*100:.2f}%")

        if pred:
            st.error("🩺 High Risk of Getting a Diabetic")
        else:
            st.success("✅ Low Risk of Getting a Diabetic")

        patient_info = {
            "Age": Age,
            "Gender": Gender,
            "Polyuria": Polyuria,
            "Polydipsia": Polydipsia,
            "SuddenWeightLoss": SuddenWeightLoss,
            "Polyphagia": Polyphagia,
            "VisualBlurring": VisualBlurring,
            "Obesity": Obesity,
            "DelayedHealing": DelayedHealing,
            "Irritability": Irritability
        }

        pdf = generate_pdf(patient_info, prob, result)

        with open(pdf, "rb") as f:
            st.download_button("⬇ Download Patient PDF Report", f, file_name=pdf)

    # ================= MODEL PERFORMANCE =================
    st.subheader("📊 Model Performance")

    df = pd.read_csv(DATA_PATH)
    X = scaler.transform(df[FEATURES])
    y = df["class"]

    y_prob = (dnn1.predict(X).ravel() + dnn2.predict(X).ravel()) / 2
    y_pred = (y_prob >= THRESHOLD).astype(int)

    acc = accuracy_score(y, y_pred)
    save_accuracy(acc)

    st.metric("DNN Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(y, y_pred)
    fig_cm, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig_cm)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    st.pyplot(fig_roc)

    # ================= FOOTER =================
    st.markdown("""
    <hr>
    <div style='text-align:center'>
    Early Stage Diabetic Prediction using DNN <br>
    @By <b>Pooja Kallappagol</b>, Research Scholar<br>
    Supervisor: <b>Dr. Shitalrani Kavale</b><br>
    Dept. of Computer Science<br>
    Karnataka State Akkamahadevi Women's University, Vijayapur
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# ===================== TAB 2 ==============================
# ==========================================================
with tab2:

    st.title("🔐 Privacy Policy")

    st.markdown("""
**Effective Date: 2026**

This application is developed for academic and research purposes.

---

### Information Collection
This app collects:
- Age
- Gender
- Clinical symptom inputs

It does NOT collect:
- Name
- Phone number
- Email
- Location data

---

### Data Usage
User inputs are used only for:
- AI-based diabetes risk prediction

No personal data is sold, shared, or used for advertising.

---

### Data Storage
- No personal data is permanently stored.
- Accuracy logs contain only model performance metrics.

---

### Medical Disclaimer
This system provides AI-assisted prediction only.
It does NOT replace professional medical advice.
Always consult a qualified healthcare provider.

---

### Contact
Dept. of Computer Science  
Karnataka State Akkamahadevi Women's University, Vijayapur
""")
