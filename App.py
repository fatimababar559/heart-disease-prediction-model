"""
Heart Disease Prediction - Streamlit web app

Loads the trained Support Vector Classifier saved by the notebook
(heart_disease_svc_model.pkl) and lets anyone try it in the browser.

IMPORTANT: the option lists below MUST be identical (same words) to the
category lists used in Step 5.1 of the notebook. LabelEncoder gives codes
by alphabetical order, so the same words always give the same numbers.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Put your notebook's test accuracy here (for example 0.78). Leave None to hide the line.
TEST_ACCURACY = None

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"

# ---- Category lists (copy of Step 5.1 in the notebook) ----------------------
SEX_OPTIONS = ["Male", "Female"]
CHEST_PAIN_OPTIONS = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
EXERCISE_ANGINA_OPTIONS = ["Yes", "No"]
HEART_RATE_OPTIONS = ["Low", "Medium", "High"]
HEART_RATE_LABELS = {
    "Low": "Low (below 140 bpm)",
    "Medium": "Medium (140-159 bpm)",
    "High": "High (160 bpm or more)",
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """Re-create the same LabelEncoders as the notebook (same lists -> same codes)."""
    return {
        "Sex": LabelEncoder().fit(SEX_OPTIONS),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN_OPTIONS),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA_OPTIONS),
        "MaxHeartRate": LabelEncoder().fit(HEART_RATE_OPTIONS),
    }


model = load_model()
encoders = build_label_encoders()

# ---- Page ---------------------------------------------------------------------
st.title("❤️ Heart Disease Predictor")
st.markdown(
    "Answer the 4 questions below and the machine learning model will estimate "
    "whether heart disease is likely. Built on the UCI Heart Disease (Cleveland) dataset."
)
if TEST_ACCURACY is not None:
    st.markdown(f"*Model accuracy on held-out test data: **{TEST_ACCURACY:.0%}***")

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX_OPTIONS)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN_OPTIONS, index=3)
with col2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA_OPTIONS, index=1)
    max_heart_rate = st.selectbox(
        "Max Heart Rate Achieved",
        HEART_RATE_OPTIONS,
        index=1,
        format_func=lambda option: HEART_RATE_LABELS[option],
    )

if st.button("Predict", type="primary", use_container_width=True):
    # Column names and order must match the training data
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MaxHeartRate": [max_heart_rate],
    })

    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "MaxHeartRate": encoders["MaxHeartRate"].transform(user_input["MaxHeartRate"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ HEART DISEASE NOT LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.caption(
    "Educational project for a Machine Learning course. Uses only 4 inputs and a small dataset. "
    "This is NOT medical advice - see a doctor for any health concern."
)