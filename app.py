
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Diabetes Prediction")
st.write(
    "Enter the requested measurements to see the model's prediction."
)

st.warning(
    "Educational demonstration only. This model is not a "
    "medical diagnostic tool. Consult a qualified healthcare "
    "professional for medical advice."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python model_training.py' first."
        )

    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    threshold = artifact.get("threshold", 0.5)

except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


st.subheader("Patient measurements")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )

        glucose = st.number_input(
            "Glucose",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=1.0,
            max_value=200.0,
            value=70.0,
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
        )

        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=100.0,
            value=28.0,
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            format="%.3f",
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=35,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True,
    )


if submitted:
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age,
        ]],
        columns=features,
    )

    try:
        probability = float(
            pipeline.predict_proba(input_data)[0, 1]
        )

        predicted_class = int(probability >= threshold)

        st.subheader("Prediction result")

        st.metric(
            "Estimated positive-class probability",
            f"{probability:.1%}",
        )

        st.progress(probability)

        if predicted_class == 1:
            st.warning(
                "The model predicts the positive class. "
                "This is not a diagnosis."
            )
        else:
            st.success(
                "The model predicts the negative class. "
                "This does not rule out diabetes."
            )

        st.caption(
            "The probability is model-estimated and may not be "
            "calibrated to an individual's actual medical risk."
        )

    except (ValueError, TypeError) as exc:
        st.error(f"Could not generate prediction: {exc}")
