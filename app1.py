
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"

app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "An educational ML prediction API. "
        "Not intended for medical diagnosis."
    ),
    version="1.0.0",
)


def load_artifact():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Run python model_training.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    if not all(
        key in artifact for key in ["pipeline", "features"]
    ):
        raise RuntimeError("The saved model artifact is invalid.")

    return artifact


artifact = load_artifact()
pipeline = artifact["pipeline"]
FEATURES = artifact["features"]
THRESHOLD = artifact.get("threshold", 0.5)


class DiabetesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pregnancies: Annotated[int, Field(ge=0, le=30)]
    glucose: Annotated[float, Field(gt=0, le=300)]
    blood_pressure: Annotated[float, Field(gt=0, le=200)]
    skin_thickness: Annotated[float, Field(ge=0, le=100)]
    insulin: Annotated[float, Field(ge=0, le=1000)]
    bmi: Annotated[float, Field(gt=0, le=100)]
    diabetes_pedigree_function: Annotated[
        float, Field(ge=0, le=5)
    ]
    age: Annotated[int, Field(ge=18, le=120)]


@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API",
        "docs": "/docs",
        "disclaimer": "Educational use only; not medical advice.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: DiabetesInput):
    values = {
        "Pregnancies": data.pregnancies,
        "Glucose": data.glucose,
        "BloodPressure": data.blood_pressure,
        "SkinThickness": data.skin_thickness,
        "Insulin": data.insulin,
        "BMI": data.bmi,
        "DiabetesPedigreeFunction":
            data.diabetes_pedigree_function,
        "Age": data.age,
    }

    input_df = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        probability = float(
            pipeline.predict_proba(input_df)[0, 1]
        )

        predicted_class = int(probability >= THRESHOLD)

        return {
            "predicted_class": predicted_class,
            "estimated_positive_class_probability": round(
                probability, 4
            ),
            "threshold": THRESHOLD,
            "model": artifact.get(
                "model_name", "Logistic Regression"
            ),
            "disclaimer": (
                "Educational prediction only; "
                "not a medical diagnosis."
            ),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid prediction input.",
        ) from exc
