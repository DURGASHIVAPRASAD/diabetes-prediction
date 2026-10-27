
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Outcome"

# In the Pima dataset, zero can represent missing measurements
# for these specific medical features.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place diabetes.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing target labels.
    df = df.dropna(subset=[TARGET])

    # The target must contain binary labels: 0 or 1.
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Outcome must contain only 0 and 1.")

    # Handle known missing-value placeholders in this dataset.
    for col in ZERO_AS_MISSING:
        df.loc[df[col] == 0, col] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows where every input feature is missing.
    df = df.dropna(subset=FEATURES, how="all")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Training requires examples from both target classes."
        )

    if df[TARGET].value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two samples."
        )

    return df


# --------------------------------------------------
# 3. Train, evaluate, and save the model
# --------------------------------------------------

def train_model():
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Imputation and scaling are fitted on training data only.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Lower predicted risk class", "Positive class"],
            output_dict=True,
            zero_division=0,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete pipeline so inference uses the same
    # preprocessing that was fitted during training.
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "threshold": 0.5,
        "model_name": "Logistic Regression",
    }

    joblib.dump(artifact, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
