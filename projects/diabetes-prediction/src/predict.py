"""Run a single prediction using the saved Decision Tree pipeline."""

from pathlib import Path

import joblib
import pandas as pd

from data_utils import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "diabetes_decision_tree.joblib"


def predict(values: dict[str, float]) -> tuple[int, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first with: python src/train.py")

    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return prediction, probability


if __name__ == "__main__":
    sample = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 30,
    }
    label, probability = predict(sample)
    print(f"Predicted class: {label}")
    print(f"Estimated probability of class 1: {probability:.3f}")
    print("Note: this is a portfolio demonstration, not medical advice.")
