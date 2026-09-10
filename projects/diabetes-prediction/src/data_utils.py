"""Reusable data loading and validation helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

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
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str | Path = "data/diabetes.csv") -> pd.DataFrame:
    """Read the CSV and validate its schema."""
    frame = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return frame[expected].copy()


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Replace benchmark zero placeholders with NaN; keep target untouched."""
    data = frame.copy()
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)
    return data[FEATURES], data[TARGET].astype(int)
