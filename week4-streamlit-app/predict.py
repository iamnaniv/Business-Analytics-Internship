"""Prediction logic, kept separate from the UI so it is easy to test."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "model"


def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.load(open(MODEL_DIR / "model_meta.json"))
    return model, meta


def predict(model, meta, values: dict):
    """Return (probability_malignant, is_malignant) for one sample given as {feature: value}."""
    row = pd.DataFrame([values], columns=meta["features"])
    proba = float(model.predict_proba(row)[0, 1])
    return proba, proba >= meta["threshold"]
