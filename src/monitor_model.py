"""Small feature schema shared by CTU-13 model training and the packet monitor."""
from pathlib import Path

import joblib
import pandas as pd

from src.monitor_flows import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "stage4" / "flow_monitor.joblib"
FEATURES = ["duration_s", "packets", *[f"proto_{protocol}" for protocol in PROTOCOLS]]


def features(rows):
    output = rows[["duration_s", "packets"]].astype(float).copy()
    for protocol in PROTOCOLS:
        output[f"proto_{protocol}"] = rows["protocol"].eq(protocol).astype(float)
    return output


def load_monitor():
    bundle = joblib.load(BUNDLE)
    if bundle["features"] != FEATURES:
        raise ValueError("The bundled monitor model and capture feature schema do not match.")
    return bundle


def alert_decision(label, gaussian_flag):
    return gaussian_flag | (label != "normal")


def classify(bundle, rows):
    values = features(rows)
    scores = bundle["gaussian"].score_samples(values)
    model = bundle["classifier"]
    probabilities = model.predict_proba(values)
    indices = probabilities.argmax(axis=1)
    return pd.DataFrame({
        "label": model.classes_[indices],
        "confidence": probabilities.max(axis=1),
        "gaussian_score": scores,
        "gaussian_flag": scores > bundle["gaussian_threshold"],
    })
