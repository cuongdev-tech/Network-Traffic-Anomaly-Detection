"""Train Random Forest tren bo du lieu CIC-IDS2017 da chuan bi.

Chay: python src/train_classifier.py
Input: data/processed/cicids_60k.csv.gz
Output: models/cic/random_forest_attack.joblib
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
IN = BASE_DIR / "data" / "processed" / "cicids_60k.csv.gz"
MODEL = BASE_DIR / "models" / "cic" / "random_forest_attack.joblib"


def main():
    if not IN.is_file():
        raise FileNotFoundError(
            f"Khong tim thay {IN}. Hay chay `python src/prepare_cicids.py` truoc."
        )

    df = pd.read_csv(IN)
    if "Label" not in df.columns:
        raise ValueError(f"Du lieu CIC-IDS2017 can cot 'Label': {IN}")

    X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    y = df["Label"].astype(str).str.strip()
    if X.empty:
        raise ValueError(f"Khong tim thay dac trung so trong {IN}")
    if y.nunique() < 2:
        raise ValueError("Can it nhat 2 lop de huan luyen Random Forest.")

    print(f"Du lieu: {IN}")
    print(f"Tong mau: {len(df):,} | So dac trung: {X.shape[1]}")
    print(f"Phan bo lop: {y.value_counts().to_dict()}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {len(Xtr):,} | Test: {len(Xte):,}")

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print(f"Accuracy : {accuracy_score(yte, pred):.4f}")
    print(
        f"Precision: {precision_score(yte, pred, average='weighted', zero_division=0):.4f}"
    )
    print(f"Recall   : {recall_score(yte, pred, average='weighted', zero_division=0):.4f}")
    print(f"F1       : {f1_score(yte, pred, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(yte, pred, labels=clf.classes_))
    print(classification_report(yte, pred, zero_division=0))
    print("Top 10 feature importance:")
    for feature, importance in sorted(
        zip(X.columns, clf.feature_importances_), key=lambda item: -item[1]
    )[:10]:
        print(f"  {feature}: {importance:.4f}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": list(clf.classes_),
            "features": list(X.columns),
        },
        MODEL,
    )
    print(f"Luu model: {MODEL}")


if __name__ == "__main__":
    main()
