"""Giai doan 3 - Phan baseline cua nhom: Logistic Regression, Decision Tree, KNN.

Du lieu: CIC-IDS2017 (Friday afternoon: DDoS + PortScan), gop thanh 3 lop
BENIGN / DDoS / PortScan. Day la bo du lieu thu 2, ben canh NETRESEC o Giai doan 1.

Chay:
    python src/train_baselines.py
    python src/train_baselines.py --sample 120000
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "cicids2017"
OUT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Cot dinh danh / ro ri thong tin, khong dua vao mo hinh
DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Fwd Header Length.1"]


def load_data():
    frames = []
    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Thieu {path}. Tai tu https://huggingface.co/datasets/c01dsnap/CIC-IDS2017 "
                f"(file {name}) va dat vao data/raw/cicids2017/"
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["source_file"] = name
        frames.append(df)
        print(f"  {name}: {len(df):,} dong")
    data = pd.concat(frames, ignore_index=True)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")

    # Chuan hoa ten nhan
    data["Label"] = data["Label"].astype(str).str.strip()
    data.loc[data["Label"].str.upper() == "BENIGN", "Label"] = "BENIGN"

    print("\nPhan bo nhan goc:")
    print(data["Label"].value_counts().to_string())
    return data


def sample_stratified(df, n, seed=42):
    if n is None or n >= len(df):
        return df
    _, out = train_test_split(df, test_size=n, random_state=seed, stratify=df["Label"])
    return out


def build_models():
    """3 mo hinh phan cong: LR, DT, KNN."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=42)),
        ]),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, class_weight="balanced",
                                               random_state=42),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
    }


def evaluate(name, model, Xtr, Xte, ytr, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0

    t1 = time.time()
    pred = model.predict(Xte)
    pred_time = time.time() - t1

    metrics = {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(pred_time, 2),
    }

    # Chi so tung lop
    per_class = pd.DataFrame(
        classification_report(yte, pred, output_dict=True, zero_division=0)
    ).T.reset_index().rename(columns={"index": "class"})
    per_class.insert(0, "model", name)
    per_class = per_class[per_class["class"].isin([str(c) for c in np.unique(yte)])]

    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f} (weighted) | "
          f"{metrics['precision_macro']:.4f} (macro)")
    print(f"Recall   : {metrics['recall_weighted']:.4f} (weighted) | "
          f"{metrics['recall_macro']:.4f} (macro)")
    print(f"F1       : {metrics['f1_weighted']:.4f} (weighted) | "
          f"{metrics['f1_macro']:.4f} (macro)")
    print(f"Train {train_time:.1f}s | Predict {pred_time:.1f}s")
    print(classification_report(yte, pred, zero_division=0))

    # Confusion matrix
    labels = sorted(np.unique(yte))
    cm = confusion_matrix(yte, pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cm_{name}.png", dpi=150)
    plt.close()

    return metrics, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60000,
                    help="So dong lay mau (0 = dung toan bo). Mac dinh 60000 de KNN chay nhanh.")
    ap.add_argument("--cv", action="store_true",
                    help="Chay them 5-fold cross-validation de kiem tra do on dinh.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Doc du lieu CIC-IDS2017...")
    data = load_data()

    if args.sample and args.sample > 0:
        data = sample_stratified(data, args.sample)
        print(f"\nLay mau {len(data):,} dong (stratified).")

    y_text = data["Label"]
    X = data.drop(columns=["Label", "source_file"])
    X = X.select_dtypes(include=[np.number])
    print(f"So dac trung: {X.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    print("Nhan:", dict(zip(le.classes_, range(len(le.classes_)))))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train: {len(Xtr):,} | Test: {len(Xte):,}")

    results = []
    per_class_frames = []
    cv_rows = []
    for name, model in build_models().items():
        metrics, per_class = evaluate(name, model, Xtr, Xte, ytr, yte)
        results.append(metrics)
        per_class_frames.append(per_class)
        joblib.dump({"model": model, "labels": list(le.classes_),
                     "features": list(X.columns)},
                    MODEL_DIR / f"baseline_{name}.joblib")

        if args.cv:
            print(f"  5-fold CV cho {name}...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_validate(
                clone(model), X, y, cv=cv, n_jobs=1,
                scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            )
            row = {"model": name}
            for key in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
                s = scores[f"test_{key}"]
                row[f"cv_{key}_mean"] = round(float(s.mean()), 4)
                row[f"cv_{key}_std"] = round(float(s.std()), 4)
            cv_rows.append(row)
            print(f"  CV F1-macro: {row['cv_f1_macro_mean']:.4f} "
                  f"(+/- {row['cv_f1_macro_std']:.4f})")

    res = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    res.to_csv(OUT_DIR / "baselines_metrics.csv", index=False)
    print("\n=== Bang so sanh ===")
    print(res[["model", "accuracy", "precision_weighted", "recall_weighted",
               "f1_weighted", "f1_macro", "train_time_s"]].to_string(index=False))

    pd.concat(per_class_frames, ignore_index=True).to_csv(
        OUT_DIR / "baselines_per_class.csv", index=False)

    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv(OUT_DIR / "baselines_cv.csv", index=False)
        print("\n=== 5-fold cross-validation ===")
        print(cv_df.to_string(index=False))

    # Bieu do so sanh
    ax = res.set_index("model")[["accuracy", "precision_weighted",
                                 "recall_weighted", "f1_weighted"]].plot(
        kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("So sanh 3 mo hinh baseline (LR / DT / KNN) - CIC-IDS2017")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baselines_compare.png", dpi=150)
    plt.close()

    with open(OUT_DIR / "baselines_meta.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "CIC-IDS2017 Friday afternoon (DDoS + PortScan)",
                   "rows_used": int(len(data)),
                   "features": int(X.shape[1]),
                   "classes": list(le.classes_),
                   "train": int(len(Xtr)), "test": int(len(Xte))},
                  f, ensure_ascii=False, indent=2)

    print(f"\nDa luu: {OUT_DIR / 'baselines_metrics.csv'}")
    print(f"Da luu hinh: {OUT_DIR / 'baselines_compare.png'}, cm_*.png")


if __name__ == "__main__":
    main()
