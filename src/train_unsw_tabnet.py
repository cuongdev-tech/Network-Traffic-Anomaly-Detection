"""Huan luyen TabNet cho bo du lieu UNSW-NB15 thay the cho MLP.
Dung pipeline chu y tuan tu (sequential attention) cua TabNet.
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
OUT_DIR = BASE_DIR / "results" / "unsw_nb15"
MODEL_DIR = BASE_DIR / "models" / "unsw_nb15"

DROP_COLS = {"label", "attack_cat", "id", "srcip", "dstip", "stime", "ltime"}


def load_unsw(sample=60000):
    train_path = DATA_DIR / "UNSW_NB15_training-set.parquet"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Thieu file parquet UNSW_NB15 trong data/raw/unsw_nb15/")

    frames = []
    for p in (train_path, test_path):
        df = pd.read_parquet(p)
        df.columns = [str(c).strip().lower() for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Chuan hoa nhan
    labels = data["label"].astype(str).str.strip().str.lower()
    norm_labels = labels.map({"0": "BENIGN", "1": "ATTACK", "benign": "BENIGN", "attack": "ATTACK", "normal": "BENIGN"})
    data["label"] = norm_labels

    if sample and sample < len(data):
        data, _ = train_test_split(data, train_size=sample, random_state=42, stratify=data["label"])

    features = data.drop(columns=list(DROP_COLS), errors="ignore")
    return features, data["label"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Doc du lieu UNSW-NB15...")
    X, y_text = load_unsw(sample=60000)
    print(f"So luong dong: {len(X):,} | So dac trung: {X.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)
    classes = list(le.classes_)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

    X_train_df, X_test_df, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    y_test_text = le.inverse_transform(y_test_enc)

    print("Tien xu ly dac trung...")
    X_tr_proc = preprocessor.fit_transform(X_train_df).astype(np.float32)
    X_te_proc = preprocessor.transform(X_test_df).astype(np.float32)

    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_tr_proc, y_train_enc, test_size=0.1, random_state=42, stratify=y_train_enc
    )

    print("Huan luyen TabNet tren UNSW-NB15...")
    clf = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.02),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 5, "gamma": 0.8},
        mask_type="sparsemax",
        verbose=1,
        seed=42,
    )

    t0 = time.time()
    clf.fit(
        X_train=X_tr_sub,
        y_train=y_tr_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=12,
        patience=4,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False,
    )
    train_time = time.time() - t0

    t1 = time.time()
    pred_enc = clf.predict(X_te_proc)
    pred_time = time.time() - t1
    pred_text = le.inverse_transform(pred_enc)

    acc = accuracy_score(y_test_text, pred_text)
    pw = precision_score(y_test_text, pred_text, average="weighted", zero_division=0)
    rw = recall_score(y_test_text, pred_text, average="weighted", zero_division=0)
    f1w = f1_score(y_test_text, pred_text, average="weighted", zero_division=0)
    pm = precision_score(y_test_text, pred_text, average="macro", zero_division=0)
    rm = recall_score(y_test_text, pred_text, average="macro", zero_division=0)
    f1m = f1_score(y_test_text, pred_text, average="macro", zero_division=0)

    print(f"\n[UNSW-NB15 TabNet] Accuracy: {acc * 100:.4f}% | F1: {f1w * 100:.4f}%")

    metrics_df = pd.DataFrame([{
        "model": "TabNet",
        "accuracy": round(acc, 6),
        "precision_weighted": round(pw, 6),
        "recall_weighted": round(rw, 6),
        "f1_weighted": round(f1w, 6),
        "precision_macro": round(pm, 6),
        "recall_macro": round(rm, 6),
        "f1_macro": round(f1m, 6),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(pred_time, 2),
    }])
    metrics_df.to_csv(OUT_DIR / "tabnet_metrics.csv", index=False)

    report = classification_report(y_test_text, pred_text, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    per_class_df.insert(0, "model", "TabNet")
    per_class_df = per_class_df[per_class_df["class"].isin(classes)]
    per_class_df.to_csv(OUT_DIR / "tabnet_per_class.csv", index=False)

    with open(OUT_DIR / "tabnet_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "UNSW-NB15 (BENIGN vs ATTACK)",
            "rows_used": len(X),
            "features": X.shape[1],
            "classes": classes,
            "train": len(X_train_df),
            "test": len(X_test_df),
            "split": "80/20 Stratified (60k sample)",
            "architecture": "TabNet (Attentive Tabular Deep Learning)",
        }, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_test_text, pred_text, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (Actual)")
    plt.title("Confusion Matrix - TabNet (UNSW-NB15)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cm_tabnet.png", dpi=150)
    plt.close()

    clf.save_model(str(MODEL_DIR / "tabnet_attack"))
    joblib.dump({"model": clf, "preprocessor": preprocessor, "classes": classes, "features": list(X.columns)},
                MODEL_DIR / "tabnet_bundle.joblib", compress=3)
    print("Hoan tat huan luyen va luu TabNet cho UNSW-NB15!")


if __name__ == "__main__":
    main()
