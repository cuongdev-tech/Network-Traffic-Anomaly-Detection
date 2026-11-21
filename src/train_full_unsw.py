"""Huan luyen 5 mo hinh Machine Learning Baseline va mo hinh TabNet (Deep Learning)
cho bo du lieu UNSW-NB15 theo 10 lop (Normal + 9 loai tan cong).
Da can bang lop Normal bang cach giam xuong ~20.000 dong de tuong duong quy mo trung binh cac lop tan cong.
Phan chia: 80% train / 20% test (Stratified).

Chay:
    python src/train_full_unsw.py --all
"""
import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tabnet_training import train_tabnet
from src.unsw_data import load_and_balance_unsw

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results" / "unsw_nb15"
MODEL_DIR = BASE_DIR / "models" / "unsw_nb15"

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (Actual)")
    plt.title(f"Confusion Matrix (10 lop) - {title}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_and_eval(name, fit_fn, pred_fn, Xte, yte_text, classes):
    t0 = time.time()
    model_obj = fit_fn()
    train_time = time.time() - t0

    t1 = time.time()
    pred_labels = pred_fn(model_obj, Xte)
    pred_time = time.time() - t1

    acc = accuracy_score(yte_text, pred_labels)
    pw = precision_score(yte_text, pred_labels, average="weighted", zero_division=0)
    rw = recall_score(yte_text, pred_labels, average="weighted", zero_division=0)
    f1w = f1_score(yte_text, pred_labels, average="weighted", zero_division=0)
    pm = precision_score(yte_text, pred_labels, average="macro", zero_division=0)
    rm = recall_score(yte_text, pred_labels, average="macro", zero_division=0)
    f1m = f1_score(yte_text, pred_labels, average="macro", zero_division=0)

    metrics = {
        "model": name,
        "accuracy": round(acc, 6),
        "precision_weighted": round(pw, 6),
        "recall_weighted": round(rw, 6),
        "f1_weighted": round(f1w, 6),
        "precision_macro": round(pm, 6),
        "recall_macro": round(rm, 6),
        "f1_macro": round(f1m, 6),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(pred_time, 2),
    }

    report = classification_report(yte_text, pred_labels, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    per_class_df.insert(0, "model", name)
    per_class_df = per_class_df[per_class_df["class"].isin(classes)]

    print(f"\n[{name}]")
    print(f"  Accuracy : {acc * 100:.4f}%")
    print(f"  F1 (w)   : {f1w * 100:.4f}% | F1 (macro): {f1m * 100:.4f}%")
    print(f"  Thoi gian: Train {train_time:.1f}s | Predict {pred_time:.1f}s")

    cm_slug = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "TabNet": "tabnet",
    }.get(name, name)
    plot_confusion_matrix(yte_text, pred_labels, classes, name, OUT_DIR / f"cm_{cm_slug}.png")

    return metrics, per_class_df, model_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=True, help="Huan luyen ca 6 mo hinh")
    parser.add_argument("--normal-samples", type=int, default=20000, help="So dong giam cho lop Normal")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, y_raw = load_and_balance_unsw(normal_samples=args.normal_samples)
    feature_names = list(X_raw.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"2. 10 LOP DU LIEU: {classes}")

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_raw.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

    X_train_df, X_test_df, y_train_enc, y_test_enc = train_test_split(
        X_raw, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    y_train_text = le.inverse_transform(y_train_enc)
    y_test_text = le.inverse_transform(y_test_enc)

    print(f"3. PHAN CHIA 80/20 (STRATIFIED):")
    print(f"   • Tap Train: {len(X_train_df):,} dong (80%)")
    print(f"   • Tap Test : {len(X_test_df):,} dong (20%)")
    print("=" * 65)

    print("Dang tien xu ly dac trung (Imputer + Scaler + One-Hot)...")
    X_train_proc = preprocessor.fit_transform(X_train_df).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test_df).astype(np.float32)
    joblib.dump({"preprocessor": preprocessor, "features": feature_names, "classes": classes},
                MODEL_DIR / "preprocessor_bundle.joblib", compress=3)
    print(f"   • Kich thuoc ma tran sau bien doi: {X_train_proc.shape[1]} dac trung so")

    all_metrics = []
    all_per_class = []

    # 1. Decision Tree (Rất nhanh)
    print("\n>>> [1/6] Huan luyen Decision Tree (10 lop)...")
    def fit_dt():
        dt = DecisionTreeClassifier(max_depth=20, random_state=42)
        dt.fit(X_train_proc, y_train_text)
        return dt
    def pred_dt(m, Xt):
        return m.predict(Xt)
    m, pc, obj = train_and_eval("DecisionTree", fit_dt, pred_dt, X_test_proc, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    joblib.dump({"model": obj, "preprocessor": preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "unsw_nb15_baseline_DecisionTree.joblib", compress=3)

    # 2. XGBoost (Rất nhanh với hist)
    print("\n>>> [2/6] Huan luyen XGBoost (10 lop)...")
    def fit_xgb():
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train_proc, y_train_enc)
        return xgb
    def pred_xgb(m, Xt):
        preds_enc = m.predict(Xt)
        return le.inverse_transform(preds_enc)
    m, pc, obj = train_and_eval("XGBoost", fit_xgb, pred_xgb, X_test_proc, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    joblib.dump({"model": obj, "preprocessor": preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "xgboost_attack.joblib", compress=3)

    # 3. Logistic Regression
    print("\n>>> [3/6] Huan luyen Logistic Regression (10 lop)...")
    def fit_lr():
        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.fit(X_train_proc, y_train_text)
        return lr
    def pred_lr(m, Xt):
        return m.predict(Xt)
    m, pc, obj = train_and_eval("LogisticRegression", fit_lr, pred_lr, X_test_proc, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    joblib.dump({"model": obj, "preprocessor": preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "unsw_nb15_baseline_LogisticRegression.joblib", compress=3)

    # 4. Random Forest
    print("\n>>> [4/6] Huan luyen Random Forest (10 lop)...")
    def fit_rf():
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train_proc, y_train_text)
        return rf
    def pred_rf(m, Xt):
        return m.predict(Xt)
    m, pc, obj = train_and_eval("Random Forest", fit_rf, pred_rf, X_test_proc, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    joblib.dump({"model": obj, "preprocessor": preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "random_forest_attack.joblib", compress=3)

    # 5. KNN (k=5)
    print("\n>>> [5/6] Huan luyen KNN (10 lop)...")
    def fit_knn():
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train_proc, y_train_text)
        return knn
    def pred_knn(m, Xt):
        return m.predict(Xt)
    m, pc, obj = train_and_eval("KNN", fit_knn, pred_knn, X_test_proc, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    joblib.dump({"model": obj, "preprocessor": preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "unsw_nb15_baseline_KNN.joblib", compress=3)

    # 6. TabNet (Deep Learning 10 lớp)
    print("\n>>> [6/6] Huan luyen TabNet (Deep Learning cho du lieu bang - 10 lop)...")
    tabnet_preprocessor = None
    tabnet_info = {}
    def fit_tabnet():
        nonlocal tabnet_preprocessor
        clf, tabnet_preprocessor, info = train_tabnet(X_train_df, y_train_enc)
        tabnet_info.update(info)
        return clf
    def pred_tabnet(m, Xt):
        preds_enc = m.predict(tabnet_preprocessor.transform(Xt).astype(np.float32))
        return le.inverse_transform(preds_enc)
    m, pc, obj = train_and_eval("TabNet", fit_tabnet, pred_tabnet, X_test_df, y_test_text, classes)
    all_metrics.append(m)
    all_per_class.append(pc)
    obj.save_model(str(MODEL_DIR / "tabnet_attack"))
    joblib.dump({"preprocessor": tabnet_preprocessor, "classes": classes, "features": feature_names},
                MODEL_DIR / "tabnet_bundle.joblib", compress=3)

    # ----------------- TONG HOP KET QUA -----------------
    print("\n" + "=" * 65)
    print("4. TONG HOP VA LUU TRU KET QUA UNSW-NB15 (10 LOP)...")
    df_metrics = pd.DataFrame(all_metrics).sort_values("accuracy", ascending=False)
    print("\n=== BANG SO SANH DO CHINH XAC 6 MO HINH UNSW-NB15 (10 LOP) ===")
    print(df_metrics[["model", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "train_time_s", "predict_time_s"]].to_string(index=False))

    baseline_names = {"LogisticRegression", "DecisionTree", "KNN", "Random Forest", "XGBoost"}
    df_base = df_metrics[df_metrics["model"].isin(baseline_names)]
    df_base.to_csv(OUT_DIR / "baselines_metrics.csv", index=False)
    base_pc = pd.concat([pc for pc in all_per_class if pc["model"].iloc[0] in baseline_names], ignore_index=True)
    base_pc.to_csv(OUT_DIR / "baselines_per_class.csv", index=False)
    with open(OUT_DIR / "baselines_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "UNSW-NB15 (10 loai: Normal + 9 Attack Categories)",
            "rows_used": len(X_raw),
            "features": len(feature_names),
            "classes": classes,
            "train": len(X_train_df),
            "test": len(X_test_df),
            "split": "80/20 Stratified",
        }, f, ensure_ascii=False, indent=2)

    df_rf_xgb = df_metrics[df_metrics["model"].isin({"Random Forest", "XGBoost"})]
    df_rf_xgb.to_csv(OUT_DIR / "random_forest_xgboost_metrics.csv", index=False)
    rf_xgb_pc = pd.concat([pc for pc in all_per_class if pc["model"].iloc[0] in {"Random Forest", "XGBoost"}], ignore_index=True)
    rf_xgb_pc.to_csv(OUT_DIR / "random_forest_xgboost_per_class.csv", index=False)
    with open(OUT_DIR / "random_forest_xgboost_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "UNSW-NB15 (10 loai: Normal + 9 Attack Categories)",
            "rows_used": len(X_raw),
            "features": len(feature_names),
            "classes": classes,
            "train": len(X_train_df),
            "test": len(X_test_df),
            "split": "80/20 Stratified",
        }, f, ensure_ascii=False, indent=2)

    df_tabnet = df_metrics[df_metrics["model"] == "TabNet"]
    df_tabnet.to_csv(OUT_DIR / "tabnet_metrics.csv", index=False)
    tab_pc = pd.concat([pc for pc in all_per_class if pc["model"].iloc[0] == "TabNet"], ignore_index=True)
    tab_pc.to_csv(OUT_DIR / "tabnet_per_class.csv", index=False)
    with open(OUT_DIR / "tabnet_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "UNSW-NB15 (10 loai: Normal + 9 Attack Categories)",
            "rows_used": len(X_raw),
            "features": len(feature_names),
            "classes": classes,
            "train": len(X_train_df),
            "test": len(X_test_df),
            "split": "80/20 Stratified",
            "architecture": "TabNet (Attentive Tabular Deep Learning)",
            **tabnet_info,
        }, f, ensure_ascii=False, indent=2)

    df_metrics.to_csv(OUT_DIR / "all_models_metrics.csv", index=False)
    print(f"\nDa luu thanh cong toan bo ket qua vao: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
