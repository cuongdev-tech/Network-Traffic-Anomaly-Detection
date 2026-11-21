"""Huan luyen 5 mo hinh Machine Learning Baseline va mo hinh TabNet (Deep Learning)
tren toan bo 512.212 dong du lieu CIC-IDS2017 (DDoS + PortScan).
Sau khi loc bo gia tri loi inf/nan con 511.807 dong hop le.
Phan chia: 80% train (409.445 dong) / 20% test (102.362 dong) (Stratified).

Chay:
    python src/train_full_cic.py --all
    python src/train_full_cic.py --models TabNet
    python src/train_full_cic.py --models DecisionTree XGBoost TabNet
"""
import argparse
import json
import sys
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cic_data import load_cleaned_cic


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models" / "cic"

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (Actual)")
    plt.title(f"Confusion Matrix - {title}")
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

    # Ve confusion matrix
    cm_slug = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "TabNet": "tabnet",
    }.get(name, name)
    plot_confusion_matrix(yte_text, pred_labels, classes, name, OUT_DIR / f"cm_{cm_slug}.png")

    return metrics, per_class_df, model_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Huan luyen ca 5 baseline va TabNet")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Danh sach mo hinh muon chay: LogisticRegression DecisionTree KNN RandomForest XGBoost TabNet")
    args = parser.parse_args()

    target_models = args.models
    if args.all or not target_models:
        target_models = ["LogisticRegression", "DecisionTree", "KNN", "Random Forest", "XGBoost", "TabNet"]
    else:
        # Chuan hoa ten
        alias = {"rf": "Random Forest", "randomforest": "Random Forest", "xgb": "XGBoost", "xgboost": "XGBoost",
                 "lr": "LogisticRegression", "dt": "DecisionTree", "knn": "KNN", "tabnet": "TabNet"}
        target_models = [alias.get(m.lower(), m) for m in target_models]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cleaned_cic()
    y_raw = data["Label"].values
    X_raw = data.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])
    feature_names = list(X_raw.columns)
    print(f"2. SO LUONG DAC TRUNG: {len(feature_names)} dac trung so")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    classes = list(le.classes_)

    # Chia tap du lieu 80% train / 20% test (Stratified)
    X_train_df, X_test_df, y_train_enc, y_test_enc = train_test_split(
        X_raw, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    y_train_text = le.inverse_transform(y_train_enc)
    y_test_text = le.inverse_transform(y_test_enc)

    print(f"3. PHAN CHIA 80/20:")
    print(f"   • Tap Train: {len(X_train_df):,} dong (80%)")
    print(f"   • Tap Test : {len(X_test_df):,} dong (20%)")
    print("=" * 65)

    # Chuyen doi sang numpy float32 de tiet kiem bo nho va tang toc
    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    X_test_np = X_test_df.to_numpy(dtype=np.float32)

    # Chuan hoa cho cac mo hinh can scale (Logistic Regression, KNN, TabNet)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_np).astype(np.float32)

    all_metrics = []
    all_per_class = []

    # ----------------- 1. Logistic Regression -----------------
    if "LogisticRegression" in target_models:
        print("\n>>> Dang huan luyen [1/6] Logistic Regression (Baseline)...")
        def fit_lr():
            lr = LogisticRegression(max_iter=300, random_state=42, n_jobs=-1)
            lr.fit(X_train_scaled, y_train_text)
            return lr
        def pred_lr(m, Xt):
            return m.predict(X_test_scaled)

        m, pc, obj = train_and_eval("LogisticRegression", fit_lr, pred_lr, X_test_scaled, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        joblib.dump({"model": obj, "scaler": scaler, "classes": classes, "features": feature_names},
                    MODEL_DIR / "baseline_LogisticRegression.joblib", compress=3)

    # ----------------- 2. Decision Tree -----------------
    if "DecisionTree" in target_models:
        print("\n>>> Dang huan luyen [2/6] Decision Tree (Baseline)...")
        def fit_dt():
            dt = DecisionTreeClassifier(max_depth=20, random_state=42)
            dt.fit(X_train_np, y_train_text)
            return dt
        def pred_dt(m, Xt):
            return m.predict(Xt)

        m, pc, obj = train_and_eval("DecisionTree", fit_dt, pred_dt, X_test_np, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        joblib.dump({"model": obj, "classes": classes, "features": feature_names},
                    MODEL_DIR / "baseline_DecisionTree.joblib", compress=3)

    # ----------------- 3. KNN -----------------
    if "KNN" in target_models:
        print("\n>>> Dang huan luyen [3/6] KNN (Baseline, k=5)...")
        def fit_knn():
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            knn.fit(X_train_scaled, y_train_text)
            return knn
        def pred_knn(m, Xt):
            return m.predict(X_test_scaled)

        m, pc, obj = train_and_eval("KNN", fit_knn, pred_knn, X_test_scaled, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        joblib.dump({"model": obj, "scaler": scaler, "classes": classes, "features": feature_names},
                    MODEL_DIR / "baseline_KNN.joblib", compress=3)

    # ----------------- 4. Random Forest -----------------
    if "Random Forest" in target_models:
        print("\n>>> Dang huan luyen [4/6] Random Forest (Baseline)...")
        def fit_rf():
            rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            rf.fit(X_train_np, y_train_text)
            return rf
        def pred_rf(m, Xt):
            return m.predict(Xt)

        m, pc, obj = train_and_eval("Random Forest", fit_rf, pred_rf, X_test_np, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        joblib.dump({"model": obj, "classes": classes, "features": feature_names},
                    MODEL_DIR / "random_forest_attack.joblib", compress=3)

    # ----------------- 5. XGBoost -----------------
    if "XGBoost" in target_models:
        print("\n>>> Dang huan luyen [5/6] XGBoost (Baseline, hist)...")
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
            xgb.fit(X_train_np, y_train_enc)
            return xgb
        def pred_xgb(m, Xt):
            preds_enc = m.predict(Xt)
            return le.inverse_transform(preds_enc)

        m, pc, obj = train_and_eval("XGBoost", fit_xgb, pred_xgb, X_test_np, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        joblib.dump({"model": obj, "classes": classes, "features": feature_names},
                    MODEL_DIR / "xgboost_attack.joblib", compress=3)

    # ----------------- 6. TabNet (Deep Learning dạng bảng) -----------------
    if "TabNet" in target_models:
        print("\n>>> Dang huan luyen [6/6] TabNet (Deep Learning cho du lieu bang)...")
        # Chia 10% tap train de lam tap validation theo doi Early Stopping
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_train_scaled, y_train_enc, test_size=0.1, random_state=42, stratify=y_train_enc
        )

        def fit_tabnet():
            clf = TabNetClassifier(
                n_d=24,
                n_a=24,
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
            clf.fit(
                X_train=X_tr_sub,
                y_train=y_tr_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                eval_name=["val"],
                eval_metric=["accuracy"],
                max_epochs=15,
                patience=5,
                batch_size=4096,
                virtual_batch_size=256,
                drop_last=False,
            )
            return clf

        def pred_tabnet(m, Xt):
            preds_enc = m.predict(X_test_scaled)
            return le.inverse_transform(preds_enc)

        m, pc, obj = train_and_eval("TabNet", fit_tabnet, pred_tabnet, X_test_scaled, y_test_text, classes)
        all_metrics.append(m)
        all_per_class.append(pc)
        # Luu trong so mo hinh TabNet
        tabnet_save_path = MODEL_DIR / "tabnet_attack"
        obj.save_model(str(tabnet_save_path))
        # Luu them file bundle chua metadata va scaler
        joblib.dump({"classes": classes, "features": feature_names, "scaler": scaler},
                    MODEL_DIR / "tabnet_bundle.joblib", compress=3)

    # ----------------- GHI KET QUA -----------------
    print("\n" + "=" * 65)
    print("4. TONG HOP VA LUU TRU KET QUA...")
    df_metrics = pd.DataFrame(all_metrics).sort_values("accuracy", ascending=False)
    print("\n=== BANG SO SANH DO CHINH XAC TOAN BO MO HINH ===")
    print(df_metrics[["model", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "train_time_s", "predict_time_s"]].to_string(index=False))

    # 1. Luu file baselines_metrics.csv (chua 5 baseline ML)
    baseline_names = {"LogisticRegression", "DecisionTree", "KNN", "Random Forest", "XGBoost"}
    df_base = df_metrics[df_metrics["model"].isin(baseline_names)]
    if not df_base.empty:
        df_base.to_csv(OUT_DIR / "baselines_metrics.csv", index=False)
        base_pc = pd.concat([pc for pc in all_per_class if pc["model"].iloc[0] in baseline_names], ignore_index=True)
        base_pc.to_csv(OUT_DIR / "baselines_per_class.csv", index=False)
        with open(OUT_DIR / "baselines_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "CIC-IDS2017 Friday afternoon (DDoS + PortScan)",
                "rows_used": len(data),
                "features": len(feature_names),
                "classes": classes,
                "train": len(X_train_df),
                "test": len(X_test_df),
                "split": "80/20 Stratified",
            }, f, ensure_ascii=False, indent=2)

    # 2. Luu file tabnet_metrics.csv (chua TabNet)
    df_tabnet = df_metrics[df_metrics["model"] == "TabNet"]
    if not df_tabnet.empty:
        df_tabnet.to_csv(OUT_DIR / "tabnet_metrics.csv", index=False)
        tab_pc = pd.concat([pc for pc in all_per_class if pc["model"].iloc[0] == "TabNet"], ignore_index=True)
        tab_pc.to_csv(OUT_DIR / "tabnet_per_class.csv", index=False)
        with open(OUT_DIR / "tabnet_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "CIC-IDS2017 Friday afternoon (DDoS + PortScan)",
                "rows_used": len(data),
                "features": len(feature_names),
                "classes": classes,
                "train": len(X_train_df),
                "test": len(X_test_df),
                "split": "80/20 Stratified",
                "architecture": "TabNet (Attentive Tabular Deep Learning)",
                "n_d": 24, "n_a": 24, "n_steps": 3,
            }, f, ensure_ascii=False, indent=2)

    # 3. Luu file tong hop all_models_metrics.csv (gop ca TabNet va 5 baseline)
    combined_metrics = [df_metrics]
    if (OUT_DIR / "tabnet_metrics.csv").exists() and "TabNet" not in df_metrics["model"].values:
        combined_metrics.append(pd.read_csv(OUT_DIR / "tabnet_metrics.csv"))
    if (OUT_DIR / "baselines_metrics.csv").exists():
        existing_base = pd.read_csv(OUT_DIR / "baselines_metrics.csv")
        missing_base = existing_base[~existing_base["model"].isin(df_metrics["model"])]
        if not missing_base.empty:
            combined_metrics.append(missing_base)
    
    final_all = pd.concat(combined_metrics, ignore_index=True).drop_duplicates(subset=["model"], keep="first")
    final_all = final_all.sort_values("accuracy", ascending=False)
    final_all.to_csv(OUT_DIR / "all_models_metrics.csv", index=False)
    print(f"\nDa luu thanh cong toan bo ket qua vao: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
