"""Buoc 8 (Option A): Train RandomForest baseline tren labeled_flows.csv (11 mau demo)."""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

IN = r"D:\PATTT\data\processed\labeled_flows.csv"
MODEL = r"D:\PATTT\models\rf_attack.joblib"
FEATS = ["duration","n_pkts","bytes_total","byte_ratio_fwd_bwd","mean_iat","mean_size","syn","ack","rst","syn_no_ack"]

def main():
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    df = pd.read_csv(IN).fillna(0)
    X = df[FEATS]
    y = df["label"]
    print(f"Tong mau: {len(df)}, phan bo: {y.value_counts().to_dict()}")
    # Tap nho 11 mau -> test 30% stratify de demo
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train: {len(Xtr)}, Test: {len(Xte)}")
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    print(f"Accuracy: {accuracy_score(yte,pred):.3f}")
    print(f"Precision: {precision_score(yte,pred,zero_division=0):.3f}")
    print(f"Recall: {recall_score(yte,pred,zero_division=0):.3f}")
    print(f"F1: {f1_score(yte,pred,zero_division=0):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(yte,pred))
    print(classification_report(yte,pred,zero_division=0))
    print("Feature importance:")
    for f,i in sorted(zip(FEATS, clf.feature_importances_), key=lambda x:-x[1]):
        print(f"  {f}: {i:.3f}")
    joblib.dump({"model":clf,"scaler":sc,"feats":FEATS}, MODEL)
    print(f"Luu model: {MODEL}")
    print("Luu y: tap demo 11 mau chi de kiem thu pipeline, chua co y nghia thong ke.")

if __name__ == "__main__":
    main()
