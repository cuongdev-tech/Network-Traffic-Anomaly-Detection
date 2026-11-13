"""Train 5 baseline models on UNSW-NB15: LR, DT, KNN, Random Forest, XGBoost.

The official train/test split is preserved. By default, stratified subsets
totaling 60,000 rows are used to keep KNN runtime comparable to the CIC run.
"""
import argparse
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
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
OUT_DIR = BASE_DIR / "results" / "unsw_nb15"
MODEL_DIR = BASE_DIR / "models" / "unsw_nb15"
TRAIN_STEM = "UNSW_NB15_training-set"
TEST_STEM = "UNSW_NB15_testing-set"
DROP_FEATURES = {
    "label",
    "attack_cat",
    "id",
    "srcip",
    "dstip",
    "stime",
    "ltime",
}


def find_split_paths(data_dir):
    for extension in (".parquet", ".csv"):
        train_path = data_dir / f"{TRAIN_STEM}{extension}"
        test_path = data_dir / f"{TEST_STEM}{extension}"
        if train_path.is_file() and test_path.is_file():
            return train_path, test_path

    raise FileNotFoundError(
        f"Khong tim thay cap file UNSW-NB15 train/test trong {data_dir}. "
        "Tai UNSW_NB15_training-set.parquet va UNSW_NB15_testing-set.parquet "
        "tu https://www.kaggle.com/datasets/dhoogla/unswnb15 va dat vao thu muc nay. "
        "Neu dung ban CSV, dat hai file cung ten voi duoi .csv."
    )


def read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Dinh dang du lieu khong duoc ho tro: {path}")


def normalize_labels(labels, source):
    if labels.isna().any():
        raise ValueError(f"Cot label co gia tri thieu trong {source}")

    values = labels.astype(str).str.strip().str.lower()
    normalized = values.map({
        "0": "BENIGN",
        "1": "ATTACK",
        "benign": "BENIGN",
        "normal": "BENIGN",
        "attack": "ATTACK",
    })
    if normalized.isna().any():
        unknown = sorted(values[normalized.isna()].unique())
        raise ValueError(
            f"Chi ho tro nhan nhi phan 0/1 (BENIGN/ATTACK); "
            f"nhan khong hop le trong {source}: {unknown[:10]}"
        )
    return normalized


def sample_partition(frame, n_rows, seed):
    if n_rows <= 0 or n_rows >= len(frame):
        return frame
    sample, _ = train_test_split(
        frame,
        train_size=n_rows,
        random_state=seed,
        stratify=frame["label"],
    )
    return sample


def load_data(data_dir=DATA_DIR, sample=60000, seed=42):
    train_path, test_path = find_split_paths(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    train.columns = [str(column).strip().lower() for column in train.columns]
    test.columns = [str(column).strip().lower() for column in test.columns]

    for split_name, frame, path in (
        ("train", train, train_path),
        ("test", test, test_path),
    ):
        if "label" not in frame.columns:
            raise ValueError(f"Khong tim thay cot 'label' trong {path}")
        frame["label"] = normalize_labels(frame["label"], path)
        if frame.drop(columns=list(DROP_FEATURES), errors="ignore").empty:
            raise ValueError(f"Khong tim thay dac trung trong tap {split_name}: {path}")

    if sample > 0:
        train_rows = round(sample * 0.7)
        test_rows = sample - train_rows
        train = sample_partition(train, train_rows, seed)
        test = sample_partition(test, test_rows, seed + 1)

    print(f"Train file: {train_path} ({len(train):,} dong)")
    print(f"Test file : {test_path} ({len(test):,} dong)")
    print(f"Train labels: {train['label'].value_counts().to_dict()}")
    print(f"Test labels : {test['label'].value_counts().to_dict()}")
    return train, test


def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]
    transformers = []

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    return ColumnTransformer(transformers)


def build_models(X):
    def pipeline(classifier):
        return Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", classifier),
        ])

    return {
        "LogisticRegression": pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "DecisionTree": pipeline(
            DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42)
        ),
        "KNN": pipeline(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "RandomForest": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),
        "XGBoost": pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }


def evaluate(name, model, X_train, X_test, y_train, y_test, class_names=None):
    started = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - started

    started = time.time()
    prediction = model.predict(X_test)
    predict_time = time.time() - started

    metrics = {
        "model": name,
        "group": "baseline",
        "accuracy": accuracy_score(y_test, prediction),
        "precision_weighted": precision_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "precision_macro": precision_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(predict_time, 2),
    }

    report = pd.DataFrame(
        classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        )
    ).T.reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", name)
    valid = [str(c) for c in sorted(pd.unique(y_test))]
    report = report[report["class"].isin(valid)]
    if class_names is not None:
        report["class"] = report["class"].map(
            lambda c: class_names[int(c)] if str(c).lstrip("-").isdigit() else c
        )

    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(
        f"Precision: {metrics['precision_weighted']:.4f} (weighted) | "
        f"{metrics['precision_macro']:.4f} (macro)"
    )
    print(
        f"Recall   : {metrics['recall_weighted']:.4f} (weighted) | "
        f"{metrics['recall_macro']:.4f} (macro)"
    )
    print(
        f"F1       : {metrics['f1_weighted']:.4f} (weighted) | "
        f"{metrics['f1_macro']:.4f} (macro)"
    )
    print(f"Train {train_time:.1f}s | Predict {predict_time:.1f}s")
    print(classification_report(y_test, prediction, zero_division=0,
                                target_names=class_names if class_names else None))

    labels = sorted(pd.unique(y_test))
    matrix = confusion_matrix(y_test, prediction, labels=labels)
    tick_labels = ([class_names[label] for label in labels]
                   if class_names is not None else labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"UNSW-NB15 Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cm_baseline_{name.lower()}.png", dpi=150)
    plt.close()
    return metrics, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=60000,
        help="Total stratified rows sampled from official train/test (0 = use all).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation on the sampled training partition.",
    )
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be zero or a positive integer")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data(sample=args.sample)

    y_train_text = train["label"]
    y_test_text = test["label"]
    label_encoder = LabelEncoder()
    y_train = pd.Series(label_encoder.fit_transform(y_train_text), index=train.index)
    y_test = pd.Series(label_encoder.transform(y_test_text), index=test.index)
    class_names = [str(c) for c in label_encoder.classes_]
    X_train = train.drop(columns=list(DROP_FEATURES), errors="ignore")
    X_test = test.drop(columns=list(DROP_FEATURES), errors="ignore")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("Tap train va test khong co cung bo cot dac trung.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Ca hai tap train va test phai co du hai lop BENIGN/ATTACK.")

    print(f"Nhan: {dict(enumerate(class_names))}")

    print(f"So dac trung: {X_train.shape[1]}")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    results = []
    per_class = []
    cv_rows = []
    for name, model in build_models(X_train).items():
        metrics, report = evaluate(name, model, X_train, X_test, y_train, y_test,
                                   class_names=class_names)
        results.append(metrics)
        per_class.append(report)
        joblib.dump(
            {
                "model": model,
                "labels": class_names,
                "features": list(X_train.columns),
            },
            MODEL_DIR / f"unsw_nb15_baseline_{name}.joblib",
            compress=3,
        )

        if args.cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_validate(
                clone(model),
                X_train,
                y_train,
                cv=cv,
                n_jobs=1,
                scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            )
            row = {"model": name}
            for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
                values = scores[f"test_{key}"]
                row[f"cv_{key}_mean"] = round(float(values.mean()), 4)
                row[f"cv_{key}_std"] = round(float(values.std()), 4)
            cv_rows.append(row)

    results_frame = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    results_frame.to_csv(OUT_DIR / "baselines_metrics.csv", index=False)
    pd.concat(per_class, ignore_index=True).to_csv(
        OUT_DIR / "baselines_per_class.csv", index=False
    )
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(OUT_DIR / "baselines_cv.csv", index=False)

    results_frame.set_index("model")[
        ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    ].plot(kind="bar", figsize=(9, 5))
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("UNSW-NB15 Baselines (LR / DT / KNN / RF / XGBoost)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baselines_compare.png", dpi=150)
    plt.close()

    with (OUT_DIR / "baselines_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "UNSW-NB15 (Kaggle dhoogla/unswnb15)",
                "target": "label (BENIGN vs ATTACK)",
                "split": "official training-set/testing-set",
                "sampled_rows": int(len(train) + len(test)),
                "train": int(len(train)),
                "test": int(len(test)),
                "features": int(X_train.shape[1]),
                "classes": class_names,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDa luu ket qua tai: {OUT_DIR}")
    print(f"Da luu model UNSW-NB15 tai: {MODEL_DIR}")


if __name__ == "__main__":
    main()
