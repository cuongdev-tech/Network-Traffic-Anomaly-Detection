"""Train Random Forest and XGBoost on CIC-IDS2017 and UNSW-NB15.

Models and results are stored per dataset:
    models/cic/random_forest_attack.joblib
    models/cic/xgboost_attack.joblib
    models/unsw_nb15/random_forest_attack.joblib
    models/unsw_nb15/xgboost_attack.joblib

Run:
    python src/train_random_forest_xgboost.py
    python src/train_random_forest_xgboost.py --dataset cic
    python src/train_random_forest_xgboost.py --dataset unsw
    python src/train_random_forest_xgboost.py --dataset unsw --sample 0
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
CIC_DATA = BASE_DIR / "data" / "processed" / "cicids_60k.csv.gz"
UNSW_DATA_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
UNSW_DROP_FEATURES = {
    "label",
    "attack_cat",
    "id",
    "srcip",
    "dstip",
    "stime",
    "ltime",
}
TEST_SIZE = 0.2
SEED = 42
MODEL_FILENAMES = {
    "Random Forest": "random_forest_attack.joblib",
    "XGBoost": "xgboost_attack.joblib",
}


def read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Dinh dang du lieu khong duoc ho tro: {path}")


def normalize_unsw_labels(labels, source):
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


def load_cic():
    if not CIC_DATA.is_file():
        raise FileNotFoundError(
            f"Khong tim thay {CIC_DATA}. Hay chay `python src/prepare_cicids.py` truoc."
        )

    frame = pd.read_csv(CIC_DATA)
    if "Label" not in frame.columns:
        raise ValueError(f"Du lieu CIC-IDS2017 can cot 'Label': {CIC_DATA}")

    labels = frame["Label"].astype(str).str.strip()
    features = frame.drop(columns=["Label", "source_file"], errors="ignore")
    features = features.select_dtypes(include=[np.number])
    if features.empty:
        raise ValueError(f"Khong tim thay dac trung so trong {CIC_DATA}")

    return (
        features,
        labels,
        "CIC-IDS2017 Friday afternoon (DDoS + PortScan)",
        CIC_DATA,
    )


def load_unsw(sample):
    train_path = UNSW_DATA_DIR / "UNSW_NB15_training-set.parquet"
    test_path = UNSW_DATA_DIR / "UNSW_NB15_testing-set.parquet"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay cap file UNSW-NB15 trong {UNSW_DATA_DIR}."
        )

    frames = []
    for path in (train_path, test_path):
        frame = read_table(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "label" not in frame.columns:
            raise ValueError(f"Khong tim thay cot 'label' trong {path}")
        frame["label"] = normalize_unsw_labels(frame["label"], path)
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    if sample and sample < len(frame):
        frame, _ = train_test_split(
            frame,
            train_size=sample,
            random_state=SEED,
            stratify=frame["label"],
        )

    features = frame.drop(columns=list(UNSW_DROP_FEATURES), errors="ignore")
    labels = frame["label"]
    if features.empty:
        raise ValueError(f"Khong tim thay dac trung trong {UNSW_DATA_DIR}")

    source = f"{train_path.name} + {test_path.name}"
    return features, labels, "UNSW-NB15 (BENIGN vs ATTACK)", source


def make_unsw_preprocessor(features):
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    transformers = []

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
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


def build_models(features, n_classes, use_preprocessor):
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"

    def wrap(classifier):
        if not use_preprocessor:
            return classifier
        return Pipeline([
            ("preprocessor", make_unsw_preprocessor(features)),
            ("classifier", classifier),
        ])

    return {
        "Random Forest": wrap(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "XGBoost": wrap(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric=eval_metric,
                random_state=SEED,
                n_jobs=-1,
            )
        ),
    }


def model_slug(name):
    return name.lower().replace(" ", "_")


def evaluate(name, model, X_train, X_test, y_train, y_test, labels, out_dir):
    started = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - started

    started = time.time()
    prediction = model.predict(X_test)
    predict_time = time.time() - started

    metrics = {
        "model": name,
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
        "f1_macro": f1_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(predict_time, 2),
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            prediction,
            labels=range(len(labels)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", name)
    report = report[report["class"].isin(labels)]

    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(
        f"Precision: {metrics['precision_weighted']:.4f} (weighted) | "
        f"{precision_score(y_test, prediction, average='macro', zero_division=0):.4f} (macro)"
    )
    print(
        f"Recall   : {metrics['recall_weighted']:.4f} (weighted) | "
        f"{recall_score(y_test, prediction, average='macro', zero_division=0):.4f} (macro)"
    )
    print(
        f"F1       : {metrics['f1_weighted']:.4f} (weighted) | "
        f"{metrics['f1_macro']:.4f} (macro)"
    )
    print(f"Train {train_time:.1f}s | Predict {predict_time:.1f}s")
    print(
        classification_report(
            y_test,
            prediction,
            labels=range(len(labels)),
            target_names=labels,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, prediction, labels=range(len(labels)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{model_slug(name)}.png", dpi=150)
    plt.close()

    return metrics, report


def run_dataset(dataset_key, sample):
    if dataset_key == "cic":
        features, labels_text, title, source = load_cic()
        out_dir = BASE_DIR / "results"
        model_dir = BASE_DIR / "models" / "cic"
        use_preprocessor = False
    else:
        features, labels_text, title, source = load_unsw(sample)
        out_dir = BASE_DIR / "results" / "unsw_nb15"
        model_dir = BASE_DIR / "models" / "unsw_nb15"
        use_preprocessor = True

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text)
    labels = list(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    print(f"\n{'=' * 60}")
    print(title)
    print(f"Source: {source}")
    print(f"Rows: {len(features):,} | Features: {features.shape[1]}")
    print(f"Classes: {dict(zip(labels, np.bincount(y)))}")
    print(f"Train: {len(X_train):,} (80%) | Test: {len(X_test):,} (20%)")

    results = []
    reports = []
    for name, model in build_models(features, len(labels), use_preprocessor).items():
        metrics, report = evaluate(
            name, model, X_train, X_test, y_train, y_test, labels, out_dir
        )
        metrics["dataset"] = dataset_key
        results.append(metrics)
        reports.append(report)

        model_path = model_dir / MODEL_FILENAMES[name]
        joblib.dump(
            {
                "model": model,
                "labels": labels,
                "features": list(features.columns),
                "dataset": title,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "metrics": metrics,
            },
            model_path,
            compress=3,
        )
        print(f"Saved model: {model_path}")

    results_frame = pd.DataFrame(results).sort_values(
        "f1_weighted", ascending=False
    )
    results_frame.to_csv(out_dir / "random_forest_xgboost_metrics.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(
        out_dir / "random_forest_xgboost_per_class.csv", index=False
    )

    ax = results_frame.set_index("model")[
        ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    ].plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Random Forest vs XGBoost - {title}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "random_forest_xgboost_compare.png", dpi=150)
    plt.close()

    meta_path = out_dir / "random_forest_xgboost_meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": title,
                "source": str(source),
                "rows": int(len(features)),
                "features": int(features.shape[1]),
                "classes": labels,
                "train": int(len(X_train)),
                "test": int(len(X_test)),
                "test_size": TEST_SIZE,
                "split_seed": SEED,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults saved to: {out_dir}")
    return results_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "cic", "unsw"],
        default="all",
        help="Dataset to train (default: both).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=60000,
        help="UNSW-NB15 total rows before the 80/20 split (0 = all rows).",
    )
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be zero or a positive integer")

    datasets = ["cic", "unsw"] if args.dataset == "all" else [args.dataset]
    summary = []
    for dataset_key in datasets:
        summary.append(run_dataset(dataset_key, args.sample))

    if len(summary) > 1:
        combined = pd.concat(summary, ignore_index=True)
        combined.to_csv(
            BASE_DIR / "results" / "random_forest_xgboost_all_metrics.csv",
            index=False,
        )
        print("\nCombined metrics:")
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
