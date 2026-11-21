"""Train and compare TabM with the existing baseline split.

Run from the repository root: python -m src.train_tabm --dataset unsw
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.tabm_training import train_candidates
from src.tabnet_training import make_preprocessor, split_validation


ROOT = Path(__file__).resolve().parents[1]


def load_dataset(dataset):
    if dataset == "unsw":
        from src.train_full_unsw import load_and_balance_unsw, plot_confusion_matrix

        features, labels = load_and_balance_unsw()
        return features, labels, "UNSW-NB15 (Normal + 9 Attack Categories)", plot_confusion_matrix

    from src.train_full_cic import load_cleaned_cic, plot_confusion_matrix

    frame = load_cleaned_cic()
    features = frame.drop(columns="Label").select_dtypes(include=[np.number])
    return features, frame["Label"], "CIC-IDS2017 Friday afternoon (DDoS + PortScan)", plot_confusion_matrix


def run(dataset, max_epochs, selection_metric, threads):
    features, labels, title, plot_confusion_matrix = load_dataset(dataset)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()
    train, test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y,
    )
    model_dir = ROOT / "models" / ("unsw_nb15" if dataset == "unsw" else "cic")
    results_dir = ROOT / "results"
    if dataset == "unsw":
        results_dir /= "unsw_nb15"
    baseline_meta = json.loads((results_dir / "baselines_meta.json").read_text(encoding="utf-8"))
    if (baseline_meta.get("train"), baseline_meta.get("test"), baseline_meta.get("classes")) != (
        len(train), len(test), classes,
    ):
        raise ValueError("Baseline artifacts use a different split or label set.")
    baselines = pd.read_csv(results_dir / "baselines_metrics.csv").set_index("model")
    if set(baselines.index) != {"LogisticRegression", "DecisionTree", "KNN", "Random Forest", "XGBoost"}:
        raise ValueError("Expected results for all five baseline models.")

    inner_train, validation, y_inner, y_val = split_validation(train, y_train)
    preprocessor = make_preprocessor(inner_train)
    x_inner = preprocessor.fit_transform(inner_train).astype(np.float32)
    x_val = preprocessor.transform(validation).astype(np.float32)
    numeric_width = len(preprocessor.transformers_[0][2])
    cardinalities = []
    if preprocessor.transformers_[1][2]:
        ordinal = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cardinalities = [len(values) + 1 for values in ordinal.categories_]
    model, selected, candidates = train_candidates(
        x_inner, y_inner, x_val, y_val, numeric_width, cardinalities,
        max_epochs, selection_metric, threads,
    )
    x_test = preprocessor.transform(test).astype(np.float32)
    started = time.perf_counter()
    predicted = model.predict(x_test)
    elapsed = time.perf_counter() - started
    metrics = {
        "model": "TabM", "accuracy": accuracy_score(y_test, predicted),
        "train_time_s": selected["train_time_s"], "predict_time_s": round(elapsed, 2),
    }
    for average in ("weighted", "macro"):
        for name, metric in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            metrics[f"{name}_{average}"] = metric(y_test, predicted, average=average, zero_division=0)
    reference = {
        metric: {"model": baselines[metric].idxmax(), "score": float(baselines[metric].max())}
        for metric in ("accuracy", "f1_macro")
    }
    metadata = {
        "dataset": title, "rows_used": len(features), "features": features.shape[1], "classes": classes,
        "train": len(train), "test": len(test), "split": "80/20 Stratified", "split_seed": 42,
        "test_indices_sha256": hashlib.sha256(test.index.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "fit_train": len(inner_train), "validation": len(validation),
        "architecture": "TabM-mini with piecewise-linear numerical embeddings",
        "selection_metric": selection_metric, "selected": selected, "validation_candidates": candidates,
        "baseline": reference,
        "beats_baseline": {
            metric: round(metrics[metric], 6) > round(item["score"], 6)
            for metric, item in reference.items()
        },
        "evaluation_note": "Historical baseline split; duplicate flow overlap is not removed.",
    }
    report = pd.DataFrame(classification_report(
        y_test, predicted, labels=np.arange(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )).T.loc[classes].reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", "TabM")
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "classes": classes, "features": list(features)},
        model_dir / "tabm_bundle.joblib", compress=3,
    )
    pd.DataFrame([metrics]).to_csv(results_dir / "tabm_metrics.csv", index=False)
    report.to_csv(results_dir / "tabm_per_class.csv", index=False)
    (results_dir / "tabm_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_confusion_matrix(
        encoder.inverse_transform(y_test), encoder.inverse_transform(predicted),
        classes, "TabM", results_dir / "cm_tabm.png",
    )
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print("Baseline comparison:", metadata["beats_baseline"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unsw", "cic"), default="unsw")
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--selection-metric", choices=("accuracy", "f1_macro"), default="accuracy")
    args = parser.parse_args()
    run(args.dataset, args.max_epochs, args.selection_metric, args.threads)


if __name__ == "__main__":
    main()
