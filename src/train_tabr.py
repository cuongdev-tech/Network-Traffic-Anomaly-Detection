"""Benchmark CPU TabR on the historical 10-class UNSW-NB15 split."""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer

from src.tabr_training import predict, tensors, train
from src.train_full_unsw import MODEL_DIR, OUT_DIR, plot_confusion_matrix
from src.unsw_data import load_and_balance_unsw, split_for_tabr, test_index_hash


def make_preprocessor(frame):
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = frame.columns.difference(numeric).tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("quantile", QuantileTransformer(output_distribution="normal", random_state=42)),
        ]), numeric),
        ("categorical", Pipeline([
            ("missing", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def save_comparison():
    baselines = pd.read_csv(OUT_DIR / "baselines_metrics.csv")
    tabr = pd.read_csv(OUT_DIR / "tabr_metrics.csv")
    comparison = pd.concat([baselines, tabr], ignore_index=True).sort_values("accuracy", ascending=False)
    comparison.to_csv(OUT_DIR / "all_models_metrics.csv", index=False)


def run(args):
    features, labels = load_and_balance_unsw()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()
    inner, validation, testing, y_inner, y_validation, y_test = split_for_tabr(features, y)
    split_hash = test_index_hash(testing)
    train_size = len(inner) + len(validation)
    reference = json.loads((OUT_DIR / "tabm_meta.json").read_text(encoding="utf-8"))
    if reference["test_indices_sha256"] != split_hash or reference["classes"] != classes:
        raise ValueError("Expected the same evaluation rows and labels as the TabM experiment.")
    if args.pilot:
        inner, _, y_inner, _ = train_test_split(
            inner, y_inner, train_size=20_000, stratify=y_inner, random_state=42,
        )
        validation, _, y_validation, _ = train_test_split(
            validation, y_validation, train_size=3_000, stratify=y_validation, random_state=42,
        )
    preprocessor = make_preprocessor(inner)
    x_train = tensors(preprocessor.fit_transform(inner))
    x_val = tensors(preprocessor.transform(validation))
    y_train = torch.as_tensor(y_inner, dtype=torch.long)
    start = time.perf_counter()
    model, best, history = train(x_train, y_train, x_val, y_validation,
                                 epochs=args.max_epochs, threads=args.threads)
    train_time = time.perf_counter() - start
    if args.pilot:
        print(f"Validation pilot only. Best checkpoint: {best}", flush=True)
        return

    x_test = tensors(preprocessor.transform(testing))
    start = time.perf_counter()
    predicted = predict(model, x_train, y_train, x_test)
    prediction_time = time.perf_counter() - start
    metrics = {
        "model": "TabR", "accuracy": accuracy_score(y_test, predicted),
        "train_time_s": round(train_time, 2), "predict_time_s": round(prediction_time, 2),
    }
    for average in ("weighted", "macro"):
        for name, metric in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            metrics[f"{name}_{average}"] = metric(y_test, predicted, average=average, zero_division=0)
    baseline_scores = reference["baseline"]
    report = pd.DataFrame(classification_report(
        y_test, predicted, labels=np.arange(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )).T.loc[classes].reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", "TabR")
    metadata = {
        "dataset": reference["dataset"], "rows_used": len(features), "features": features.shape[1],
        "classes": classes, "train": train_size, "test": len(testing),
        "split": "80/20 Stratified", "split_seed": 42, "test_indices_sha256": split_hash,
        "fit_train": len(inner), "validation": len(validation),
        "architecture": "CPU TabR with approximate IVF-flat search and cached training contexts",
        "selection_metric": "validation accuracy", "max_epochs": args.max_epochs, "threads": args.threads,
        "selected": best, "history": history, "baseline": baseline_scores,
        "beats_baseline": {
            name: round(metrics[name], 6) > round(scores["score"], 6)
            for name, scores in baseline_scores.items()
        },
        "evaluation_note": "Historical split preserves existing duplicate-feature overlap. "
                           "Validation and test retrieval use only the inner training pool.",
    }
    joblib.dump({"weights": model.state_dict(), "preprocessor": preprocessor,
                 "classes": classes, "features": list(features), "input_width": x_train.shape[1]},
                MODEL_DIR / "tabr_checkpoint.joblib", compress=3)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "tabr_metrics.csv", index=False)
    report.to_csv(OUT_DIR / "tabr_per_class.csv", index=False)
    (OUT_DIR / "tabr_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_confusion_matrix(
        labels=classes, y_true=encoder.inverse_transform(y_test), y_pred=encoder.inverse_transform(predicted),
        title="TabR", filename=OUT_DIR / "cm_tabr.png",
    )
    save_comparison()
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print(f"Baseline comparison: {metadata['beats_baseline']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.threads < 1:
        parser.error("Epochs and threads must be positive.")
    run(args)


if __name__ == "__main__":
    main()
