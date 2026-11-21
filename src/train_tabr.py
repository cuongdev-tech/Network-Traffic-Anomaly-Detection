"""Benchmark CPU TabR against the saved UNSW-NB15 or CIC-IDS2017 baselines."""
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

from src.cic_data import load_cleaned_cic
from src.tabr_training import predict, sample_reference, tensors, train
from src.train_full_unsw import MODEL_DIR, OUT_DIR, plot_confusion_matrix as plot_unsw_confusion
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


def save_comparison(results_dir):
    baselines = pd.read_csv(results_dir / "baselines_metrics.csv")
    tabr = pd.read_csv(results_dir / "tabr_metrics.csv")
    comparison = pd.concat([baselines, tabr], ignore_index=True).sort_values("accuracy", ascending=False)
    comparison.to_csv(results_dir / "all_models_metrics.csv", index=False)


def run(args):
    results_dir = OUT_DIR
    checkpoint = MODEL_DIR / "tabr_checkpoint.joblib"
    plot_confusion = plot_unsw_confusion
    if args.dataset == "cic":
        from src.train_full_cic import MODEL_DIR as CIC_MODELS, OUT_DIR as CIC_RESULTS
        from src.train_full_cic import plot_confusion_matrix as plot_cic_confusion

        frame = load_cleaned_cic()
        labels = frame.pop("Label")
        features = frame.select_dtypes(include=[np.number])
        results_dir = CIC_RESULTS
        checkpoint = CIC_MODELS / "tabr_checkpoint.joblib"
        plot_confusion = plot_cic_confusion
    else:
        features, labels = load_and_balance_unsw()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()
    inner, validation, testing, y_inner, y_validation, y_test = split_for_tabr(features, y)
    split_hash = test_index_hash(testing)
    train_size = len(inner) + len(validation)
    if args.dataset == "cic":
        reference = json.loads((results_dir / "baselines_meta.json").read_text(encoding="utf-8"))
        if (reference["train"], reference["test"], reference["classes"]) != (train_size, len(testing), classes):
            raise ValueError("CIC baselines use a different split or class set.")
        baselines = pd.read_csv(results_dir / "baselines_metrics.csv").set_index("model")
        baseline_scores = {
            metric: {"model": baselines[metric].idxmax(), "score": float(baselines[metric].max())}
            for metric in ("accuracy", "f1_macro")
        }
    else:
        reference = json.loads((OUT_DIR / "tabm_meta.json").read_text(encoding="utf-8"))
        if reference["test_indices_sha256"] != split_hash or reference["classes"] != classes:
            raise ValueError("Expected the same evaluation rows and labels as the TabM experiment.")
        baseline_scores = reference["baseline"]
    if args.pilot:
        inner, _, y_inner, _ = train_test_split(
            inner, y_inner, train_size=20_000, stratify=y_inner, random_state=42,
        )
        validation, _, y_validation, _ = train_test_split(
            validation, y_validation, train_size=3_000, stratify=y_validation, random_state=42,
        )
    elif args.dataset == "cic":
        # Reuse the same CPU-sized stratified reference selection for training and inference.
        inner, y_inner = sample_reference(inner, y_inner, 90_000)
        print(f"CIC trial: {len(inner):,} fit/reference, {len(validation):,} validation, "
              f"{len(testing):,} held out", flush=True)
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
        "test_errors": int((y_test != predicted).sum()),
        "beats_baseline": {
            name: round(metrics[name], 6) > round(scores["score"], 6)
            for name, scores in baseline_scores.items()
        },
        "evaluation_note": "Historical split preserves existing duplicate-feature overlap. "
                           "Validation and test retrieval use only the inner training pool.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"weights": model.state_dict(), "preprocessor": preprocessor,
                 "classes": classes, "features": list(features), "input_width": x_train.shape[1]},
                checkpoint, compress=3)
    pd.DataFrame([metrics]).to_csv(results_dir / "tabr_metrics.csv", index=False)
    report.to_csv(results_dir / "tabr_per_class.csv", index=False)
    (results_dir / "tabr_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_confusion(
        labels=classes, y_true=encoder.inverse_transform(y_test), y_pred=encoder.inverse_transform(predicted),
        title="TabR", filename=results_dir / "cm_tabr.png",
    )
    save_comparison(results_dir)
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print(f"Baseline comparison: {metadata['beats_baseline']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unsw", "cic"), default="unsw")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.threads < 1:
        parser.error("Epochs and threads must be positive.")
    run(args)


if __name__ == "__main__":
    main()
