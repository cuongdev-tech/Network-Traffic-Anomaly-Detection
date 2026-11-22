"""Compare Gaussian and supervised detectors on the same held-out flows."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from giao_dien.prediction import load_bundle
from src.cic_data import load_cleaned_cic
from src.gaussian_anomaly import GaussianAnomalyDetector
from src.unsw_data import load_and_balance_unsw, split_for_tabr, test_index_hash


ROOT = Path(__file__).resolve().parents[1]


def binary_metrics(actual, predicted, scores):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision_attack": precision_score(actual, predicted, zero_division=0),
        "recall_attack": recall_score(actual, predicted, zero_division=0),
        "f1_attack": f1_score(actual, predicted, zero_division=0),
        "f1_macro_binary": f1_score(actual, predicted, average="macro", zero_division=0),
        "false_positive_rate": fp / (tn + fp),
        "roc_auc": roc_auc_score(actual, scores),
        "average_precision": average_precision_score(actual, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def supervised_predictions(dataset, name, test, normal_label):
    bundle = load_bundle(ROOT, dataset, name)
    features = test.loc[:, bundle["features"]]
    if "preprocessor" in bundle:
        values = bundle["preprocessor"].transform(features).astype(np.float32)
    else:
        values = features.to_numpy(dtype=np.float32)
    probabilities = bundle["model"].predict_proba(values)
    labels = np.array(bundle["labels"])
    predictions = labels[probabilities.argmax(axis=1)]
    return predictions, 1.0 - probabilities[:, labels == normal_label].ravel()


def run(dataset, output):
    if dataset == "cic":
        features = load_cleaned_cic(verbose=False)
        labels = features.pop("Label")
        normal_label = "BENIGN"
        results = ROOT / "results"
    else:
        features, labels = load_and_balance_unsw(verbose=False)
        normal_label = "Normal"
        results = ROOT / "results" / "unsw_nb15"
    fit, validation, test, y_fit, y_validation, y_test = split_for_tabr(features, labels)
    split_id = test_index_hash(test)
    published = json.loads((results / "tabr_meta.json").read_text(encoding="utf-8"))
    if split_id != published["test_indices_sha256"]:
        raise ValueError("The evaluation split does not match the saved TabR result.")

    actual = y_test.ne(normal_label).to_numpy()
    numeric = features.select_dtypes(include=np.number).columns
    normal_fit = fit.loc[y_fit.eq(normal_label), numeric]
    normal_validation = validation.loc[y_validation.eq(normal_label), numeric]
    del features, fit, validation
    start = perf_counter()
    detector = GaussianAnomalyDetector().fit(normal_fit)
    validation_scores = detector.score_samples(normal_validation)
    threshold = float(np.quantile(validation_scores, 0.99, method="higher"))
    build_seconds = perf_counter() - start
    start = perf_counter()
    scores = detector.score_samples(test.loc[:, numeric])
    predicted = scores > threshold
    score_seconds = perf_counter() - start
    rows = [{
        "dataset": dataset, "method": "Gaussian", **binary_metrics(actual, predicted, scores),
        "build_seconds": build_seconds, "predict_seconds": score_seconds,
    }]

    published_scores = pd.read_csv(results / "all_models_metrics.csv").set_index("model")
    for method in ("XGBoost", "TabR"):
        start = perf_counter()
        classified, attack_probability = supervised_predictions(dataset, method, test, normal_label)
        duration = perf_counter() - start
        if abs(accuracy_score(y_test, classified) - published_scores.loc[method, "accuracy"]) > 5e-6:
            raise ValueError(f"The {method} checkpoint does not match the published {dataset} accuracy.")
        # Collapse the selected multiclass prediction to normal vs attack.
        rows.append({
            "dataset": dataset, "method": method,
            **binary_metrics(actual, classified != normal_label, attack_probability),
            "build_seconds": None, "predict_seconds": None,
            "restore_and_predict_seconds": duration,
        })

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"{dataset}_flow_metrics.csv", index=False, float_format="%.8f")
    per_class = pd.DataFrame({"class": y_test.to_numpy(), "flagged": predicted})
    per_class.groupby("class")["flagged"].agg(
        test_samples="count", flagged="sum", flag_rate="mean",
    ).to_csv(output / f"{dataset}_gaussian_by_class.csv", float_format="%.8f")
    metadata = {
        "test_samples": len(test),
        "test_indices_sha256": split_id,
        "normal_fit": len(normal_fit),
        "normal_validation": len(normal_validation),
        "gaussian_features": list(numeric),
        "threshold": threshold,
        "threshold_quantile": 0.99,
        "validation_normal_flag_rate": float((validation_scores > threshold).mean()),
    }
    (output / f"{dataset}_gaussian_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(pd.DataFrame(rows)[["method", "accuracy", "f1_macro_binary", "false_positive_rate"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cic", "unsw"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage3")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
