"""Evaluate RealMLP on the established CIC-IDS2017 test split.

RealMLP's reference implementation lives in PyTabKit 1.8.0.
Run from the repository root: python -m src.train_realmlp_cic
"""
import argparse
import json
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import torch
from pytabkit.models.realmlp.model import RealMLPConfig
from pytabkit.models.realmlp.train import fit, predict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.train_full_cic import MODEL_DIR, OUT_DIR, load_cleaned_cic, plot_confusion_matrix
from src.unsw_data import split_for_tabr, test_index_hash


def run(epochs, threads):
    torch.set_num_threads(threads)
    frame = load_cleaned_cic()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame.pop("Label"))
    classes = encoder.classes_.tolist()
    frame = frame.select_dtypes(include=[np.number])
    feature_names = frame.columns.tolist()
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    inner, validation, testing, _, y_val, y_test = split_for_tabr(positions, labels)
    split_hash = test_index_hash(testing)
    known = json.loads((OUT_DIR / "tabr_cic_trial" / "tabr_meta.json").read_text(encoding="utf-8"))
    if known["test_indices_sha256"] != split_hash or known["classes"] != classes:
        raise ValueError("The CIC test split differs from the established benchmark.")

    fit_rows = np.concatenate([inner.to_numpy(), validation.to_numpy()])
    x_fit = torch.from_numpy(np.ascontiguousarray(frame.iloc[fit_rows].to_numpy(dtype=np.float32)))
    y_fit = torch.from_numpy(labels[fit_rows, None])
    train_idx = torch.arange(len(inner)).unsqueeze(0)
    val_idx = torch.arange(len(inner), len(fit_rows)).unsqueeze(0)
    config = RealMLPConfig.td_class(
        n_epochs=epochs, hidden_width=128, plr_hidden_1=8, batch_size=1024,
        p_drop=0.05, use_ls=False, use_early_stopping=True,
        early_stopping_multiplicative_patience=1, early_stopping_additive_patience=6,
        verbosity=1,
    )
    print(f"RealMLP: {len(inner):,} fit, {len(validation):,} validation, {len(testing):,} held out", flush=True)
    started = time.perf_counter()
    model, selection = fit(
        config, x_fit, torch.empty((len(fit_rows), 0), dtype=torch.long), [], y_fit,
        train_idx, val_idx, n_classes=len(classes), seed=42, device="cpu",
    )
    train_seconds = time.perf_counter() - started
    with torch.inference_mode():
        val_prediction = predict(model, x_fit[len(inner):], torch.empty((len(validation), 0), dtype=torch.long))
        validation_accuracy = accuracy_score(y_val, val_prediction[0].argmax(dim=1).numpy())
    del x_fit

    x_test = torch.from_numpy(np.ascontiguousarray(frame.iloc[testing.to_numpy()].to_numpy(dtype=np.float32)))
    started = time.perf_counter()
    with torch.inference_mode():
        scores = predict(model, x_test, torch.empty((len(testing), 0), dtype=torch.long))
    predicted = scores[0].argmax(dim=1).numpy()
    prediction_seconds = time.perf_counter() - started
    metrics = {
        "model": "RealMLP", "accuracy": accuracy_score(y_test, predicted),
        "f1_macro": f1_score(y_test, predicted, average="macro"),
        "test_errors": int((y_test != predicted).sum()),
        "train_time_s": round(train_seconds, 2), "predict_time_s": round(prediction_seconds, 2),
    }
    baseline = known["baseline"]
    metadata = {
        "dataset": known["dataset"], "test_indices_sha256": split_hash, "classes": classes,
        "fit_train": len(inner), "validation": len(validation), "test": len(testing),
        "validation_accuracy": validation_accuracy, "selection": selection,
        "implementation": "PyTabKit 1.8.0 standalone RealMLP-TD with CPU-sized width and batch",
        "seed": 42, "threads": threads, "config": asdict(config), "baseline": baseline,
        "beats_baseline": {
            metric: round(metrics[metric], 6) > round(item["score"], 6)
            for metric, item in baseline.items()
        },
    }
    report = pd.DataFrame(classification_report(
        y_test, predicted, labels=np.arange(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )).T.loc[classes].reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", "RealMLP")
    destination = OUT_DIR / "realmlp_cic_trial"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": classes, "features": feature_names},
                MODEL_DIR / "realmlp_trial_checkpoint.joblib", compress=3)
    pd.DataFrame([metrics]).to_csv(destination / "realmlp_metrics.csv", index=False)
    report.to_csv(destination / "realmlp_per_class.csv", index=False)
    (destination / "realmlp_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_confusion_matrix(
        encoder.inverse_transform(y_test), encoder.inverse_transform(predicted), classes,
        "RealMLP", destination / "cm_realmlp.png",
    )
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print(f"Baseline comparison: {metadata['beats_baseline']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.threads < 1:
        parser.error("Epochs and threads must be positive.")
    run(args.max_epochs, args.threads)


if __name__ == "__main__":
    main()
