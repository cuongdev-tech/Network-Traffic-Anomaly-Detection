"""Train only UNSW TabNet; preserve all five ML models and the 10 labels.

Run: python -m src.tune_tabnet_unsw
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.tabnet_training import split_validation, train_tabnet, validation_scores
from src.train_full_unsw import BASE_DIR, MODEL_DIR, OUT_DIR, load_and_balance_unsw, plot_confusion_matrix


def improves_validation(candidate, incumbent):
    return candidate["accuracy"] > incumbent["accuracy"] and candidate["f1_macro"] >= incumbent["f1_macro"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.threads < 1:
        parser.error("--max-epochs and --threads must be positive")
    torch.set_num_threads(args.threads)
    frame, labels = load_and_balance_unsw()
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    classes = list(encoder.classes_)
    training, testing, y_training, y_testing = train_test_split(
        frame, encoded, test_size=0.2, random_state=42, stratify=encoded)
    _, validation, _, y_validation = split_validation(training, y_training)

    # The incumbent used this same validation split in the original training script.
    from giao_dien.prediction import load_bundle
    incumbent = load_bundle(BASE_DIR, "unsw", "TabNet")
    if list(incumbent["labels"]) != classes or list(incumbent["features"]) != list(frame):
        raise ValueError("Incumbent classes/features differ; cannot compare this split")
    before = validation_scores(incumbent["model"], incumbent["preprocessor"], validation, y_validation)
    print("Incumbent validation:", before, flush=True)
    model, preprocessor, metadata = train_tabnet(training, y_training, args.max_epochs, args.threads)
    metadata["incumbent_validation_scores"] = before
    metadata["promoted"] = improves_validation(metadata["validation_scores"], before)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(model.history.history)).to_csv(OUT_DIR / "tabnet_tuning_history.csv", index=False)
    (OUT_DIR / "tabnet_tuning_validation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["promoted"]:
        print("Validation did not improve both accuracy and macro F1; keeping incumbent.", flush=True)
        return

    # Evaluate the chosen model on the unchanged test split once, after selection.
    started = time.perf_counter()
    predicted = model.predict(preprocessor.transform(testing).astype(np.float32))
    predict_time = time.perf_counter() - started
    metrics = {"model": "TabNet", "accuracy": round(accuracy_score(y_testing, predicted), 6)}
    for average in ("weighted", "macro"):
        for name, score in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            metrics[f"{name}_{average}"] = round(score(y_testing, predicted, average=average, zero_division=0), 6)
    metrics.update(train_time_s=metadata["train_time_s"], predict_time_s=round(predict_time, 2))
    report = pd.DataFrame(classification_report(
        y_testing, predicted, labels=np.arange(len(classes)), target_names=classes,
        output_dict=True, zero_division=0)).T.loc[classes].reset_index().rename(columns={"index": "class"})
    report.insert(0, "model", "TabNet")
    metadata.update(dataset="UNSW-NB15 (Normal + 9 Attack Categories)", rows_used=len(frame),
                    features=len(frame.columns), classes=classes, train=len(training), test=len(testing),
                    split="80/20 Stratified", architecture="TabNet (Attentive Tabular Deep Learning)",
                    evaluation_note="Same random split as baselines; duplicate flow overlap is not removed.")
    previous_metrics = pd.read_csv(OUT_DIR / "tabnet_metrics.csv")
    combined = pd.read_csv(OUT_DIR / "all_models_metrics.csv")
    combined = pd.concat([combined[combined.model != "TabNet"], pd.DataFrame([metrics])], ignore_index=True)
    combined = combined.sort_values("accuracy", ascending=False)
    comparison = pd.DataFrame([
        {"version": "before", **previous_metrics.iloc[0].to_dict()}, {"version": "after", **metrics}])

    # Stage complete artifacts before replacing the live model, and keep originals.
    with tempfile.TemporaryDirectory(prefix="tabnet_tuning_", dir=MODEL_DIR) as temporary:
        stage = Path(temporary)
        model.save_model(str(stage / "tabnet_attack"))
        joblib.dump({"preprocessor": preprocessor, "classes": classes, "features": list(frame)},
                    stage / "tabnet_bundle.joblib", compress=3)
        pd.DataFrame([metrics]).to_csv(stage / "tabnet_metrics.csv", index=False)
        report.to_csv(stage / "tabnet_per_class.csv", index=False)
        (stage / "tabnet_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        combined.to_csv(stage / "all_models_metrics.csv", index=False)
        comparison.to_csv(stage / "tabnet_improvement.csv", index=False)
        plot_confusion_matrix(encoder.inverse_transform(y_testing), encoder.inverse_transform(predicted),
                              classes, "TabNet", stage / "cm_tabnet.png")
        for name in ("tabnet_attack.zip", "tabnet_bundle.joblib", "tabnet_metrics.csv", "tabnet_per_class.csv",
                     "tabnet_meta.json", "all_models_metrics.csv", "tabnet_improvement.csv", "cm_tabnet.png"):
            destination = (MODEL_DIR if name.endswith((".zip", ".joblib")) else OUT_DIR) / name
            backup = destination.parent / "tabnet_before_tuning" / name
            if destination.exists() and not backup.exists():
                backup.parent.mkdir(exist_ok=True)
                shutil.copy2(destination, backup)
            shutil.copy2(stage / name, destination)
    print(comparison.to_string(index=False), flush=True)
    print("Updated model and results; original artifacts are in tabnet_before_tuning/.", flush=True)


if __name__ == "__main__":
    main()
