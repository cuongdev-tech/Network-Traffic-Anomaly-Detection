"""Benchmark the flow classifier with paced, pre-recorded NetFlow data in-process."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import platform
from time import perf_counter, sleep

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.monitor_model import ROOT, alert_decision, classify
from src.train_monitor import fit_bundle, prepare, source_folds


DATA = ROOT / "data" / "processed" / "stage4_replay_ctu13.csv.gz"
CASES = {"CTU-11": "Rbot", "CTU-12": "NSIS.ay"}


def feed(bundle, flows, *, fold, batch_size, cadence):
    """Release completed flows in trace order on a fixed, software-only schedule."""
    ordered = flows.sort_values("completed_s", kind="stable")
    start = perf_counter()
    for batch_number, offset in enumerate(range(0, len(ordered), batch_size)):
        batch = ordered.iloc[offset:offset + batch_size]
        due = start + batch_number * cadence
        delay = due - perf_counter()
        if delay > 0:
            sleep(delay)
        started = perf_counter()
        predicted = classify(bundle, batch)
        alert = alert_decision(predicted["label"], predicted["gaussian_flag"])
        finished = perf_counter()
        timing = {
            "capture": batch["capture"].iloc[0], "fold": fold, "batch": batch_number + 1,
            "flows": len(batch), "compute_ms": (finished - started) * 1000,
            "schedule_to_decision_ms": (finished - due) * 1000,
        }
        prediction = batch[["capture", "source_group", "label"]].rename(columns={"label": "actual"})
        prediction = prediction.assign(
            row=batch.index, fold=fold, decision=predicted["label"].to_numpy(),
            alert=alert.to_numpy(), schedule_to_decision_ms=timing["schedule_to_decision_ms"],
        )
        yield prediction, timing


def summarize(predictions):
    rows = []
    for capture, case in predictions.groupby("capture", sort=True):
        attack = case["actual"].ne("normal")
        alerts = case["alert"]
        tn, fp, fn, tp = confusion_matrix(attack, alerts, labels=[False, True]).ravel()
        p50, p95 = np.percentile(case["schedule_to_decision_ms"], [50, 95])
        rows.append({
            "capture": capture, "scenario": CASES[capture], "flows": len(case),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy_binary": (tp + tn) / len(case),
            "precision_attack": precision_score(attack, alerts, zero_division=0),
            "recall_attack": recall_score(attack, alerts, zero_division=0),
            "f1_attack": f1_score(attack, alerts, zero_division=0),
            "false_positive_rate": fp / (tn + fp),
            "p50_ms": p50, "p95_ms": p95,
        })
    return pd.DataFrame(rows)


def replay(rows, *, batch_size, cadence):
    predictions, timings, manifest = [], [], []
    for fold, fit_ids, test_ids, train, test in source_folds(rows):
        assert not set(fit_ids).intersection(test_ids)
        manifest.append({"fold": fold, "fit_sources": list(fit_ids), "test_sources": list(test_ids)})
        bundle = fit_bundle(train)
        for _, case in test.groupby("capture", sort=True):
            for prediction, timing in feed(bundle, case, fold=fold, batch_size=batch_size, cadence=cadence):
                predictions.append(prediction)
                timings.append(timing)
    combined = pd.concat(predictions).sort_values("row")
    if combined["row"].tolist() != rows.index.tolist():
        raise ValueError("Replay must score each input flow exactly once within its held-source fold.")
    return combined, pd.DataFrame(timings), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Rebuild the timed test fixture from the original .binetflow files.")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cadence-ms", type=float, default=100)
    args = parser.parse_args()
    if args.batch_size < 1 or not isfinite(args.cadence_ms) or args.cadence_ms <= 0:
        parser.error("Use a positive batch size and finite, positive cadence.")
    if args.source_dir:
        dataset = prepare(args.source_dir, include_timing=True)
        dataset.to_csv(args.data, index=False, compression={"method": "gzip", "mtime": 0})
    rows = pd.read_csv(args.data)
    if set(rows["capture"]) != set(CASES):
        parser.error("Replay requires both CTU-13 captures.")
    args.output.mkdir(parents=True, exist_ok=False)
    predictions, timing, manifest = replay(rows, batch_size=args.batch_size, cadence=args.cadence_ms / 1000)
    metrics = summarize(predictions)
    predictions.to_csv(args.output / "decisions.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0}, float_format="%.8f")
    timing.to_csv(args.output / "timing.csv", index=False, float_format="%.8f")
    metrics.to_csv(args.output / "metrics.csv", index=False, float_format="%.8f")
    environment = {
        "mode": "offline_paced_flow_replay", "distribution": platform.freedesktop_os_release().get("PRETTY_NAME")
        if platform.system() == "Linux" else platform.system(),
        "kernel": platform.release(), "python": platform.python_version(),
        "scikit_learn": version("scikit-learn"), "batch_size": args.batch_size,
        "cadence_ms": args.cadence_ms, "input_sha256": sha256(args.data.read_bytes()).hexdigest(),
        "folds": manifest,
        "timing": "Planned batch release to alert decisions, measured in-process. Includes schedule lag and scoring.",
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
