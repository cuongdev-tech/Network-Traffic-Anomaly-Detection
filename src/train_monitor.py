"""Train the live monitor from shared CTU-13 NetFlow features on source-host splits."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.evaluate_ctu13_graph import SOURCE
from src.gaussian_anomaly import GaussianAnomalyDetector
from src.monitor_flows import PROTOCOLS
from src.monitor_model import BUNDLE, FEATURES, ROOT, alert_decision, classify, features


DATA = ROOT / "data" / "processed" / "stage4_ctu13.parquet"
RESULTS = ROOT / "results" / "stage4"
LABELS = ["normal", "rbot", "nsis_ay"]


def prepare(source, *, include_timing=False):
    captures = []
    for (capture, meta), attack in zip(SOURCE.items(), LABELS[1:]):
        path = source / meta["filename"]
        if sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"{path.name} does not match the documented CTU-13 file.")
        columns = ["SrcAddr", "Dur", "TotPkts", "Proto", "Label"]
        if include_timing:
            columns.append("StartTime")
        df = pd.read_csv(path, usecols=columns)
        labels = df["Label"].str.extract(r"(?i)\b(Normal|Botnet)\b", expand=False)
        keep = labels.notna() & df["Proto"].isin(PROTOCOLS)
        selected = df.loc[keep].rename(columns={"Dur": "duration_s", "TotPkts": "packets", "Proto": "protocol"})
        # Source aliases serve only to separate the partitions.
        identifiers, _ = pd.factorize(selected["SrcAddr"])
        selected["source_group"] = [f"{capture}-h{i}" for i in identifiers]
        selected["label"] = np.where(labels.loc[keep].str.lower().eq("normal"), "normal", attack)
        fields = ["source_group", "duration_s", "packets", "protocol", "label"]
        if include_timing:
            started = pd.to_datetime(selected["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
            selected["completed_s"] = (started - started.min()).dt.total_seconds() + selected["duration_s"]
            selected["capture"] = capture
            fields.extend(("capture", "completed_s"))
        captures.append(selected[fields])
    return pd.concat(captures, ignore_index=True)


def source_classes(rows):
    return (rows.assign(attack=rows["label"].ne("normal"))
            .sort_values("attack").groupby("source_group")["label"].last())


def fit_bundle(rows):
    normal = rows.loc[rows["label"].eq("normal")]
    normal_sources = sorted(normal["source_group"].unique())
    if len(normal_sources) < 2:
        raise ValueError("Gaussian calibration needs at least two sources with normal flow records.")
    rng = np.random.default_rng(42)
    calibration_ids = rng.permutation(normal_sources)[:max(1, round(0.2 * len(normal_sources)))]
    calibration = normal["source_group"].isin(calibration_ids)
    gaussian = GaussianAnomalyDetector().fit(features(normal.loc[~calibration]))
    scores = gaussian.score_samples(features(normal.loc[calibration]))
    classifier = RandomForestClassifier(
        n_estimators=120, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=1,
    ).fit(features(rows), rows["label"])
    return {
        "features": FEATURES, "gaussian": gaussian,
        "gaussian_threshold": float(np.quantile(scores, 0.99, method="higher")),
        "classifier": classifier, "sklearn_version": version("scikit-learn"),
    }


def source_folds(rows):
    sources = source_classes(rows)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (fit, test) in enumerate(folds.split(sources, sources), 1):
        train_rows = rows.loc[rows["source_group"].isin(sources.index[fit])]
        test_rows = rows.loc[rows["source_group"].isin(sources.index[test])]
        yield fold, sources.index[fit], sources.index[test], train_rows, test_rows


def cross_validate(rows):
    parts, details = [], []
    for fold, fit, test, train_rows, test_rows in source_folds(rows):
        results = classify(fit_bundle(train_rows), test_rows)
        results.index = test_rows.index
        parts.append(results)
        details.append({
            "fold": fold, "fit_sources": len(fit), "test_sources": len(test),
            "test_flows": len(test_rows), "test_by_label": test_rows["label"].value_counts().to_dict(),
        })
    return pd.concat(parts).reindex(rows.index), details


def run(rows):
    outcome, folds = cross_validate(rows)
    bundle = fit_bundle(rows)
    actual = rows["label"]
    alert = alert_decision(outcome["label"], outcome["gaussian_flag"])
    attack = actual.ne("normal")
    tn, fp, fn, tp = confusion_matrix(attack, alert, labels=[False, True]).ravel()
    classification = classification_report(
        actual, outcome["label"], labels=LABELS, output_dict=True, zero_division=0,
    )
    metrics = {
        "source": "CTU-13 captures 11 and 12, TCP/UDP/ICMP, known labels only",
        "features": FEATURES, "gaussian_threshold": bundle["gaussian_threshold"], "sklearn_version": bundle["sklearn_version"],
        "validation": "Three stratified source-host folds. All reported predictions are held-source predictions.",
        "sources": int(rows["source_group"].nunique()), "flows": len(rows),
        "folds": folds,
        "accuracy_multiclass": float((outcome["label"] == actual).mean()),
        "macro_f1_multiclass": f1_score(actual, outcome["label"], labels=LABELS, average="macro"),
        "gaussian_only_alerts": int(outcome["gaussian_flag"].sum()),
        "combined_alerts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE, compress=3)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "monitor_holdout.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(classification).T.to_csv(RESULTS / "monitor_holdout_per_class.csv", float_format="%.8f")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Rebuild example features from the two original CTU-13 files.")
    args = parser.parse_args()
    if args.source_dir is not None:
        rows = prepare(args.source_dir)
        rows.to_parquet(DATA, index=False)
    else:
        rows = pd.read_parquet(DATA)
    run(rows)


if __name__ == "__main__":
    main()
