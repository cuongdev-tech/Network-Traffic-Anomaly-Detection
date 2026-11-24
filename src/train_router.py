
"""Train the optional router from real, independently annotated flow JSONL files.

Each record needs ground_truth and capture_group. Never train from predictions
or a dataset-name label. Group-disjoint test is reserved for evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from src.attack_router import (
    DEFAULT_ROUTER, FEATURES, ROUTER_VERSION, ROUTES, choose_route,
    common_features, target_route,
)


def load_rows(paths):
    rows, labels, groups = [], [], []
    seen = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") not in {"flow_alert", "flow_decision"}:
                    continue
                # Unlabeled traffic must not silently become Normal.
                if "ground_truth" not in event or not event.get("capture_group"):
                    continue
                group = str(event["capture_group"]).strip()
                if not group:
                    raise ValueError("Empty capture_group")
                key = (group, event.get("flow_start_utc"), event.get("flow_end_utc"),
                       event.get("src"), event.get("dst"), event.get("sport"),
                       event.get("dport"), event.get("protocol"))
                if key in seen:
                    raise ValueError(f"Duplicate annotated flow: {key}")
                seen.add(key)
                rows.append(common_features(event))
                labels.append(target_route(event["ground_truth"]))
                groups.append(group)
    if not rows:
        raise ValueError("No valid annotated flows. Add independently verified ground truth first.")
    return pd.DataFrame(rows, columns=FEATURES), np.asarray(labels), np.asarray(groups)


def split_groups(y, groups):
    """Choose a deterministic group-disjoint 70/30 split with class coverage."""
    X = np.zeros(len(y))
    if len(set(groups)) < 2 or set(y) != set(ROUTES):
        raise ValueError("Need multiple independent captures covering all three routing targets")
    splits = GroupShuffleSplit(n_splits=100, test_size=0.30, random_state=42)
    for train, test in splits.split(X, y, groups):
        if set(y[train]) == set(ROUTES) and set(y[test]) == set(ROUTES):
            return train, test
    raise ValueError("No group-disjoint split with all routing targets; collect more independent captures")


def train_router(paths, output, threshold=0.8):
    if not 0.5 < threshold <= 1:
        raise ValueError("Threshold must be in (0.5, 1]")
    output = Path(output)
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a new output path; do not overwrite a trained artifact")
    X, y, groups = load_rows(paths)
    train, test = split_groups(y, groups)
    model = RandomForestClassifier(n_estimators=150, max_depth=12,
        min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=2)
    model.fit(X.iloc[train], y[train])
    raw = model.predict(X.iloc[test])
    routed = [choose_route(p, model.classes_, threshold)["route"]
              for p in model.predict_proba(X.iloc[test])]
    def covers(actual, chosen):
        return chosen == "both" or chosen == actual
    coverage = [covers(a, b) for a, b in zip(y[test], routed)]
    report = {
        "note": "Routing scores are not calibrated; independent group holdout, not live IDS accuracy",
        "train_rows": len(train), "test_rows": len(test),
        "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])),
        "train_targets": dict(Counter(map(str, y[train]))),
        "test_targets": dict(Counter(map(str, y[test]))), "threshold": threshold,
        "raw_report": classification_report(y[test], raw, labels=list(ROUTES), output_dict=True, zero_division=0),
        "routing_confusion": confusion_matrix(y[test], routed, labels=list(ROUTES)).tolist(),
        "route_order": list(ROUTES),
        "fallback_or_both_rate": routed.count("both") / len(routed),
        "target_coverage": float(np.mean(coverage)),
        "unsafe_exclusions": int(len(coverage) - sum(coverage)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"version": ROUTER_VERSION, "features": FEATURES, "model": model,
              "threshold": threshold, "report": report}
    # Explicit publication only: inspect the report before deploying the artifact.
    with output.open("xb") as stream:
        joblib.dump(bundle, stream)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help=f"Candidate artifact; after review, default deployment is {DEFAULT_ROUTER}")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    try:
        print(json.dumps(train_router(args.events, args.output, args.threshold), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
