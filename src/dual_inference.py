
"""Dual-dataset inference for validated CIC/UNSW feature rows."""

import argparse
import gc
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.exceptions import InconsistentVersionWarning

from giao_dien.prediction import load_bundle


ROOT = Path(__file__).resolve().parents[1]

MODELS = (
    "LogisticRegression",
    "DecisionTree",
    "KNN",
    "Random Forest",
    "XGBoost",
    "TabR",
)

BRANCHES = {
    "cic": (
        77,
        {"BENIGN", "DDoS", "PortScan"},
        "mau_du_doan_cic.csv",
    ),
    "unsw": (
        34,
        {
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Normal",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        },
        "mau_du_doan_unsw.csv",
    ),
}

CATEGORICAL = {"proto", "service", "state"}


def checked_frame(frame, features, dataset):
    """Validate and arrange features in training order."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty DataFrame")

    if not frame.columns.is_unique:
        raise ValueError("Duplicate feature names")

    missing = [
        name for name in features
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing features: " + ", ".join(missing)
        )

    X = frame.loc[:, features].copy()

    for name in features:
        if dataset == "unsw" and name in CATEGORICAL:
            if X[name].isna().any():
                raise ValueError(
                    f"Missing categorical value: {name}"
                )

            X[name] = X[name].astype(str).str.strip()

            if X[name].eq("").any():
                raise ValueError(
                    f"Empty categorical value: {name}"
                )
        else:
            X[name] = pd.to_numeric(
                X[name], errors="coerce"
            )

            if not np.isfinite(
                X[name].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Missing/nonfinite numeric feature: {name}"
                )

    return X


def decode_label(value, labels):
    if isinstance(value, (int, np.integer)):
        if not 0 <= int(value) < len(labels):
            raise ValueError("Invalid encoded class")

        value = labels[int(value)]

    value = str(value)

    if value not in labels:
        raise ValueError(f"Unexpected class: {value}")

    return value


def predict_bundle(bundle, dataset, name, frame):
    """Predict using the checkpoint's fitted preprocessing."""

    model = bundle["model"]
    labels = list(bundle["labels"])

    batch = checked_frame(
        frame,
        list(bundle["features"]),
        dataset,
    )

    if "preprocessor" in bundle:
        batch = bundle["preprocessor"].transform(
            batch
        ).astype(np.float32)

    elif "scaler" in bundle:
        batch = bundle["scaler"].transform(
            batch.to_numpy(dtype=np.float32)
        ).astype(np.float32)

    elif (
        "classes" in bundle
        and not hasattr(model, "feature_names_in_")
    ):
        batch = batch.to_numpy(dtype=np.float32)

    expected = getattr(
        model,
        "n_features_in_",
        getattr(model, "input_dim", None),
    )

    if expected is not None and batch.shape[1] != expected:
        raise ValueError(
            f"Transformed width {batch.shape[1]} "
            f"!= expected {expected}"
        )

    if not np.isfinite(
        np.asarray(batch, dtype=float)
    ).all():
        raise ValueError("Nonfinite values after preprocessing")

    proba = np.asarray(
        model.predict_proba(batch),
        dtype=float,
    )

    class_order = (
        labels
        if name == "TabR"
        else [
            decode_label(value, labels)
            for value in model.classes_
        ]
    )

    if (
        proba.shape != (len(frame), len(class_order))
        or set(class_order) != set(labels)
        or not np.isfinite(proba).all()
        or (proba < -1e-6).any()
        or (proba > 1.000001).any()
        or not np.allclose(
            proba.sum(axis=1), 1, atol=1e-4
        )
    ):
        raise ValueError("Invalid class probability output")

    indices = proba.argmax(axis=1)

    return [
        {
            "label": (
                "Normal"
                if class_order[index] == "BENIGN"
                else class_order[index]
            ),
            "raw_label": class_order[index],
            "confidence": float(proba[row, index]),
        }
        for row, index in enumerate(indices)
    ]


class DualInferenceEngine:
    """Cache model bundles for a persistent inference worker."""

    def __init__(
        self,
        root=ROOT,
        *,
        threads=2,
        models=MODELS,
    ):
        self.root = Path(root)
        self.threads = max(1, int(threads))
        self.models = tuple(models)

        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or not set(self.models) <= set(MODELS)
        ):
            raise ValueError("Invalid model selection")

        self.bundles = {}
        self.schemas = {}

    def get_bundle(self, dataset, name):
        key = dataset, name

        if key not in self.bundles:
            count, classes, _ = BRANCHES[dataset]

            with warnings.catch_warnings(
                record=True
            ) as notices, threadpool_limits(
                limits=self.threads
            ):
                warnings.simplefilter(
                    "always",
                    InconsistentVersionWarning,
                )

                bundle = load_bundle(
                    self.root,
                    dataset,
                    name,
                )

            extra_warnings = sorted({
                str(w.message)
                for w in notices
                if isinstance(
                    w.message,
                    InconsistentVersionWarning,
                )
            })

            if extra_warnings:
                bundle["_version_warning"] = " | ".join(
                    filter(
                        None,
                        [
                            bundle.get("_version_warning", ""),
                            *extra_warnings,
                        ],
                    )
                )

            for notice in notices:
                if not isinstance(
                    notice.message,
                    InconsistentVersionWarning,
                ):
                    warnings.warn(
                        str(notice.message),
                        notice.category,
                    )

            features = list(bundle["features"])

            if (
                len(features) != count
                or len(set(features)) != count
            ):
                raise ValueError(
                    f"Expected {count} unique "
                    f"{dataset} features"
                )

            if (
                set(bundle["labels"]) != classes
                or len(bundle["labels"]) != len(classes)
            ):
                raise ValueError(
                    "Checkpoint class schema mismatch"
                )

            if (
                dataset in self.schemas
                and set(features) != self.schemas[dataset]
            ):
                raise ValueError(
                    "Checkpoint feature schema mismatch"
                )

            if hasattr(bundle["model"], "n_jobs"):
                bundle["model"].n_jobs = self.threads

            self.schemas[dataset] = set(features)
            self.bundles[key] = bundle

        return self.bundles[key]

    def infer_model(self, dataset, name, frame):
        started = perf_counter()

        try:
            bundle = self.get_bundle(dataset, name)

            load_ms = (
                perf_counter() - started
            ) * 1000

            started = perf_counter()

            with threadpool_limits(limits=self.threads):
                predictions = predict_bundle(
                    bundle,
                    dataset,
                    name,
                    frame,
                )

            return {
                "model": name,
                "status": "OK",
                "predictions": predictions,
                "inference_batch_ms": round(
                    (perf_counter() - started) * 1000,
                    3,
                ),
                "load_or_lookup_ms": round(load_ms, 3),
                "version_warning": bundle.get(
                    "_version_warning", ""
                ),
            }

        except Exception as exc:
            return {
                "model": name,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def infer_branch(self, dataset, frame):
        if frame is None:
            return {
                "status": "FEATURES_UNAVAILABLE",
                "models": [],
            }

        results = [
            self.infer_model(dataset, name, frame)
            for name in self.models
        ]

        successes = sum(
            result["status"] == "OK"
            for result in results
        )

        if successes == len(results):
            status = "OK"
        elif successes:
            status = "INCOMPLETE"
        else:
            status = "ERROR"

        return {
            "status": status,
            "models": results,
        }

    def predict_flow(
        self,
        flow_id,
        *,
        cic_features=None,
        unsw_features=None,
    ):
        """Both feature rows must describe the same live flow."""

        def row(values):
            return (
                None
                if values is None
                else pd.DataFrame([values])
            )

        return {
            "flow_id": flow_id,
            "cic": self.infer_branch(
                "cic", row(cic_features)
            ),
            "unsw": self.infer_branch(
                "unsw", row(unsw_features)
            ),
        }

    def release(self, dataset, name):
        self.bundles.pop((dataset, name), None)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        choices=("cic", "unsw", "both"),
        required=True,
    )

    parser.add_argument(
        "--limit", type=int, default=3
    )

    parser.add_argument(
        "--threads", type=int, default=2
    )

    parser.add_argument(
        "--resident",
        action="store_true",
        help="Keep model bundles loaded",
    )

    parser.add_argument(
        "--skip-tabr",
        action="store_true",
    )

    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.limit < 1 or args.threads < 1:
        parser.error(
            "--limit and --threads must be positive"
        )

    selected = (
        MODELS[:-1]
        if args.skip_tabr
        else MODELS
    )

    engine = DualInferenceEngine(
        threads=args.threads,
        models=selected,
    )

    results = []

    datasets = (
        tuple(BRANCHES)
        if args.dataset == "both"
        else (args.dataset,)
    )

    for dataset in datasets:
        count, _, filename = BRANCHES[dataset]

        path = ROOT / "giao_dien" / filename

        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            nrows=args.limit,
        )

        frame.columns = frame.columns.str.strip()

        if frame.shape[1] != count:
            raise ValueError(
                f"Expected {count} sample columns, "
                f"got {frame.shape[1]}"
            )

        for name in selected:
            print(
                f"[{dataset}] Loading/checking {name}...",
                flush=True,
            )

            result = engine.infer_model(
                dataset, name, frame
            )

            result.update(
                dataset=dataset,
                feature_count=count,
                sample_rows=len(frame),
            )

            results.append(result)

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )

            if not args.resident:
                engine.release(dataset, name)
                gc.collect()

    report = {
        "mode": "checkpoint_preflight",
        "note": (
            "Independent CSV samples, no ground truth, "
            "not a live-flow comparison"
        ),
        "passed": sum(
            r["status"] == "OK" for r in results
        ),
        "total": len(results),
        "version_warnings": sum(
            bool(r.get("version_warning"))
            for r in results
        ),
        "resident": args.resident,
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "x", encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    print(
        f"Preflight: {report['passed']}/{report['total']} OK; "
        f"version warnings: {report['version_warnings']}",
        flush=True,
    )

    return int(
        report["passed"] != report["total"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
