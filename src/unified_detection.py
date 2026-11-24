"""Run the two existing TabR checkpoints on independently verified feature rows."""
from importlib import import_module
from pathlib import Path
from math import isfinite

from src.cic_feature_extractor import extract_cic_features
from src.unsw_feature_extractor import extract_unsw_features
from src.dual_inference import DualInferenceEngine, ROOT

EXTRACTORS = {
    "cic": extract_cic_features,
    "unsw": extract_unsw_features,
}

TABR_MODE = "tabr_dual"


def load_provider(spec):
    """Only configure a trusted, separately validated module:function."""

    if not spec:
        return None

    module, sep, name = spec.partition(":")

    if not sep or not module or not name:
        raise ValueError(
            "Expected feature provider as module:function"
        )

    provider = getattr(import_module(module), name)

    if not callable(provider):
        raise ValueError("Feature provider must be callable")

    return provider


def summarize_branch(branch):

    for item in branch.get("models", []):

        if item.get("model") != "TabR":
            continue

        predictions = item.get("predictions", [])

        if (
            item.get("status") == "OK"
            and len(predictions) == 1
        ):

            pred = predictions[0]

            label = pred.get("label")
            confidence = pred.get("confidence")

            if (
                isinstance(label, str)
                and label
                and isinstance(confidence, (int, float))
                and isfinite(confidence)
                and 0 <= confidence <= 1
            ):

                return {
                    "status": "OK",
                    "candidate": label,
                    "confidence": confidence,
                    "completed": 1,
                    "required": 1,
                }

        return {
            "status": (
                item.get("status")
                if item.get("status") != "OK"
                else "ERROR"
            ),
            "candidate": None,
            "confidence": None,
            "completed": 0,
            "required": 1,
        }

    return {
        "status": branch.get(
            "status", "FEATURES_UNAVAILABLE"
        ),
        "candidate": None,
        "confidence": None,
        "completed": 0,
        "required": 1,
    }


def aggregate(dual):

    branches = {
        ds: summarize_branch(dual.get(ds, {}))
        for ds in EXTRACTORS
    }

    labels = sorted({
        s["candidate"]
        for s in branches.values()
        if s["candidate"]
    })

    completed = sum(
        s["completed"]
        for s in branches.values()
    )

    attack = any(
        label != "Normal"
        for label in labels
    )

    if completed == 0:

        status = "UNAVAILABLE"

    elif completed == 1:

        status = "INCOMPLETE"

    elif len(labels) > 1:

        status = "CONFLICT"

    else:

        status = (
            "ATTACK_CANDIDATE"
            if attack
            else "NORMAL_CANDIDATE"
        )

    return {
        "status": status,
        "candidate_labels": labels,
        "attack_candidate": attack,
        "branches": branches,
        "completed_models": completed,
        "required_models": 2,
    }


class UnifiedDetectionEngine:

    def __init__(
        self,
        root=ROOT,
        *,
        provider=None,
        threads=2,
        max_cached_models=2,
    ):

        self.provider = provider

        self.engine = DualInferenceEngine(
            Path(root),
            threads=threads,
            models=("TabR",),
            strict_versions=True,
            max_cached_models=max_cached_models,
        )

    def detect(self, event):

        rows, audits = {}, {}

        for dataset, extractor in EXTRACTORS.items():

            reference, error = None, None

            if self.provider is not None:

                try:

                    reference = self.provider(
                        event, dataset
                    )

                except Exception as exc:

                    error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            audit = extractor(
                event, reference
            )

            if error:

                audit.update(
                    status="PROVIDER_ERROR",
                    error=error,
                    model_features=None,
                )

            rows[dataset] = audit["model_features"]

            audits[dataset] = {
                k: v
                for k, v in audit.items()
                if k not in {
                    "candidate_features",
                    "model_features",
                }
            }

        dual = self.engine.predict_flow(
            event["flow_id"],
            cic_features=rows["cic"],
            unsw_features=rows["unsw"],
            route="both",
        )

        return {
            **dual,
            "mode": TABR_MODE,
            "feature_audit": audits,
            "decision": aggregate(dual),
        }
