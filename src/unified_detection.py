"""Two TabR checkpoints with explicit experimental inference."""
import os
from importlib import import_module
from math import isfinite
from pathlib import Path

from src.cic_feature_extractor import extract_cic_features
from src.unsw_feature_extractor import extract_unsw_features
from src.verified_feature_provider import candidate_row
from src.dual_inference import DualInferenceEngine, ROOT


EXTRACTORS = {
    "cic": extract_cic_features,
    "unsw": extract_unsw_features,
}

TABR_MODE = "tabr_dual"

DEMO = "EXPERIMENTAL_UNVERIFIED"


def load_provider(spec):

    if not spec:
        return None

    module, sep, name = spec.partition(":")

    if not sep or not module or not name:

        raise ValueError(
            "Expected feature provider as module:function"
        )

    provider = getattr(
        import_module(module),
        name,
    )

    if not callable(provider):

        raise ValueError(
            "Feature provider must be callable"
        )

    return provider


# ==========================================
# SUMMARIZE MODEL RESULTS
# ==========================================

def summarize_branch(branch):

    result = {
        "status": branch.get(
            "status", "FEATURES_UNAVAILABLE"
        ),

        "candidate": None,
        "confidence": None,
        "completed": 0,
        "required": 1,

        "experimental": (
            branch.get("feature_quality") == DEMO
        ),
    }

    for item in branch.get("models", []):

        if item.get("model") != "TabR":
            continue

        predictions = item.get(
            "predictions", []
        )

        result["status"] = item.get(
            "status", "ERROR"
        )

        if (
            result["status"] == "OK"
            and len(predictions) == 1
        ):

            pred = predictions[0]

            label = pred.get("label")

            confidence = pred.get(
                "confidence"
            )

            if (
                isinstance(label, str)
                and label
                and isinstance(
                    confidence, (int, float)
                )
                and isfinite(confidence)
                and 0 <= confidence <= 1
            ):

                result.update(
                    candidate=label,
                    confidence=confidence,
                    completed=1,
                )

                return result

        if result["status"] == "OK":

            result["status"] = "ERROR"

        return result

    return result


# ==========================================
# COMBINE CIC AND UNSW RESULTS
# ==========================================

def aggregate(dual):

    branches = {

        ds: summarize_branch(
            dual.get(ds, {})
        )

        for ds in EXTRACTORS
    }

    results = [

        b for b in branches.values()

        if b["completed"]
    ]

    labels = sorted({

        b["candidate"]

        for b in results
    })

    completed = len(results)

    experimental = any(

        b["experimental"]

        for b in results
    )

    # Experimental predictions cannot
    # generate confirmed TabR alerts.

    attack = any(

        b["candidate"] != "Normal"

        and not b["experimental"]

        for b in results
    )

    if not completed:

        status = "UNAVAILABLE"

    elif experimental:

        status = DEMO

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

        "experimental": experimental,

        "branches": branches,

        "completed_models": completed,

        "required_models": 2,
    }


# ==========================================
# TABR REALTIME DETECTION
# ==========================================

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

        self.experimental = (

            os.environ.get(
                "ATTT_EXPERIMENTAL_TABR"
            ) == "1"

        )

        self.engine = DualInferenceEngine(

            Path(root),

            threads=threads,

            models=("TabR",),

            strict_versions=True,

            max_cached_models=max_cached_models,

        )

    def detect(self, event):

        rows = {}
        audits = {}
        qualities = {}

        # ==================================
        # FEATURE EXTRACTION
        # ==================================

        for dataset, extractor in EXTRACTORS.items():

            reference = None
            error = None

            if self.provider is not None:

                try:

                    reference = self.provider(
                        event,
                        dataset,
                    )

                except Exception as exc:

                    error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            audit = extractor(
                event,
                reference,
            )

            if error:

                audit.update(

                    status="PROVIDER_ERROR",

                    error=error,

                    model_features=None,

                )

            row = audit["model_features"]

            quality = (

                "REFERENCE_SCHEMA_OK"

                if row is not None

                else "UNAVAILABLE"

            )

            # ==============================
            # EXPERIMENTAL INFERENCE
            # ==============================

            if (

                row is None

                and self.experimental

                and audit["status"] == "UNVERIFIED"

            ):

                try:

                    row = candidate_row(
                        dataset,
                        audit,
                    )

                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    OverflowError,
                ) as exc:

                    audit.update(

                        status="INVALID_FLOW",

                        error=str(exc),

                    )

                if row is not None:

                    quality = DEMO

                    audit["status"] = DEMO

            rows[dataset] = row

            qualities[dataset] = quality

            audits[dataset] = {

                k: v

                for k, v in audit.items()

                if k not in {

                    "candidate_features",

                    "model_features",

                }
            }

        # ==================================
        # RUN BOTH EXISTING TABR CHECKPOINTS
        # ==================================

        dual = self.engine.predict_flow(

            event["flow_id"],

            cic_features=rows["cic"],

            unsw_features=rows["unsw"],

            route="both",

        )

        # ==================================
        # LABEL EXPERIMENTAL RESULTS
        # ==================================

        for dataset, quality in qualities.items():

            branch = dual[dataset]

            branch["feature_quality"] = quality

            audit = audits[dataset]

            if (

                rows[dataset] is None

                and audit["status"] == "UNVERIFIED"

                and not audit["missing_features"]

            ):

                branch["status"] = "UNVERIFIED"

            if quality == DEMO:

                for model in branch.get("models", []):

                    for prediction in model.get(
                        "predictions", []
                    ):

                        prediction["candidate_label"] = (
                            prediction["label"]
                        )

                        prediction["label"] = (
                            "[DEMO] " + prediction["label"]
                        )

                        prediction["feature_quality"] = DEMO

        dual.update(

            mode=TABR_MODE,

            feature_audit=audits,

            experimental_enabled=self.experimental,

        )

        dual["decision"] = aggregate(dual)

        return dual
