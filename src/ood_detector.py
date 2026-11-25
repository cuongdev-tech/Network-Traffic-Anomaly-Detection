
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.cic_feature_extractor import CIC_FEATURES
from src.unsw_feature_extractor import UNSW_FEATURES

SCHEMA = {
    "cic": CIC_FEATURES,
    "unsw": UNSW_FEATURES,
}

CATS = {"proto", "service", "state"}


# ==========================================
# TRAINING FEATURE STATISTICS
# ==========================================

def stats(frame, schema):

    result = {}

    for name in schema:

        s = frame[name]

        if name in CATS:

            result[name] = {
                "categories": sorted(
                    s.dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
            }

        else:

            x = pd.to_numeric(
                s, errors="coerce"
            ).to_numpy(dtype=float)

            x = x[np.isfinite(x)]

            if not len(x):

                raise ValueError(
                    f"No finite reference values: {name}"
                )

            lo, mid, hi = np.quantile(
                x, [0.005, 0.5, 0.995]
            )

            result[name] = dict(
                lo=float(lo),
                median=float(mid),
                hi=float(hi),
            )

    return result


def outside(value, spec):

    if "categories" in spec:

        return (
            not isinstance(value, str)
            or value.strip() not in spec["categories"]
        )

    try:

        x = float(value)

    except (ValueError, TypeError, OverflowError):

        return True

    return (
        not np.isfinite(x)
        or x < spec["lo"]
        or x > spec["hi"]
    )


# ==========================================
# BUILD OOD PROFILE
# ==========================================

def build_profile(dataset, train, validation, normal):

    schema = SCHEMA[dataset]

    reference = stats(train, schema)

    counts = np.zeros(
        len(validation), dtype=int
    )

    unknown = np.zeros(
        len(validation), dtype=bool
    )

    for name, spec in reference.items():

        s = validation[name]

        if name in CATS:

            bad = ~s.astype(str).str.strip().isin(
                spec["categories"]
            ).to_numpy()

            unknown |= bad

        else:

            x = pd.to_numeric(
                s, errors="coerce"
            ).to_numpy(dtype=float)

            bad = (
                ~np.isfinite(x)
                | (x < spec["lo"])
                | (x > spec["hi"])
            )

        counts += bad

    limit = int(
        np.quantile(
            counts,
            0.99,
            method="higher",
        )
    )

    return dict(
        version=1,
        dataset=dataset,
        features=list(schema),
        reference=reference,
        normal_reference=stats(normal, schema),
        train_rows=len(train),
        validation_rows=len(validation),
        max_outliers=limit,
        validation_rejection=float(
            np.mean((counts > limit) | unknown)
        ),
    )


@lru_cache(maxsize=2)
def load_profile(root, dataset):

    path = (
        Path(root)
        / "models"
        / "stage4"
        / f"ood_{dataset}.json"
    )

    if not path.exists():

        return None

    p = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (

        p["version"] != 1

        or p["dataset"] != dataset

        or p["features"] != list(SCHEMA[dataset])

    ):

        raise ValueError(
            "OOD profile schema mismatch"
        )

    return p


# ==========================================
# CHECK LIVE FLOW
# ==========================================

def assess(root, dataset, row):

    if row is None:

        return {
            "status": "UNAVAILABLE"
        }

    p = load_profile(
        str(root), dataset
    )

    if p is None:

        return {
            "status": "NOT_CONFIGURED"
        }

    bad = [

        name

        for name in SCHEMA[dataset]

        if outside(
            row.get(name),
            p["reference"][name],
        )

    ]

    unknown = sorted(
        set(bad) & CATS
    )

    rejected = (
        bool(unknown)
        or len(bad) > p["max_outliers"]
    )

    return dict(

        status=(
            "OOD" if rejected
            else "IN_RANGE"
        ),

        count=len(bad),

        limit=p["max_outliers"],

        features=bad,

        unknown_categories=unknown,
    )


# ==========================================
# ATTACH QUALITY INFORMATION
# ==========================================

def attach_quality(
    dual,
    rows,
    root,
    gaussian_flag,
):

    if os.getenv("ATTT_QUALITY_GATE") != "1":

        return

    dual["quality_gate"] = {

        "enabled": True,

        "gaussian_flag": gaussian_flag,
    }

    for ds in SCHEMA:

        try:

            result = assess(
                root, ds, rows[ds]
            )

            threshold = float(

                os.getenv(
                    f"ATTT_MIN_CONF_{ds.upper()}",
                    "0",
                )

            )

            if not 0 <= threshold <= 1:

                raise ValueError(
                    "Confidence threshold must be 0..1"
                )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
        ) as exc:

            result = {

                "status": "PROFILE_ERROR",

                "error": str(exc),
            }

            threshold = 0.0

        dual[ds]["ood"] = result

        dual[ds]["min_confidence"] = threshold


# ==========================================
# QUALITY DECISION
# ==========================================

def apply_quality(dual, decision):

    if not dual.get(
        "quality_gate", {}
    ).get("enabled"):

        return decision

    statuses = []

    accepted = []

    for ds, result in decision["branches"].items():

        branch = dual.get(ds, {})

        status = branch.get(
            "ood", {}
        ).get(
            "status",
            "NOT_CONFIGURED",
        )

        result["ood"] = branch.get(
            "ood", {}
        )

        if not result["completed"]:

            statuses.append(
                "UNCERTAIN_INCOMPLETE"
            )

        elif status == "OOD":

            statuses.append(
                "UNCERTAIN_OOD"
            )

        elif status != "IN_RANGE":

            statuses.append(
                "UNCERTAIN_NO_PROFILE"
            )

        elif result["confidence"] < branch.get(
            "min_confidence", 0
        ):

            statuses.append(
                "UNCERTAIN_LOW_CONFIDENCE"
            )

        else:

            accepted.append(
                result["candidate"]
            )

    status = next(

        (

            s for s in (

                "UNCERTAIN_OOD",

                "UNCERTAIN_NO_PROFILE",

                "UNCERTAIN_INCOMPLETE",

                "UNCERTAIN_LOW_CONFIDENCE",

            )

            if s in statuses

        ),

        None,
    )

    if status is None:

        if len(set(accepted)) > 1:

            status = "UNCERTAIN_CONFLICT"

        elif (

            accepted == ["Normal", "Normal"]

            and dual["quality_gate"].get(
                "gaussian_flag"
            ) is True

        ):

            status = "UNCERTAIN_GAUSSIAN"

        else:

            status = "EXPERIMENTAL_UNVERIFIED"

    return {

        **decision,

        "raw_status": decision["status"],

        "status": status,

        "accepted_labels": sorted(
            set(accepted)
        ),

        # Experimental TabR labels must not
        # automatically become attack alerts.

        "attack_candidate": False,
    }
