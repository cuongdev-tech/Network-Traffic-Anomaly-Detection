
import argparse
import json

from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.cic_data import load_cleaned_cic

from src.unsw_data import (
    load_and_balance_unsw,
    split_for_tabr,
    test_index_hash,
)

from src.ood_detector import (
    SCHEMA,
    build_profile,
    load_profile,
    outside,
)

from src.cic_feature_extractor import (
    extract_cic_features,
)

from src.unsw_feature_extractor import (
    extract_unsw_features,
)

ROOT = Path(__file__).resolve().parents[1]

EXTRACT = {
    "cic": extract_cic_features,
    "unsw": extract_unsw_features,
}


# ==========================================
# REBUILD ORIGINAL TRAINING SPLIT
# ==========================================

def make_profile(dataset):

    if dataset == "cic":

        x = load_cleaned_cic(
            verbose=False
        )

        y = x.pop("Label")

        x = x.select_dtypes(
            include=[np.number]
        )

        meta = ROOT / "results/tabr_meta.json"

    else:

        x, y = load_and_balance_unsw(
            verbose=False
        )

        meta = (
            ROOT
            / "results/unsw_nb15/tabr_meta.json"
        )

    m = json.loads(

        meta.read_text(
            encoding="utf-8"
        )

    )

    encoder = LabelEncoder()

    encoded = encoder.fit_transform(y)

    train, val, test, yt, _, _ = (
        split_for_tabr(x, encoded)
    )

    # Verify historical split and schema.

    if (

        set(x.columns) != set(SCHEMA[dataset])

        or len(x) != m["rows_used"]

        or encoder.classes_.tolist() != m["classes"]

        or test_index_hash(test)
        != m["test_indices_sha256"]

    ):

        raise ValueError(
            "Historical dataset or split "
            "does not match TabR metadata"
        )

    if len(train) != m["fit_train"]:

        train, _, yt, _ = train_test_split(

            train,

            yt,

            train_size=m["fit_train"],

            stratify=yt,

            random_state=42,
        )

    normal_label = (

        "BENIGN"

        if dataset == "cic"

        else "Normal"

    )

    normal = train.loc[

        yt == encoder.transform(
            [normal_label]
        )[0]

    ]

    p = build_profile(

        dataset,

        train,

        val,

        normal,
    )

    p["test_indices_sha256"] = (
        m["test_indices_sha256"]
    )

    path = (

        ROOT

        / "models/stage4"

        / f"ood_{dataset}.json"

    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(

        json.dumps(

            p,

            indent=2,

            allow_nan=False,
        ),

        encoding="utf-8",
    )

    print("PROFILE:", path)

    print(
        "Training rows:",
        p["train_rows"],
    )

    print(
        "Validation rows:",
        p["validation_rows"],
    )

    print(
        "Validation rejection:",
        p["validation_rejection"],
    )

    print(
        "Outlier limit:",
        p["max_outliers"],
    )


# ==========================================
# AUDIT LIVE DISTRIBUTION
# ==========================================

def audit(dataset, events):

    p = load_profile(
        str(ROOT), dataset
    )

    if p is None:

        raise ValueError(
            "Build the OOD profile first"
        )

    seen = Counter()

    missing = Counter()

    tails = Counter()

    normal_tails = Counter()

    totals = Counter()

    examples = {

        name: []

        for name in SCHEMA[dataset]

    }

    with Path(events).open(
        encoding="utf-8"
    ) as stream:

        for line in stream:

            if not line.strip():
                continue

            e = json.loads(line)

            if e.get("event") not in {

                "flow_decision",

                "flow_alert",

            }:

                continue

            a = EXTRACT[dataset](e)

            totals["flows"] += 1

            totals["invalid"] += int(

                a["status"] == "INVALID_FLOW"

            )

            missing.update(
                a["missing_features"]
            )

            for name, value in a[
                "candidate_features"
            ].items():

                seen[name] += 1

                if len(examples[name]) < 5:

                    examples[name].append(
                        value
                    )

                tails[name] += int(

                    outside(
                        value,
                        p["reference"][name],
                    )

                )

                normal_tails[name] += int(

                    outside(
                        value,
                        p["normal_reference"][name],
                    )

                )

    report = [

        dict(

            feature=name,

            seen=seen[name],

            missing=missing[name],

            outside_train=tails[name],

            outside_normal=normal_tails[name],

            live_examples=examples[name],

            train_reference=p["reference"][name],

            normal_reference=p["normal_reference"][name],

        )

        for name in SCHEMA[dataset]

    ]

    report.sort(

        key=lambda r: (

            r["outside_normal"],

            r["missing"],

        ),

        reverse=True,

    )

    print(json.dumps(

        dict(

            dataset=dataset,

            totals=dict(totals),

            features=report,

        ),

        indent=2,

    ))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        choices=SCHEMA,
        required=True,
    )

    ap.add_argument(
        "--build",
        action="store_true",
    )

    ap.add_argument(
        "--events",
        type=Path,
    )

    args = ap.parse_args()

    if not args.build and not args.events:

        ap.error(
            "Choose --build or --events"
        )

    if args.build:

        make_profile(
            args.dataset
        )

    if args.events:

        audit(

            args.dataset,

            args.events,
        )
