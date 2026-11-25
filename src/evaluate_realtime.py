
import argparse
import csv
import json

from pathlib import Path

import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


CLASSES = {

    "cic": {

        "Normal",
        "PortScan",
        "DDoS",

    },

    "unsw": {

        "Normal",
        "Analysis",
        "Backdoor",
        "DoS",
        "Exploits",
        "Fuzzers",
        "Generic",
        "Reconnaissance",
        "Shellcode",
        "Worms",

    },
}


def label(value):

    return (
        "Normal"
        if value == "BENIGN"
        else value
    )


# ==========================================
# EVALUATION METRICS
# ==========================================

def metrics(pairs):

    n = len(pairs)

    covered = [

        (t, p)

        for t, p in pairs

        if p is not None

    ]

    correct = sum(

        t == p

        for t, p in covered

    )

    normals = sum(

        t == "Normal"

        for t, p in pairs

    )

    normal_covered = sum(

        t == "Normal"

        for t, p in covered

    )

    fp = sum(

        t == "Normal"
        and p != "Normal"

        for t, p in covered

    )

    attacks = sum(

        t != "Normal"

        for t, p in pairs

    )

    detected = sum(

        t != "Normal"
        and p != "Normal"

        for t, p in covered

    )

    def ratio(a, b):

        return a / b if b else None

    return dict(

        labeled=n,

        predicted=len(covered),

        coverage=ratio(
            len(covered), n
        ),

        accuracy_covered=ratio(
            correct, len(covered)
        ),

        accuracy_all=ratio(
            correct, n
        ),

        macro_f1_covered=float(

            f1_score(

                *zip(*covered),

                average="macro",

                zero_division=0,

            )

        ) if covered else None,

        precision_macro_covered=float(

            precision_score(

                *zip(*covered),

                average="macro",

                zero_division=0,

            )

        ) if covered else None,

        recall_macro_covered=float(

            recall_score(

                *zip(*covered),

                average="macro",

                zero_division=0,

            )

        ) if covered else None,

        normal_count=normals,

        attack_count=attacks,

        false_positives=fp,

        false_positive_rate_all_normal=ratio(
            fp, normals
        ),

        false_positive_rate_covered_normal=ratio(
            fp, normal_covered
        ),

        attack_detection_rate_all=ratio(
            detected, attacks
        ),
    )


# ==========================================
# EVALUATE REALTIME RESULTS
# ==========================================

def evaluate(

    events,

    labels=None,

    normal_http=None,

    confidence=0.0,
):

    truth = {}

    if labels:

        with Path(labels).open(

            encoding="utf-8-sig",

            newline="",

        ) as f:

            for r in csv.DictReader(f):

                if r["flow_id"] in truth:

                    raise ValueError(
                        "Duplicate ground truth flow_id"
                    )

                truth[r["flow_id"]] = r

    raw = {

        d: []

        for d in ("cic", "unsw")

    }

    accepted = {

        d: []

        for d in raw

    }

    alerts = benign = attacked = detected = 0

    delays = []

    found = set()

    with Path(events).open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            e = json.loads(line)

            if e.get("event") not in {

                "flow_decision",

                "flow_alert",

            }:

                continue

            fid = e["flow_id"]

            if normal_http:

                src, dst, port = normal_http

                if (

                    e.get("src"),

                    e.get("dst"),

                    str(e.get("dport")),

                    e.get("protocol"),

                ) != (

                    src,

                    dst,

                    port,

                    "tcp",

                ):

                    continue

                t = {

                    "cic": "Normal",

                    "unsw": "Normal",

                }

            else:

                t = truth.get(fid)

                if t is None:
                    continue

            if fid in found:

                raise ValueError(
                    "Duplicate flow_id in events"
                )

            found.add(fid)

            known = {

                d: label(t[d].strip())

                for d in raw

                if t.get(d, "").strip()

            }

            if any(

                v not in CLASSES[d]

                for d, v in known.items()

            ):

                raise ValueError(
                    f"Invalid ground truth class for {fid}"
                )

            delay = e.get(
                "post_last_packet_ms"
            )

            if (

                isinstance(delay, (int, float))

                and np.isfinite(delay)

                and delay >= 0

            ):

                delays.append(delay)

            if (

                len(known) == 2

                and all(
                    v == "Normal"
                    for v in known.values()
                )

            ):

                benign += 1

                alerts += int(
                    e["event"] == "flow_alert"
                )

            elif any(

                v != "Normal"

                for v in known.values()

            ):

                attacked += 1

                detected += int(
                    e["event"] == "flow_alert"
                )

            dual = e.get(
                "dual_inference"
            ) or {}

            for d in raw:

                target = t.get(
                    d, ""
                ).strip()

                if not target:

                    continue

                target = label(target)

                branch = dual.get(d, {})

                m = next(

                    (

                        m for m in branch.get(
                            "models", []
                        )

                        if m.get("model") == "TabR"

                        and m.get("status") == "OK"

                    ),

                    {},
                )

                preds = m.get(
                    "predictions", []
                )

                pred = (

                    preds[0]

                    if len(preds) == 1

                    else {}

                )

                candidate = pred.get(

                    "candidate_label",

                    pred.get("label"),

                )

                candidate = (

                    label(
                        candidate.removeprefix("[DEMO] ")
                    )

                    if candidate

                    else None

                )

                raw[d].append(
                    (target, candidate)
                )

                accept = (

                    candidate is not None

                    and branch.get(
                        "ood", {}
                    ).get("status") == "IN_RANGE"

                    and pred.get("confidence", 0)
                    >= max(

                        confidence,

                        branch.get("min_confidence", 0),

                    )

                )

                accepted[d].append(

                    (

                        target,

                        candidate if accept else None,

                    )

                )

    return dict(

        matched_flows=len(found),

        unmatched_label_ids=len(
            set(truth) - found
        ),

        actual_flow_alert_fpr=(
            alerts / benign if benign else None
        ),

        actual_flow_alert_recall=(
            detected / attacked if attacked else None
        ),

        latency_ms=dict(

            samples=len(delays),

            p50=float(np.median(delays))
            if delays else None,

            p95=float(np.percentile(delays, 95))
            if delays else None,

        ),

        branches={

            d: {

                "raw": metrics(raw[d]),

                "ood_confidence_filtered": metrics(
                    accepted[d]
                ),

            }

            for d in raw

        },

        note=(
            "Filtered metrics are selective, "
            "not improved raw model accuracy. "
            "No attack recall without attack labels."
        ),
    )


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--events",
        required=True,
        type=Path,
    )

    group = ap.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--labels",
        type=Path,
    )

    group.add_argument(
        "--normal-http",
        nargs=3,
        metavar=("SRC", "DST", "PORT"),
    )

    ap.add_argument(
        "--confidence",
        type=float,
        default=0.0,
    )

    args = ap.parse_args()

    if not 0 <= args.confidence <= 1:

        ap.error(
            "Confidence must be between 0 and 1"
        )

    print(json.dumps(

        evaluate(

            args.events,

            args.labels,

            args.normal_http,

            args.confidence,

        ),

        indent=2,

        allow_nan=False,

    ))
