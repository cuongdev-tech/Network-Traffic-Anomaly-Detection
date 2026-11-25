import hashlib
import json
import os
from collections import Counter
from math import isfinite
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.behavior_detector import BehaviorWindow

from src.cic_feature_extractor import (
    CIC_FEATURES,
    extract_cic_features,
)

from src.dual_inference import (
    DualInferenceEngine,
    ROOT,
)

from src.netflow_realtime import (
    NetFlowTabR,
    extract_netflow_features,
)


# ==========================================
# PORTSCAN + DOS + DDOS
# ==========================================

class RealtimeBehaviorWindow(BehaviorWindow):

    def _reset(self):

        super()._reset()

        self.pair_packets = Counter()

    def observe(self, obs):

        events = super().observe(obs)

        if (
            isfinite(obs.timestamp)
            and self.start is not None
            and obs.timestamp >= self.start
            and not self.truncated
        ):

            self.pair_packets[
                (obs.src, obs.dst)
            ] += 1

        return events

    def advance(self, timestamp):

        # Preserve the existing PortScan
        # and DDoS detection.

        if (
            self.start is None
            or not isfinite(timestamp)
            or timestamp < self.start + self.seconds
        ):

            return super().advance(timestamp)

        start = self.start
        pairs = self.pair_packets
        totals = self.by_dst

        events = super().advance(timestamp)

        if not events or events[0].get("truncated"):
            return events

        ddos_destinations = {
            e["dst"]
            for e in events
            if e.get("label") == "DDoS"
        }

        dominant = {}

        for (src, dst), count in pairs.items():

            previous = dominant.get(dst)

            if previous is None or count > previous[1]:

                dominant[dst] = (src, count)

        dos_alerts = []

        for dst, (src, count) in dominant.items():

            if dst in ddos_destinations:
                continue

            total = totals[dst]

            if not total:
                continue

            rate = count / self.seconds

            share = count / total

            # Experimental traffic-concentration
            # heuristic, not proof of service outage.

            if rate >= self.flood_pps and share >= 0.8:

                dos_alerts.append({

                    "event": "behavior_alert",

                    "label": "DoS",

                    "source": "Host behavior (heuristic)",

                    "src": src,
                    "dst": dst,

                    "window_start_epoch": start,

                    "window_end_epoch":
                        start + self.seconds,

                    "window_seconds": self.seconds,

                    "evidence": (
                        f"Suspected DoS: {rate:.1f} packets/s; "
                        f"dominant source share {share:.1%}"
                    ),
                })

        events[0]["alerts"] += len(dos_alerts)

        return events + dos_alerts


# ==========================================
# ALERT POLICY
# ==========================================

def load_policy(root, classes):

    folder = Path(root) / "models/unsw_realtime"

    checkpoint = folder / "tabr_checkpoint.joblib"

    path = folder / "alert_policy.json"

    try:

        policy = json.loads(
            path.read_text(encoding="utf-8")
        )

        digest = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()

        if policy["checkpoint_sha256"] != digest:
            raise ValueError("Policy checkpoint mismatch")

        if policy["schema_version"] != "netflow_v1_7":
            raise ValueError("Policy schema mismatch")

        thresholds = policy["thresholds"]

        expected = set(classes) - {"Normal"}

        if set(thresholds) != expected:
            raise ValueError("Policy class mismatch")

        for value in thresholds.values():

            if value is None:
                continue

            if (
                not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):

                raise ValueError("Invalid threshold")

        fpr = float(
            policy["test_metrics"]["false_positive_rate"]
        )

        target = float(policy["target_fpr"])

        if (
            not isfinite(fpr)
            or not isfinite(target)
            or not 0 <= fpr <= target <= 1
        ):

            raise ValueError(
                "Policy failed test FPR requirement"
            )

        return thresholds, "READY"

    except Exception as exc:

        # Fail closed: keep logging predictions,
        # but do not emit NetFlow model alerts.

        return {}, f"DISABLED: {exc}"


# ==========================================
# CIC + NETFLOW TABR
# ==========================================

class RealtimeDetectionEngine:

    def __init__(
        self,
        root=ROOT,
        *,
        provider=None,
        threads=2,
        max_cached_models=2,
    ):

        self.provider = provider

        self.cic = DualInferenceEngine(
            root=Path(root),
            threads=threads,
            models=("TabR",),
            strict_versions=True,
            max_cached_models=1,
        )

        self.netflow = NetFlowTabR(
            threads=threads
        )

        self.thresholds, self.policy_status = (
            load_policy(root, self.netflow.classes)
        )

        print(
            "NetFlow alert policy:",
            self.policy_status,
            flush=True,
        )

    def detect_cic(self, event):

        reference = None

        if self.provider is not None:

            reference = self.provider(event, "cic")

        audit = extract_cic_features(
            event, reference
        )

        row = audit.get("model_features")

        quality = audit.get("status", "UNKNOWN")

        if (
            row is None
            and quality == "UNVERIFIED"
            and os.getenv("ATTT_EXPERIMENTAL_TABR") == "1"
        ):

            candidate = audit.get(
                "candidate_features", {}
            )

            if all(k in candidate for k in CIC_FEATURES):

                row = {
                    k: candidate[k]
                    for k in CIC_FEATURES
                }

                quality = "EXPERIMENTAL_UNVERIFIED"

        if row is None:

            return {

                "status": "FEATURES_UNAVAILABLE",

                "models": [],

                "feature_quality": quality,
            }

        result = self.cic.infer_model(
            "cic",
            "TabR",
            pd.DataFrame([row]),
        )

        return {

            "status": result["status"],

            "models": [result],

            "feature_quality": quality,

            "alert_mode": "LOG_ONLY",
        }

    def detect_netflow(self, event):

        try:

            features = extract_netflow_features(
                event
            )

        except Exception as exc:

            return {

                "status": "FEATURES_UNAVAILABLE",

                "models": [],

                "error": str(exc),
            }

        started = perf_counter()

        try:

            result = self.netflow.predict(
                features
            )

            label = result["label"]

            confidence = float(
                result["confidence"]
            )

            if (
                label not in self.netflow.classes
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
            ):

                raise ValueError(
                    "Invalid prediction"
                )

            threshold = self.thresholds.get(label)

            accepted = (
                self.policy_status == "READY"
                and label != "Normal"
                and threshold is not None
                and confidence >= threshold
            )

            return {

                "status": "OK",

                "dataset": "NF-UNSW-NB15-V1",

                "feature_quality":
                    "EXPERIMENTAL_NETFLOW",

                "features": features,

                "policy_status": self.policy_status,

                "gate": {

                    "accepted": bool(accepted),

                    "threshold": threshold,
                },

                "models": [{

                    "model": "TabR",

                    "status": "OK",

                    "predictions": [{

                        "label": label,

                        "confidence": confidence,
                    }],

                    "inference_batch_ms": round(
                        (perf_counter() - started)
                        * 1000,
                        3,
                    ),
                }],
            }

        except Exception as exc:

            return {

                "status": "ERROR",

                "models": [],

                "error": str(exc),
            }

    def detect(self, event):

        try:

            cic = self.detect_cic(event)

        except Exception as exc:

            cic = {

                "status": "ERROR",

                "models": [],

                "error": str(exc),
            }

        unsw = self.detect_netflow(event)

        completed = 0

        labels = set()

        for branch in (cic, unsw):

            if branch["status"] == "OK":

                completed += 1

                for model in branch["models"]:

                    for p in model.get("predictions", []):

                        labels.add(p["label"])

        return {

            "flow_id": event["flow_id"],

            "mode": "tabr_cic_netflow",

            "cic": cic,

            "unsw": unsw,

            "decision": {

                "status": "EXPERIMENTAL",

                "candidate_labels": sorted(labels),

                "completed_models": completed,

                "required_models": 2,

                # Model advisories are emitted separately.
                # Prevent duplicate generic flow alerts.

                "attack_candidate": False,
            },
        }


# ==========================================
# GENERATE NETFLOW ADVISORY
# ==========================================

def netflow_candidate(event):

    dual = event.get("dual_inference") or {}

    branch = dual.get("unsw") or {}

    gate = branch.get("gate") or {}

    if (
        branch.get("status") != "OK"
        or gate.get("accepted") is not True
    ):

        return None

    prediction = branch["models"][0]["predictions"][0]

    label = prediction["label"]

    confidence = prediction["confidence"]

    threshold = gate["threshold"]

    if label == "Normal":
        return None

    return {

        "event": "model_candidate",

        "label": label,

        "source": "TabR NetFlow (nghi van)",

        "flow_id": event.get("flow_id"),

        "src": event.get("src"),

        "dst": event.get("dst"),

        "confidence": confidence,

        "threshold": threshold,

        "evidence": (
            f"Score {confidence:.1%}; "
            f"threshold {threshold:.1%}; "
            "experimental advisory"
        ),
    }
