
"""Optional behavior router, trained on independently labeled live-flow records."""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.cic_feature_extractor import flow_parts, number, stat_value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTER = ROOT / "models" / "stage4" / "attack_router.joblib"
ROUTER_VERSION = 1
ROUTES = ("cic", "unsw", "both")
FEATURES = (
    "duration_s", "forward_packets", "backward_packets",
    "forward_ip_bytes", "backward_ip_bytes",
    "forward_mean_iat_s", "backward_mean_iat_s", "flow_mean_iat_s",
    "forward_syn", "backward_syn", "forward_rst", "backward_rst",
    "tcp", "udp",
)


def target_route(label):
    """Routing target is the desired label vocabulary, NOT the dataset of origin."""
    value = str(label).strip().casefold().replace("_", "").replace(" ", "")
    if value in {"ddos", "portscan"}:
        return "cic"
    if value in {"analysis", "backdoor", "dos", "exploits", "fuzzers",
                 "generic", "reconnaissance", "shellcode", "worms"}:
        return "unsw"
    if value in {"normal", "benign", "ambiguous"}:
        return "both"
    raise ValueError(f"Unsupported ground-truth label: {label!r}")


def common_features(event):
    """Exactly the same function is used during training and deployment."""
    stats, f, b, duration, _ = flow_parts(event)
    values = (
        duration, f["packets"], b["packets"],
        f["ip_length"]["total"], b["ip_length"]["total"],
        stat_value(f["iat_seconds"], "mean"),
        stat_value(b["iat_seconds"], "mean"),
        stat_value(stats["flow_iat_seconds"], "mean"),
        f["tcp_flags"].get("SYN", 0), b["tcp_flags"].get("SYN", 0),
        f["tcp_flags"].get("RST", 0), b["tcp_flags"].get("RST", 0),
        int(event["protocol"] == "tcp"), int(event["protocol"] == "udp"),
    )
    row = {name: number(value) for name, value in zip(FEATURES, values)}
    if any(value < 0 for value in row.values()):
        raise ValueError("Negative common feature")
    return row


def choose_route(probabilities, classes, threshold):
    """Conservative fallback. Probability is a routing score, not certainty."""
    p = np.asarray(probabilities, dtype=float)
    names = tuple(map(str, classes))
    if (p.shape != (len(ROUTES),) or len(names) != len(ROUTES)
            or set(names) != set(ROUTES) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1)
            or not 0.5 < threshold <= 1):
        raise ValueError("Invalid router probabilities or threshold")
    index = int(p.argmax())
    route = names[index]
    confidence = float(p[index])
    reason = "router_selected"
    if route == "both":
        reason = "router_requested_both"
    elif confidence < threshold:
        route, reason = "both", "low_routing_score"
    return {"route": route, "reason": reason, "confidence": confidence,
            "scores": dict(zip(names, map(float, p)))}


class AttackRouter:
    def __init__(self, path=DEFAULT_ROUTER):
        self.bundle = None
        self.problem = "router_not_trained"
        path = Path(path)
        if not path.is_file():
            return
        try:
            # Only load a trusted artifact trained locally by src.train_router.
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                bundle = joblib.load(path)
            if (bundle["version"] != ROUTER_VERSION
                    or tuple(bundle["features"]) != FEATURES
                    or set(bundle["model"].classes_) != set(ROUTES)
                    or not 0.5 < float(bundle["threshold"]) <= 1):
                raise ValueError("Router contract mismatch")
            self.bundle = bundle
            self.problem = ""
        except Exception as error:
            self.problem = f"router_unavailable: {type(error).__name__}: {error}"

    def predict(self, event):
        if self.bundle is None:
            return {"route": "both", "reason": self.problem, "confidence": None, "scores": {}}
        try:
            X = pd.DataFrame([common_features(event)], columns=FEATURES)
            model = self.bundle["model"]
            return choose_route(model.predict_proba(X)[0], model.classes_,
                                float(self.bundle["threshold"]))
        except Exception as error:
            return {"route": "both", "reason": f"router_error: {error}",
                    "confidence": None, "scores": {}}
