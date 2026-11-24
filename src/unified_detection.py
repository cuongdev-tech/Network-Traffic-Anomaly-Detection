
"""Audit and compare both independent branches without guessing missing features."""
from collections import Counter
from importlib import import_module
from pathlib import Path

from src.cic_feature_extractor import extract_cic_features
from src.unsw_feature_extractor import extract_unsw_features
from src.dual_inference import DualInferenceEngine, MODELS, ROOT

EXTRACTORS = {"cic": extract_cic_features, "unsw": extract_unsw_features}


def load_provider(spec):
    """Optional trusted local adapter, enabled only after external parity testing.

    Signature: provider(event, dataset) -> None or dict with flow_id, dataset,
    extractor (versioned identity), and features (exact training-schema row).
    No provider exists in the original project. Do not point this at untrusted code.
    """
    if not spec:
        return None
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Expected feature provider as module:function")
    provider = getattr(import_module(module), name)
    if not callable(provider):
        raise ValueError("Feature provider must be callable")
    return provider


def summarize_branch(branch):
    models = branch.get("models", [])
    votes = Counter()
    for result in models:
        predictions = result.get("predictions", [])
        if result.get("status") == "OK" and len(predictions) == 1:
            votes[predictions[0]["label"]] += 1
    total = sum(votes.values())
    result = {"status": branch["status"], "votes": dict(votes), "completed": total,
              "required": len(MODELS), "candidate": None, "agreement": None}
    # Compare only a complete set of six results. A missing model is not a vote.
    if branch["status"] != "OK" or total != len(MODELS):
        return result
    label, count = votes.most_common(1)[0]
    result.update(agreement=count / total, status="DISAGREEMENT")
    if count > total / 2:
        result.update(status="CONSENSUS", candidate=label)
    return result


def aggregate(dual):
    branches = {name: summarize_branch(dual[name]) for name in EXTRACTORS}
    summaries = list(branches.values())
    candidates = [s["candidate"] for s in summaries if s["candidate"]]
    attacks = sorted({label for label in candidates if label != "Normal"})
    complete = all(s["status"] in {"CONSENSUS", "DISAGREEMENT"} for s in summaries)
    if not complete:
        status = "INCOMPLETE" if any(s["completed"] for s in summaries) else "UNAVAILABLE"
    elif any(s["status"] == "DISAGREEMENT" for s in summaries):
        status = "DISAGREEMENT"
    elif len(set(candidates)) > 1:
        status = "CONFLICT"
    elif attacks:
        status = "ATTACK_CANDIDATE"
    else:
        status = "NORMAL_CANDIDATE"
    # A majority attack vote is a candidate alert, not proof of ground truth.
    return {"status": status, "candidate_labels": sorted(set(candidates)),
            "attack_candidate": bool(attacks), "branches": branches,
            "completed_models": sum(s["completed"] for s in summaries),
            "required_models": 2 * len(MODELS)}


class UnifiedDetectionEngine:
    def __init__(self, root=ROOT, *, provider=None, threads=2):
        root = Path(root)
        self.provider = provider
        self.engine = DualInferenceEngine(root, threads=threads, strict_versions=True,
                                          max_cached_models=1)

    def detect(self, event):
        rows, audits = {}, {}
        for dataset, extractor in EXTRACTORS.items():
            reference = None
            error = None
            if self.provider is not None:
                try:
                    reference = self.provider(event, dataset)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            audit = extractor(event, reference)
            if error:
                audit.update(status="PROVIDER_ERROR", error=error, model_features=None)
            rows[dataset] = audit["model_features"]
            # Do not duplicate full feature vectors or raw packet contents in the log.
            audits[dataset] = {key: value for key, value in audit.items()
                               if key not in {"candidate_features", "model_features"}}
        dual = self.engine.predict_flow(event["flow_id"], cic_features=rows["cic"],
                                        unsw_features=rows["unsw"], route="both")
        return {**dual, "mode": "dual_independent", "feature_audit": audits, "decision": aggregate(dual)}
