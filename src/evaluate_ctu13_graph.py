"""Evaluate the existing offline graph and periodicity rules on CTU-13 labeled NetFlows."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.botnet_graph import TOPOLOGY_METHODS, analyze_flows, windowed_flows
from src.ctu13_graph_models import evaluate_graph_models


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ctu13_stage3"
OUTPUT = ROOT / "results" / "stage3" / "ctu13"
WINDOW_SECONDS = 900
EDGE_KEY = ["capture", "window", "src", "dst"]
SOURCE = {
    "CTU-11": {
        "filename": "capture20110818-2.binetflow",
        "scenario_description": "Rbot (IRC)",
        "reference_topology": "centralized",
        "rows": 107251,
        "sha256": "d761b4513ee42be57193fe8f442e9234e73ac81f4fbd01248e37c23ca8ec1cf2",
        "mirror": "https://www.kaggle.com/datasets/eshaadutta/ctu-13-11",
    },
    "CTU-12": {
        "filename": "capture20110819.binetflow",
        "scenario_description": "NSIS.ay (P2P)",
        "reference_topology": "p2p",
        "rows": 325471,
        "sha256": "615e28233cb3c494b74c1016af940394c2765df9a285413256e65b0ebd11bbca",
        "mirror": "https://www.kaggle.com/datasets/berty2/ctu-13-dataset",
    },
}


def prepare_capture(source_dir, capture, settings):
    path = source_dir / settings["filename"]
    if sha256(path.read_bytes()).hexdigest() != settings["sha256"]:
        raise ValueError(f"{path.name} does not match the documented source file.")
    raw = pd.read_csv(path, usecols=["StartTime", "SrcAddr", "DstAddr", "Label"], dtype=str)
    if len(raw) != settings["rows"] or raw.isna().any().any():
        raise ValueError(f"Unexpected row count or missing metadata in {path.name}.")
    labels = raw["Label"].str.extract(r"(?i)\b(Botnet|Normal|Background)\b", expand=False).str.lower()
    if labels.isna().any():
        raise ValueError(f"Unrecognized CTU label in {path.name}.")
    times = pd.to_datetime(raw["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
    # Keep all graph relations and time differences without sharing source addresses.
    hosts = pd.Index(pd.unique(pd.concat([raw["SrcAddr"], raw["DstAddr"]])))
    aliases = pd.Series([f"h{i:05d}" for i in range(len(hosts))], index=hosts)
    return pd.DataFrame({
        "file": capture,
        "start": (times - times.min()).dt.total_seconds(),
        "src": raw["SrcAddr"].map(aliases),
        "dst": raw["DstAddr"].map(aliases),
        "label": labels,
    })


def prepare(source_dir, data_dir):
    flows = pd.concat(
        [prepare_capture(source_dir, capture, settings) for capture, settings in SOURCE.items()],
        ignore_index=True,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(data_dir / "flows.parquet", index=False)
    manifest = {
        "source": "https://www.stratosphereips.org/datasets-ctu13/",
        "paper": "https://doi.org/10.1016/j.cose.2014.05.011",
        "description": "All data rows; projected to capture, start in relative seconds, pseudonymous endpoints and label category.",
        "parquet_sha256": sha256((data_dir / "flows.parquet").read_bytes()).hexdigest(),
        "scenarios": SOURCE,
    }
    (data_dir / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def attach_truth(flows, edges):
    """Assign truth after all alerts are computed. Background-only is unscored."""
    windowed = windowed_flows(flows, WINDOW_SECONDS)
    windowed["label"] = flows.loc[windowed.index, "label"]
    windowed["botnet"] = windowed["label"].eq("botnet")
    windowed["normal"] = windowed["label"].eq("normal")
    grouped = windowed.groupby(EDGE_KEY, sort=False)[["botnet", "normal"]].any().reset_index()
    grouped["truth"] = np.select([grouped["botnet"], grouped["normal"]],
                                 ["botnet", "normal"], default="unknown")
    return edges.merge(grouped[EDGE_KEY + ["truth"]], on=EDGE_KEY, validate="one_to_one")


def summarize(edges):
    rows = []
    for capture, data in edges.groupby("capture", sort=True):
        known = data.loc[data["truth"].ne("unknown")]
        actual = known["truth"].eq("botnet")
        for method in (*TOPOLOGY_METHODS, "periodicity_only"):
            if method in TOPOLOGY_METHODS:
                alerts = data[method].ne("unclassified")
                match = int((known[method].eq(SOURCE[capture]["reference_topology"]) & actual).sum())
            else:
                alerts = data["periodic"]
                match = None
            predicted = alerts.loc[known.index]
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average="binary", zero_division=0,
            )
            rows.append({
                "capture": capture, "method": method,
                "scored_edges": len(known), "unknown_edges": len(data) - len(known),
                "unknown_alerts": int(alerts.loc[data["truth"].eq("unknown")].sum()),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "precision": precision, "recall": recall, "f1": f1,
                "reference_topology_matches": match,
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            })
    return pd.DataFrame(rows)


def evaluate(data_dir, output):
    flows = pd.read_parquet(data_dir / "flows.parquet")
    labels = flows.groupby(["file", "label"]).size().unstack(fill_value=0)
    expected = pd.Series({capture: meta["rows"] for capture, meta in SOURCE.items()})
    if not labels.sum(axis=1).equals(expected):
        raise ValueError("The bundle must contain the complete two CTU-13 captures.")
    edges, components = analyze_flows(flows, window_seconds=WINDOW_SECONDS)
    edges = attach_truth(flows, edges)
    metrics = summarize(edges)
    graph_models, folds = evaluate_graph_models(edges)
    candidates = components.loc[components["topology"].ne("unclassified")]
    alerts = edges.loc[edges["periodic"] | edges[list(TOPOLOGY_METHODS)].ne("unclassified").any(axis=1)]
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "flow_labels.csv")
    metrics.to_csv(output / "edge_metrics.csv", index=False, float_format="%.8f")
    candidates.to_csv(output / "candidate_components.csv", index=False, float_format="%.8f")
    alerts.to_csv(output / "alert_edges.csv", index=False, float_format="%.8f")
    graph_models.to_csv(output / "graph_model_metrics.csv", index=False, float_format="%.8f")
    folds.to_csv(output / "graph_model_folds.csv", index=False)
    print(metrics.to_string(index=False))
    print(graph_models.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Rebuild the prepared Parquet from two CTU-13 .binetflow files.")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.source_dir is not None:
        prepare(args.source_dir, args.data_dir)
    evaluate(args.data_dir, args.output)


if __name__ == "__main__":
    main()
