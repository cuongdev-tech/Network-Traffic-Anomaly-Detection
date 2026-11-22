"""Evaluate offline graph detection on labeled synthetic connection records and local captures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.botnet_graph import analyze_flows


ROOT = Path(__file__).resolve().parents[1]


def synthetic_flows():
    """Write no traffic. Each independently labeled case is a table of timestamps."""
    star = [(f"client-{i}", "hub") for i in range(5)]
    ring = [(f"peer-{i}", f"peer-{(i + 1) % 5}") for i in range(5)]
    mesh = ring + [("peer-0", "peer-2"), ("peer-1", "peer-3")]
    chain = ring[:-1]
    early = [10, 50, 90, 130, 170, 210]
    jittered = [10, 52, 90, 129, 172, 208]
    irregular = [0, 4, 11, 45, 120, 129]
    cases = [
        ("centralized_clean", "centralized", star, early),
        ("centralized_jitter", "centralized", star, jittered),
        ("p2p_ring", "p2p", ring, early),
        ("p2p_mesh", "p2p", mesh, jittered),
        ("p2p_sparse", "p2p", ring, early),
        ("normal_irregular_star", "none", star, irregular),
        ("normal_periodic_service", "none", star, early),
        ("normal_periodic_pair", "none", [("client", "service")], early),
        ("normal_irregular_chain", "none", chain, irregular),
        ("normal_irregular_mesh", "none", mesh, irregular),
    ]
    rows = []
    for case, _, pairs, schedule in cases:
        for i, (src, dst) in enumerate(pairs):
            times = schedule[:3] if case == "p2p_sparse" and i == len(pairs) - 1 else schedule
            rows.extend((case, src, dst, tick + i * 2) for tick in times)
    return (pd.DataFrame(rows, columns=["file", "src", "dst", "start"]),
            {case: label for case, label, _, _ in cases})


def evaluate_cases(edges, components, truth):
    predictions = []
    for capture, expected in truth.items():
        observed = components.loc[components["capture"].eq(capture)]
        for method in ("topology_only", "periodic_topology"):
            candidates = observed.loc[observed["method"].eq(method), "topology"]
            labels = sorted(set(candidates) - {"unclassified"})
            outcome = "+".join(labels) if labels else "none"
            predictions.append((capture, expected, method, outcome))
        periodic = edges.loc[edges["capture"].eq(capture), "periodic"].any()
        predictions.append((capture, expected, "periodicity_only", "alert" if periodic else "none"))

    details = pd.DataFrame(predictions, columns=["capture", "expected", "method", "prediction"])
    summary = []
    for method, group in details.groupby("method", sort=False):
        actual = group["expected"].ne("none")
        flagged = group["prediction"].ne("none")
        summary.append({
            "method": method, "cases": len(group),
            "accuracy_binary": accuracy_score(actual, flagged),
            "precision": precision_score(actual, flagged, zero_division=0),
            "recall": recall_score(actual, flagged, zero_division=0),
            "f1": f1_score(actual, flagged, zero_division=0),
            "false_positives": int((~actual & flagged).sum()),
            "false_negatives": int((actual & ~flagged).sum()),
            "accuracy_topology": (None if method == "periodicity_only"
                                  else accuracy_score(group["expected"], group["prediction"])),
        })
    return details, pd.DataFrame(summary)


def plot_examples(edges, output):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for ax, (case, title) in zip(axes, [
        ("centralized_clean", "Centralized"),
        ("p2p_ring", "P2P"),
        ("normal_periodic_service", "Benign periodic service"),
    ]):
        rows = edges.loc[edges["capture"].eq(case) & edges["periodic"]]
        graph = nx.from_pandas_edgelist(rows, "src", "dst")
        positions = nx.spring_layout(graph, seed=42)
        nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#7788a0", width=1.5)
        nx.draw_networkx_nodes(graph, positions, ax=ax, node_color="#21608e", node_size=95)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Graphs of periodic host pairs in synthetic cases")
    fig.savefig(output, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "flows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage3")
    args = parser.parse_args()

    flows, truth = synthetic_flows()
    edges, components = analyze_flows(flows)
    cases, summary = evaluate_cases(edges, components, truth)
    local_edges, local_components = analyze_flows(pd.read_csv(args.input))
    results = {
        "synthetic_flows": flows, "synthetic_case_results": cases, "graph_method_metrics": summary,
        "capture_edges": local_edges, "capture_components": local_components,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(args.output / f"{name}.csv", index=False, float_format="%.8f")
    plot_examples(edges, args.output / "graph_examples.png")
    print(summary.to_string(index=False))
    print(f"Offline input: {len(local_edges)} directed edge-windows, {local_edges['periodic'].sum()} periodic.")


if __name__ == "__main__":
    main()
