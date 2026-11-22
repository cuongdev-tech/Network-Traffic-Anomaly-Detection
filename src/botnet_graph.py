"""Find periodic connections and candidate topologies in offline flow metadata."""
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


EDGE_COLUMNS = [
    "capture", "window", "src", "dst", "connections", "mean_gap_s", "std_gap_s",
    "gap_cv", "periodic",
]
TOPOLOGY_METHODS = ("topology_only", "periodic_topology")
COMPONENT_COLUMNS = [
    "capture", "window", "method", "component", "topology", "nodes", "edges",
    "hub", "hub_coverage", "core_fraction",
]


def _classify_component(graph):
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = dict(graph.degree())
    hub = max(sorted(degrees), key=degrees.get)
    coverage = degrees[hub] / (n - 1)
    core_fraction = len(nx.k_core(graph, k=2)) / n
    if n >= 4 and coverage >= 0.8 and m <= 1.2 * (n - 1):
        topology = "centralized"
    elif n >= 4 and core_fraction >= 0.8 and degrees[hub] <= 2 * np.median(list(degrees.values())):
        topology = "p2p"
    else:
        topology = "unclassified"
    return topology, n, m, hub if topology == "centralized" else "", coverage, core_fraction


def windowed_flows(flows, window_seconds):
    """Normalize endpoints and assign time windows while retaining source row indices."""
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window must be positive.")
    required = {"src", "dst", "start"}
    if not required.issubset(flows):
        raise ValueError("Expected src, dst and start (seconds), with optional file for capture identity.")
    data = flows.loc[:, sorted(required)].copy()
    data["capture"] = flows["file"] if "file" in flows else "input"
    for column in ("src", "dst", "capture"):
        data[column] = data[column].astype("string").str.strip()
    data["start"] = pd.to_numeric(data["start"], errors="coerce")
    if data.isna().any().any() or not np.isfinite(data["start"]).all():
        raise ValueError("Missing connection endpoints, capture IDs, or non-finite timestamps.")
    if data[["src", "dst", "capture"]].eq("").any().any():
        raise ValueError("Endpoints and capture IDs must be non-empty.")
    data = data.loc[data["src"] != data["dst"]]
    data["window"] = ((data["start"] - data.groupby("capture")["start"].transform("min"))
                      // window_seconds).astype(int)
    return data


def analyze_flows(flows, *, window_seconds=900, min_connections=5, max_gap_cv=0.15):
    """Return directed-pair evidence and candidate components in each capture window."""
    if min_connections < 3 or not np.isfinite(max_gap_cv) or max_gap_cv < 0:
        raise ValueError("Minimum connections must be >=3 and tolerance nonnegative.")
    data = windowed_flows(flows, window_seconds)
    edge_rows = []
    for key, group in data.groupby(["capture", "window", "src", "dst"], sort=True):
        starts = np.unique(group["start"].to_numpy())
        gaps = np.diff(starts)
        mean = gaps.mean() if gaps.size else np.nan
        std = gaps.std() if gaps.size else np.nan
        cv = std / mean
        periodic = len(starts) >= min_connections and cv <= max_gap_cv
        edge_rows.append((*key, len(starts), mean, std, cv, periodic))
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS).astype({"periodic": bool})

    component_rows = []
    for method in TOPOLOGY_METHODS:
        edges[method] = "unclassified"
    for (capture, window), group in edges.groupby(["capture", "window"], sort=True):
        for method, selected in zip(TOPOLOGY_METHODS, (group, group.loc[group["periodic"]])):
            graph = nx.from_pandas_edgelist(selected, source="src", target="dst")
            components = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), min(nodes)))
            labels = {}
            for number, nodes in enumerate(components, 1):
                features = _classify_component(graph.subgraph(nodes))
                component_rows.append((capture, window, method, number, *features))
                labels.update(dict.fromkeys(nodes, features[0]))
            edges.loc[selected.index, method] = selected["src"].map(labels)
    return edges, pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Offline CSV, e.g. data/processed/flows.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--window-seconds", type=float, default=900)
    parser.add_argument("--min-connections", type=int, default=5)
    parser.add_argument("--max-gap-cv", type=float, default=0.15)
    args = parser.parse_args()

    edges, components = analyze_flows(
        pd.read_csv(args.input), window_seconds=args.window_seconds,
        min_connections=args.min_connections, max_gap_cv=args.max_gap_cv,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.output / "edges.csv", index=False)
    components.to_csv(args.output / "components.csv", index=False)
    print(f"{len(edges)} directed edges, {edges['periodic'].sum()} periodic edge-windows")


if __name__ == "__main__":
    main()
