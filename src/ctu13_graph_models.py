"""Cross-validate supervised botnet edge classification with source-host groups."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def graph_features(edges):
    """Compute features from unlabeled window graphs, with no host identifiers."""
    windows = ["capture", "window"]
    counts = edges["connections"]
    out_degree = edges.groupby(windows + ["src"])["dst"].transform("size")
    in_degree = edges.groupby(windows + ["dst"])["src"].transform("size")
    source_activity = counts.groupby([edges[c] for c in windows + ["src"]]).transform("sum")
    destination_activity = counts.groupby([edges[c] for c in windows + ["dst"]]).transform("sum")
    observed = pd.MultiIndex.from_frame(edges[windows + ["src", "dst"]])
    reverse = pd.MultiIndex.from_frame(edges[windows + ["dst", "src"]])
    return pd.DataFrame({
        "connections": np.log1p(counts),
        "mean_gap": np.log1p(edges["mean_gap_s"]),
        "gap_variation": edges["gap_cv"],
        "out_degree": np.log1p(out_degree),
        "in_degree": np.log1p(in_degree),
        "source_activity": np.log1p(source_activity),
        "destination_activity": np.log1p(destination_activity),
        "source_share": counts / source_activity,
        "destination_share": counts / destination_activity,
        "reciprocal": reverse.isin(observed).astype(int),
    }, index=edges.index)


def evaluate_graph_models(edges):
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42, n_jobs=2),
        ),
    }
    features = graph_features(edges)
    rows, details = [], []
    for capture, graph in edges.groupby("capture", sort=True):
        known = graph.loc[graph["truth"].ne("unknown")]
        actual = known["truth"].eq("botnet")
        sources = actual.groupby(known["src"], sort=True).any()
        positive_sources = int(sources.sum())
        negative_sources = len(sources) - positive_sources
        if min(positive_sources, negative_sources) < 3:
            raise ValueError(f"{capture} needs at least three source hosts in each stratum.")
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        predictions = {name: pd.Series(False, index=known.index) for name in models}
        for fold, (train, test) in enumerate(folds.split(sources, sources), 1):
            train_mask = known["src"].isin(sources.index[train])
            test_mask = known["src"].isin(sources.index[test])
            for name, model in models.items():
                model.fit(features.loc[known.index[train_mask]], actual.loc[train_mask])
                test_index = known.index[test_mask]
                forecast = model.predict(features.loc[test_index])
                predictions[name].loc[test_index] = forecast
                details.append({
                    "capture": capture, "method": name, "fold": fold,
                    "train_sources": len(train), "test_sources": len(test),
                    "test_edges": len(test_index), "test_positive_edges": int(actual.loc[test_index].sum()),
                    "true_positives": int((forecast & actual.loc[test_index]).sum()),
                })
        for name, forecast in predictions.items():
            tn, fp, fn, tp = confusion_matrix(actual, forecast, labels=[False, True]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, forecast, average="binary", zero_division=0,
            )
            rows.append({
                "capture": capture, "method": name, "sources": len(sources),
                "positive_sources": positive_sources, "scored_edges": len(known),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "precision": precision, "recall": recall, "f1": f1,
            })
    return pd.DataFrame(rows), pd.DataFrame(details)
