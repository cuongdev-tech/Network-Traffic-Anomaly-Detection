"""Exploratory data analysis for the official UNSW-NB15 train/test files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
OUT_DIR = BASE_DIR / "results" / "eda_unsw_nb15"
SPLITS = {
    "train": "UNSW_NB15_training-set.parquet",
    "test": "UNSW_NB15_testing-set.parquet",
}


def load_data():
    frames = []
    for split, filename in SPLITS.items():
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Khong tim thay {path}. Tai bo UNSW-NB15 tu "
                "https://www.kaggle.com/datasets/dhoogla/unswnb15 va dat hai "
                "file train/test trong data/raw/unsw_nb15/."
            )
        frame = pd.read_parquet(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"label", "attack_cat"}
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} thieu cot bat buoc: {sorted(missing_columns)}")
        frame["split"] = split
        frames.append(frame)
        print(f"{split}: {len(frame):,} dong | {path.name}")

    if frames[0].columns.difference(["split"]).tolist() != frames[1].columns.difference(["split"]).tolist():
        raise ValueError("Hai split UNSW-NB15 khong co cung bo cot.")
    return pd.concat(frames, ignore_index=True)


def save_distribution_plot(counts, title, ylabel, output, horizontal=False):
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.42)) if horizontal else (8, 5))
    plot = counts.sort_values().plot(
        kind="barh" if horizontal else "bar",
        ax=ax,
        color="#4C72B0",
    )
    ax.set_title(title)
    ax.set_xlabel(ylabel if horizontal else "")
    ax.set_ylabel("" if horizontal else ylabel)
    if not horizontal:
        ax.tick_params(axis="x", rotation=25)
    for container in plot.containers:
        plot.bar_label(container, fmt="%.0f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / output, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    feature_columns = [column for column in data.columns if column not in {
        "label", "attack_cat", "split",
    }]
    numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    numeric_data = data[numeric_features]
    missing_values = int(data.isna().sum().sum())
    infinite_values = int(np.isinf(numeric_data.to_numpy(dtype=float)).sum())
    feature_columns = [
        column for column in data.columns
        if column not in {"label", "attack_cat", "split"}
    ]
    train = data.loc[data["split"] == "train"]
    test = data.loc[data["split"] == "test"]
    train_feature_hashes = pd.util.hash_pandas_object(
        train[feature_columns], index=False
    )
    test_feature_hashes = pd.util.hash_pandas_object(
        test[feature_columns], index=False
    )
    train_feature_signatures = set(train_feature_hashes.unique())
    test_feature_signatures = set(test_feature_hashes.unique())
    train_overlap_rows = int(train_feature_hashes.isin(test_feature_signatures).sum())
    test_overlap_rows = int(test_feature_hashes.isin(train_feature_signatures).sum())
    overlap = pd.DataFrame([
        {
            "metric": "distinct_feature_signatures_in_both_splits",
            "count": len(train_feature_signatures & test_feature_signatures),
        },
        {
            "metric": "train_rows_with_feature_signature_in_test",
            "count": train_overlap_rows,
        },
        {
            "metric": "test_rows_with_feature_signature_in_train",
            "count": test_overlap_rows,
        },
        {
            "metric": "train_duplicate_feature_rows",
            "count": int(train_feature_hashes.duplicated().sum()),
        },
        {
            "metric": "test_duplicate_feature_rows",
            "count": int(test_feature_hashes.duplicated().sum()),
        },
    ])
    overlap["percent_of_split"] = overlap.apply(
        lambda row: round(
            row["count"] / len(train if row["metric"].startswith("train") else test) * 100,
            2,
        ) if row["metric"].startswith(("train", "test")) else np.nan,
        axis=1,
    )
    overlap.to_csv(OUT_DIR / "split_feature_overlap.csv", index=False)
    overview = pd.DataFrame([{
        "dataset": "UNSW-NB15",
        "source": "Kaggle dhoogla/unswnb15",
        "rows": len(data),
        "train_rows": int(data["split"].eq("train").sum()),
        "test_rows": int(data["split"].eq("test").sum()),
        "feature_columns": len(feature_columns),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "missing_values": missing_values,
        "infinite_numeric_values": infinite_values,
        "duplicate_rows": int(data.duplicated().sum()),
        "distinct_feature_signatures_in_both_splits": len(
            train_feature_signatures & test_feature_signatures
        ),
        "test_rows_with_feature_signature_in_train": test_overlap_rows,
        "test_rows_with_feature_signature_in_train_percent": round(
            test_overlap_rows / len(test) * 100, 2
        ),
    }])
    overview.to_csv(OUT_DIR / "dataset_overview.csv", index=False)
    print("\nDataset overview:")
    print(overview.to_string(index=False))

    label_counts = (
        data.groupby(["split", "label"], observed=True).size()
        .rename("count").reset_index()
    )
    label_counts["percent_within_split"] = (
        label_counts["count"]
        / label_counts.groupby("split")["count"].transform("sum")
        * 100
    ).round(2)
    label_counts.to_csv(OUT_DIR / "binary_label_distribution.csv", index=False)

    overall_labels = data["label"].value_counts().sort_index()
    save_distribution_plot(
        overall_labels,
        "UNSW-NB15 binary label distribution",
        "Rows",
        "binary_label_distribution.png",
    )

    attack_counts = (
        data["attack_cat"].fillna("Missing").astype(str).value_counts().sort_values()
    )
    attack_table = attack_counts.rename_axis("attack_cat").rename("count").reset_index()
    attack_table["percent"] = (attack_table["count"] / len(data) * 100).round(2)
    attack_table.to_csv(OUT_DIR / "attack_category_distribution.csv", index=False)
    save_distribution_plot(
        attack_counts,
        "UNSW-NB15 attack categories (includes Normal)",
        "Rows",
        "attack_category_distribution.png",
        horizontal=True,
    )

    categorical_rows = []
    for column in categorical_features:
        counts = data[column].fillna("Missing").astype(str).value_counts()
        for value, count in counts.items():
            categorical_rows.append({
                "feature": column,
                "value": value,
                "count": int(count),
                "percent": round(float(count / len(data) * 100), 2),
            })
        top = counts.head(10).sort_values()
        save_distribution_plot(
            top,
            f"Top values: {column}",
            "Rows",
            f"categorical_{column}.png",
            horizontal=True,
        )
    pd.DataFrame(categorical_rows).to_csv(
        OUT_DIR / "categorical_feature_distribution.csv", index=False
    )

    summary_features = [
        column for column in numeric_features if data[column].nunique(dropna=True) > 1
    ]
    grouped_summary = data.groupby("label", observed=True)[summary_features].agg(
        ["mean", "median", "std"]
    )
    grouped_summary.columns = [
        f"{feature}_{statistic}" for feature, statistic in grouped_summary.columns
    ]
    grouped_summary.to_csv(OUT_DIR / "numeric_summary_by_label.csv")

    selected_features = [
        column for column in (
            "dur", "rate", "sload", "dload", "spkts", "dpkts", "sbytes", "dbytes"
        ) if column in numeric_features
    ]
    transformed = data[["label", *selected_features]].copy()
    transformed[selected_features] = np.log1p(
        transformed[selected_features].clip(lower=0)
    )
    melted = transformed.melt(
        id_vars="label", var_name="feature", value_name="log1p(value)"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=melted,
        x="feature",
        y="log1p(value)",
        hue="label",
        showfliers=False,
        ax=ax,
    )
    ax.set_title("UNSW-NB15 numeric features by binary label (log1p scale)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "numeric_features_by_label.png", dpi=150)
    plt.close(fig)

    correlation = data[numeric_features].corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("UNSW-NB15 numeric feature correlations")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "numeric_correlation.png", dpi=150)
    plt.close()

    print("\nBinary labels:")
    print(data["label"].value_counts().to_string())
    print("\nAttack categories:")
    print(data["attack_cat"].fillna("Missing").value_counts().to_string())
    print("\nExact feature overlap across official splits:")
    print(overlap.to_string(index=False))
    print(f"\nEDA outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
