
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/raw/unsw_realtime/NF-UNSW-NB15.csv"

OUTPUT = ROOT / "data/processed/unsw_realtime/netflow_v1"

SEED = 42

FEATURES = [
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

LIMITS = {
    "Normal": 30000,
    "Exploits": 5000,
    "Fuzzers": 4000,
    "Reconnaissance": 3000,
    "Generic": 2000,
    "DoS": 2000,
    "Analysis": 1000,
    "Backdoor": 1000,
    "Shellcode": 800,
    "Worms": None,
}

LABEL_MAP = {
    name.lower(): name
    for name in LIMITS
}

LABEL_MAP.update({
    "benign": "Normal",
    "backdoors": "Backdoor",
})


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        SOURCE,
        usecols=FEATURES + ["Label", "Attack"],
        low_memory=False,
    )

    print("Original rows:", len(df))

    # Normalize labels

    df["target"] = (
        df["Attack"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)
    )

    if df["target"].isna().any():
        raise ValueError(
            "Unknown attack category detected"
        )

    binary = pd.to_numeric(
        df["Label"], errors="coerce"
    )

    expected = df["target"].ne("Normal").astype(int)

    if not binary.eq(expected).all():
        raise ValueError("Inconsistent dataset labels")

    df = df[FEATURES + ["target"]].copy()

    # Validate numeric features

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    numeric = df[FEATURES].to_numpy(dtype=float)

    valid = (
        np.isfinite(numeric).all(axis=1)
        & (numeric >= 0).all(axis=1)
    )

    df = df.loc[valid].copy()

    # Restrict V1 to TCP and UDP because the
    # current Scapy metadata supports them.

    df = df[df["PROTOCOL"].isin([6, 17])].copy()

    print("\n===== SUPPORTED DATA =====")

    print(df["target"].value_counts())

    # Group identical model inputs into
    # the same partition to reduce leakage.

    hashes = pd.util.hash_pandas_object(
        df[FEATURES],
        index=False,
    )

    buckets = hashes % 100

    train_pool = df.loc[buckets < 70].copy()

    val = df.loc[
        (buckets >= 70) & (buckets < 85)
    ].copy()

    test = df.loc[buckets >= 85].copy()

    # Undersample training only

    selected = []

    for label, limit in LIMITS.items():

        group = train_pool[
            train_pool["target"] == label
        ]

        if limit is not None:

            group = group.sample(
                n=min(limit, len(group)),
                random_state=SEED,
            )

        selected.append(group)

    train = pd.concat(
        selected,
        ignore_index=True,
    )

    train = train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # Check class coverage

    distribution = pd.DataFrame({

        "train": train["target"].value_counts(),

        "validation": val["target"].value_counts(),

        "test": test["target"].value_counts(),

    }).reindex(list(LIMITS)).fillna(0).astype(int)

    print("\n===== FINAL CLASS DISTRIBUTION =====")

    print(distribution.to_string())

    print("\n===== SPLIT SIZES =====")

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    if (distribution == 0).any().any():

        raise ValueError(
            "At least one class is missing from a split. "
            "Review the class distribution before training."
        )

    # Save prepared data

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        OUTPUT / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT / "validation.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT / "test.csv",
        index=False,
    )

    distribution.to_csv(
        OUTPUT / "class_distribution.csv"
    )

    print("\n===== SUCCESS =====")

    print("Features:", len(FEATURES))

    print("Classes:", train["target"].nunique())

    print("Saved to:", OUTPUT)


if __name__ == "__main__":

    main()
