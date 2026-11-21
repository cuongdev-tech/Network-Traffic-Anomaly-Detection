"""Load the versioned UNSW-NB15 data and reproduce the TabR split."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "unsw_nb15"
DROP_COLS = {"label", "attack_cat", "id", "srcip", "dstip", "stime", "ltime"}


def load_and_balance_unsw(normal_samples=20000, seed=42, *, data_dir=DATA_DIR, verbose=True):
    if verbose:
        print("=" * 65)
        print("1. DOC VA CAN BANG DU LIEU UNSW-NB15 (10 LOP)...")
    train_path = data_dir / "UNSW_NB15_training-set.parquet"
    test_path = data_dir / "UNSW_NB15_testing-set.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Khong tim thay file parquet UNSW_NB15 trong {data_dir}")

    frames = []
    for p in (train_path, test_path):
        df = pd.read_parquet(p)
        df.columns = [str(c).strip().lower() for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    raw_total = len(data)
    if verbose:
        print(f"  -> Tong so dong raw ban dau: {raw_total:,}")

    # Chuan hoa attack_cat (10 lop)
    data["attack_cat"] = data["attack_cat"].astype(str).str.strip()
    data.loc[data["attack_cat"].str.lower() == "normal", "attack_cat"] = "Normal"

    if verbose:
        print("  -> Phan bo 10 lop truoc khi can bang:")
        for lbl, count in data["attack_cat"].value_counts().items():
            print(f"     • {lbl:<15}: {count:>7,} ({count / raw_total * 100:>5.2f}%)")

    # Giam bot so dong cua Normal de can bang voi 9 lop con lai
    df_normal = data[data["attack_cat"] == "Normal"]
    df_attack = data[data["attack_cat"] != "Normal"]

    if len(df_normal) > normal_samples:
        df_normal_balanced = df_normal.sample(n=normal_samples, random_state=seed)
        if verbose:
            print(f"\n  -> Da giam Normal tu {len(df_normal):,} xuong {len(df_normal_balanced):,} dong.")
    else:
        df_normal_balanced = df_normal

    balanced_data = pd.concat([df_normal_balanced, df_attack], ignore_index=True)
    balanced_data = balanced_data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if verbose:
        print(f"  -> Tong so dong sau khi can bang: {len(balanced_data):,}")
        print("  -> Phan bo 10 lop sau khi can bang:")
        for lbl, count in balanced_data["attack_cat"].value_counts().items():
            print(f"     • {lbl:<15}: {count:>7,} ({count / len(balanced_data) * 100:>5.2f}%)")
        print("=" * 65)

    y = balanced_data["attack_cat"]
    X = balanced_data.drop(columns=list(DROP_COLS), errors="ignore")
    return X, y


def split_for_tabr(features, labels):
    train, test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    inner, validation, y_inner, y_validation = train_test_split(
        train, y_train, test_size=0.1, stratify=y_train, random_state=42,
    )
    return inner, validation, test, y_inner, y_validation, y_test


def test_index_hash(frame):
    return hashlib.sha256(frame.index.to_numpy(dtype="<i8").tobytes()).hexdigest()
