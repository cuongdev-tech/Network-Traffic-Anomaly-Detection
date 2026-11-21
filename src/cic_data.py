"""Load the versioned CIC-IDS2017 CSV files for training and inference."""
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cicids2017"
FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Fwd Header Length.1", "source_file"]


def load_cleaned_cic(*, data_dir=DATA_DIR, verbose=True):
    if verbose:
        print("=" * 65)
        print("1. DOC VA TIEN XU LY DU LIEU CIC-IDS2017 (512.212 DONG GOC)...")
    frames = []
    for name in FILES:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        if verbose:
            print(f"  + {name}: {len(df):,} dong")

    raw_total = sum(len(f) for f in frames)
    data = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  -> Tong so dong raw: {raw_total:,}")

    # Loc inf, -inf va nan
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")

    data["Label"] = data["Label"].astype(str).str.strip()
    data.loc[data["Label"].str.upper() == "BENIGN", "Label"] = "BENIGN"

    if verbose:
        print(f"  -> Sau khi loai bo inf/nan: {len(data):,} dong hop le (giam {raw_total - len(data):,} dong loi)")
        print("  -> Phan bo nhan:")
        for lbl, count in data["Label"].value_counts().items():
            print(f"     • {lbl}: {count:,} ({count / len(data) * 100:.2f}%)")
        print("=" * 65)
    return data
