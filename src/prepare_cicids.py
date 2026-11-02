"""Chuan bi bo du lieu chinh CIC-IDS2017 (thay cho NETRESEC vi qua it flow).

Doc 2 file CSV goc, lam sach, lay mau stratified 60k va luu 1 file duy nhat
de EDA / train / thanh vien khac dung chung, khong phai xu ly lai.

Chay: python src/prepare_cicids.py
Output: data/processed/cicids_60k.csv.gz
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "cicids2017"
OUT_PATH = BASE_DIR / "data" / "processed" / "cicids_60k.csv.gz"

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Fwd Header Length.1"]


def load_raw():
    frames = []
    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Thieu {path}.\nTai 2 file CSV tu "
                "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017 "
                "va dat vao data/raw/cicids2017/ (xem README)."
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"  {name}: {len(df):,} dong")

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")
    data["Label"] = data["Label"].astype(str).str.strip()
    data.loc[data["Label"].str.upper() == "BENIGN", "Label"] = "BENIGN"
    return data


def main():
    print("Doc du lieu CIC-IDS2017...")
    data = load_raw()
    print(f"\nTong sau khi lam sach: {len(data):,} flow")
    print(data["Label"].value_counts().to_string())

    _, sample = train_test_split(
        data, test_size=60000, random_state=42, stratify=data["Label"])
    sample = sample.reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False, compression="gzip")

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nDa luu {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"Shape: {sample.shape} | So dac trung: {sample.shape[1] - 1}")
    print(sample["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
