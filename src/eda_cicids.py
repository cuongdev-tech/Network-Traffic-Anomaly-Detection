"""EDA tren bo du lieu CIC-IDS2017 (phan viec cua mo hinh baseline LR/DT/KNN).

Xuat hinh vao results/eda_cicids/ va bang thong ke results/eda_cicids/summary.csv.

Chay: python src/eda_cicids.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "cicids2017"
OUT_DIR = BASE_DIR / "results" / "eda_cicids"

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

USECOLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "Label",
]


def load():
    frames = []
    for name in FILES:
        path = DATA_DIR / name
        header = pd.read_csv(path, nrows=0)
        mapping = {c.strip(): c for c in header.columns}
        usecols = [mapping[c] for c in USECOLS if c in mapping]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["Label"] = data["Label"].astype(str).str.strip()
    data.loc[data["Label"].str.upper() == "BENIGN", "Label"] = "BENIGN"
    return data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Tong so flow: {len(df):,}")

    counts = df["Label"].value_counts()
    print("\nPhan bo lop:")
    print(counts.to_string())

    # 1. Phan bo lop
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Phan bo lop - CIC-IDS2017 (Friday afternoon)")
    plt.ylabel("So flow")
    plt.xticks(rotation=0)
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # 2. Flow Duration theo lop (log scale)
    plt.figure(figsize=(7, 4))
    df.boxplot(column="Flow Duration", by="Label", showfliers=False)
    plt.yscale("log")
    plt.suptitle("")
    plt.title("Flow Duration theo lop (log scale)")
    plt.ylabel("Duration (microseconds)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "duration_by_class.png", dpi=150)
    plt.close()

    # 3. Flow Bytes/s theo lop (log scale)
    plt.figure(figsize=(7, 4))
    df.boxplot(column="Flow Bytes/s", by="Label", showfliers=False)
    plt.yscale("log")
    plt.suptitle("")
    plt.title("Flow Bytes/s theo lop (log scale)")
    plt.ylabel("Bytes/s")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bytes_per_sec_by_class.png", dpi=150)
    plt.close()

    # 4. Top 10 port dich theo tung lop
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, label in zip(axes, ["BENIGN", "DDoS", "PortScan"]):
        top = df.loc[df["Label"] == label, "Destination Port"].value_counts().head(10)
        top.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top 10 port dich - {label}")
        ax.set_xlabel("So flow")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_ports_by_class.png", dpi=150)
    plt.close()

    # 5. Co TCP trung binh theo lop
    flags = df.groupby("Label")[["SYN Flag Count", "RST Flag Count", "ACK Flag Count"]].mean()
    flags.plot(kind="bar", figsize=(7, 4))
    plt.title("So co TCP trung binh moi flow theo lop")
    plt.ylabel("Trung binh so co")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tcp_flags_by_class.png", dpi=150)
    plt.close()

    # Bang thong ke
    summary = df.groupby("Label").agg(
        n_flows=("Label", "size"),
        duration_mean_us=("Flow Duration", "mean"),
        duration_median_us=("Flow Duration", "median"),
        bytes_per_s_median=("Flow Bytes/s", "median"),
        fwd_pkts_mean=("Total Fwd Packets", "mean"),
        bwd_pkts_mean=("Total Backward Packets", "mean"),
        syn_mean=("SYN Flag Count", "mean"),
        rst_mean=("RST Flag Count", "mean"),
        n_dst_ports=("Destination Port", "nunique"),
    ).round(2)
    summary.to_csv(OUT_DIR / "summary.csv")
    print("\nThong ke theo lop:")
    print(summary.to_string())
    print(f"\nDa luu hinh + summary vao {OUT_DIR}")


if __name__ == "__main__":
    main()
