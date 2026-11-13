"""Gop ket qua 5 mo hinh baseline cua ca 2 bo du lieu vao mot bang duy nhat.

Doc:
    results/baselines_metrics.csv            (CIC-IDS2017)
    results/unsw_nb15/baselines_metrics.csv  (UNSW-NB15)

Xuat:
    results/baselines_all_datasets.csv
    results/baselines_all_datasets.png

Chay: python src/compare_baselines.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES = {
    "CIC-IDS2017": BASE_DIR / "results" / "baselines_metrics.csv",
    "UNSW-NB15": BASE_DIR / "results" / "unsw_nb15" / "baselines_metrics.csv",
}
OUT_CSV = BASE_DIR / "results" / "baselines_all_datasets.csv"
OUT_PNG = BASE_DIR / "results" / "baselines_all_datasets.png"

COLUMNS = ["dataset", "model", "group", "accuracy", "precision_weighted",
           "recall_weighted", "f1_weighted", "f1_macro",
           "train_time_s", "predict_time_s"]


def main():
    frames = []
    for dataset, path in SOURCES.items():
        if not path.is_file():
            print(f"Bo qua {dataset}: khong tim thay {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)
        frames.append(frame)

    if not frames:
        raise SystemExit("Chua co file ket qua baseline. Hay chay 2 script train truoc.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[[c for c in COLUMNS if c in combined.columns]]
    combined = combined.sort_values(["dataset", "f1_weighted"], ascending=[True, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    print(combined[["dataset", "model", "accuracy", "f1_weighted",
                    "f1_macro"]].to_string(index=False))

    for metric in ("accuracy", "f1_weighted"):
        pivot = combined.pivot(index="model", columns="dataset", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"5 mo hinh baseline - {metric} tren 2 bo du lieu")
        ax.set_xlabel("")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUT_PNG.with_name(f"baselines_all_datasets_{metric}.png"), dpi=150)
        plt.close()

    print(f"\nDa luu: {OUT_CSV}")
    print(f"Da luu hinh: {OUT_PNG.with_name('baselines_all_datasets_*.png')}")


if __name__ == "__main__":
    main()
