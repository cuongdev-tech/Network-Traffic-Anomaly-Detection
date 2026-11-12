"""Read the project's result files without loading models or raw datasets."""
import json
from pathlib import Path

import pandas as pd

DATASETS = {"cic": "CIC-IDS2017", "unsw": "UNSW-NB15"}
METRICS = {
    "accuracy": "Accuracy",
    "precision_weighted": "Precision weighted",
    "recall_weighted": "Recall weighted",
    "f1_weighted": "F1 weighted",
    "f1_macro": "F1 macro",
}
MODEL_NAMES = {
    "LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
    "KNN": "KNN", "Random Forest": "Random Forest", "XGBoost": "XGBoost", "MLP": "MLP",
}
SOURCES = (
    ("baselines", "Baseline", {"LogisticRegression", "DecisionTree", "KNN"}),
    ("random_forest_xgboost", "RF / XGBoost", {"Random Forest", "XGBoost"}),
    ("mlp", "MLP", {"MLP"}),
)


def result_dir(root, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Bộ dữ liệu không hợp lệ: {dataset}")
    return Path(root) / "results" / ("unsw_nb15" if dataset == "unsw" else "")


def read_csv(path):
    try:
        return pd.read_csv(path), ""
    except (OSError, ValueError) as error:
        return pd.DataFrame(), f"Không đọc được {Path(path).name}: {error}"


def load_results(root, dataset):
    folder = result_dir(root, dataset)
    frames, issues = [], []
    for stem, group, names in SOURCES:
        frame, error = read_csv(folder / f"{stem}_metrics.csv")
        if error:
            issues.append(error)
            continue
        if not {"model", "accuracy"}.issubset(frame.columns):
            issues.append(f"{stem}_metrics.csv thiếu cột model hoặc accuracy.")
            continue
        if "dataset" in frame:
            frame = frame.loc[frame["dataset"] == dataset].copy()
        frame = frame.loc[frame["model"].isin(names)].copy()
        for column in [*METRICS, "train_time_s", "predict_time_s"]:
            frame[column] = pd.to_numeric(frame.get(column, float("nan")), errors="coerce")
            invalid = ~frame[column].between(0, 1) if column in METRICS else frame[column] < 0
            if (invalid & frame[column].notna()).any():
                issues.append(f"{stem}_metrics.csv có giá trị {column} không hợp lệ.")
                frame.loc[invalid, column] = float("nan")
        try:
            meta = json.loads((folder / f"{stem}_meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("metadata phải là object")
        except (OSError, ValueError) as error:
            meta = {}
            issues.append(f"Không đọc được {stem}_meta.json: {error}")
        train, test = meta.get("train"), meta.get("test")
        split = f"{train:,} / {test:,}" if isinstance(train, int) and isinstance(test, int) else "Chưa có"
        protocol = "Split chính thức" if "official" in str(meta.get("split", "")) else "Chia ngẫu nhiên"
        frame["_protocol"] = f"{protocol} · {split}"
        frame["_group"], frame["_stem"] = group, stem
        frame["_meta"] = [meta] * len(frame)
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), issues


def confusion_path(folder, model):
    slug = {"Random Forest": "random_forest", "XGBoost": "xgboost", "MLP": "mlp"}.get(model, model)
    return Path(folder) / f"cm_{slug}.png"


def per_class(folder, row):
    frame, error = read_csv(Path(folder) / f"{row['_stem']}_per_class.csv")
    if error or not {"model", "class"}.issubset(frame.columns):
        return pd.DataFrame(), error or "Bảng chỉ số từng lớp thiếu cột model/class."
    frame = frame.loc[frame["model"] == row["model"]].drop(columns="model").copy()
    classes = row["_meta"].get("classes", [])
    mapping = {str(index): name for index, name in enumerate(classes)}
    frame["class"] = frame["class"].astype(str).replace(mapping)
    return frame, ""
