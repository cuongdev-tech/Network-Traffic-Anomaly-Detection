"""Predict CSV flows using only the project's trusted model artifacts."""
import csv
import gzip
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from giao_dien.data import DATASETS, MODEL_NAMES


def normalize_predictions(predictions):
    meanings = {
        "BENIGN": (0, "Bình thường", "Không có tấn công"),
        "Normal": (0, "Bình thường", "Không có tấn công"),
        "ATTACK": (1, "Tấn công", "Tấn công khác"),
        "DDoS": (1, "Tấn công", "DDoS"),
        "PortScan": (1, "Tấn công", "PortScan"),
        "Generic": (1, "Tấn công", "Generic"),
        "Exploits": (1, "Tấn công", "Exploits"),
        "Fuzzers": (1, "Tấn công", "Fuzzers"),
        "DoS": (1, "Tấn công", "DoS"),
        "Reconnaissance": (1, "Tấn công", "Reconnaissance"),
        "Analysis": (1, "Tấn công", "Analysis"),
        "Backdoor": (1, "Tấn công", "Backdoor"),
        "Shellcode": (1, "Tấn công", "Shellcode"),
        "Worms": (1, "Tấn công", "Worms"),
    }
    unknown = set(predictions).difference(meanings)
    if unknown:
        raise ValueError("Model trả về nhãn không được hỗ trợ: " + ", ".join(sorted(unknown)))
    result = pd.DataFrame([meanings[value] for value in predictions],
                          columns=["prediction_code", "prediction_status", "attack_type"])
    result["prediction"] = predictions
    result.insert(0, "prediction_class", [
        "Bình thường" if value in {"Normal", "BENIGN"} else
        "Tấn công khác" if value == "ATTACK" else value for value in predictions
    ])
    return result


def model_paths(root, dataset):
    if dataset not in DATASETS:
        raise ValueError("Bộ dữ liệu không hợp lệ.")
    folder = Path(root) / "models" / ("cic" if dataset == "cic" else "unsw_nb15")
    baseline = "baseline_" if dataset == "cic" else "unsw_nb15_baseline_"
    filenames = {name: f"{baseline}{name}.joblib"
                 for name in ("LogisticRegression", "DecisionTree", "KNN")}
    filenames.update({"Random Forest": "random_forest_attack.joblib",
                      "XGBoost": "xgboost_attack.joblib", "TabNet": "tabnet_bundle.joblib",
                      "TabM": "tabm_bundle.joblib"})
    if dataset == "unsw":
        filenames["TabR"] = "tabr_checkpoint.joblib"
    return {name: folder / filename for name, filename in filenames.items()
            if (folder / filename).is_file()}


def load_bundle(root, dataset, name):
    path = model_paths(root, dataset).get(name)
    if path is None:
        raise ValueError("Không tìm thấy model đã chọn trong thư mục models.")
    # Only load bundled artifacts; uploaded files are data, never pickle/joblib.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not bundle.get("features"):
        raise ValueError("Model thiếu thông tin đặc trưng hoặc nhãn.")
    labels = bundle.get("classes", bundle.get("labels"))
    if labels is None or not len(labels):
        raise ValueError("Model thiếu danh sách classes/labels.")
    bundle["labels"] = list(labels)
    if name == "TabNet":
        weights = path.with_name("tabnet_attack.zip")
        if weights.is_file():
            from pytorch_tabnet.tab_model import TabNetClassifier
            model = TabNetClassifier(device_name="cpu", verbose=0)
            model.load_model(str(weights))
            bundle["model"] = model
        elif "model" not in bundle:
            raise ValueError("Thiếu trọng số models/.../tabnet_attack.zip.")
        if getattr(bundle["model"], "output_dim", len(bundle["labels"])) != len(bundle["labels"]):
            raise ValueError("Trọng số TabNet không khớp số lớp trong metadata. Cần dùng cùng lần huấn luyện.")
        if not any(key in bundle for key in ("scaler", "preprocessor")):
            raise ValueError("TabNet thiếu scaler/preprocessor đã được fit khi huấn luyện.")
    if name == "TabR":
        from src.tabr_training import load_classifier

        bundle["model"] = load_classifier(Path(root), bundle)
    if "model" not in bundle:
        raise ValueError("File model không chứa mô hình dự đoán.")
    if dataset == "unsw" and "classes" in bundle and "preprocessor" not in bundle:
        shared_path = path.with_name("preprocessor_bundle.joblib")
        if not shared_path.is_file():
            raise ValueError("Thiếu bộ tiền xử lý UNSW. Cần lưu preprocessor từ cùng lần huấn luyện; không fit lại trên CSV nhập.")
        shared = joblib.load(shared_path)
        if shared.get("features") != bundle["features"] or list(shared.get("classes", [])) != bundle["labels"]:
            raise ValueError("Bộ tiền xử lý UNSW không khớp đặc trưng hoặc lớp của model.")
        bundle["preprocessor"] = shared["preprocessor"]
    bundle["_version_warning"] = ""
    if any(isinstance(item.message, InconsistentVersionWarning) for item in notices):
        bundle["_version_warning"] = (
            "Lưu ý: phiên bản scikit-learn hiện tại khác phiên bản lưu model. "
            "Nên dùng cùng môi trường huấn luyện hoặc train lại trước khi sử dụng thực tế.")
    return bundle


def predict_file(root, dataset, name, path):
    path = Path(path)
    if not path.name.lower().endswith((".csv", ".csv.gz")):
        raise ValueError("Hãy chọn file CSV hoặc CSV.GZ; PCAP cần trích đặc trưng trước.")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        headers = [column.strip() for column in next(csv.reader(stream), [])]
    if len(headers) != len(set(headers)):
        raise ValueError("File có tên cột trùng nhau sau khi bỏ khoảng trắng.")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip()
    if frame.empty:
        raise ValueError("File không có dòng dữ liệu để dự đoán.")
    reserved = {"prediction", "prediction_class", "prediction_code", "prediction_status", "attack_type",
                "prediction_probability", "prediction_dataset", "prediction_model"}
    if reserved.intersection(frame.columns):
        raise ValueError("File đã chứa cột kết quả dự đoán. Hãy chọn dữ liệu đầu vào gốc.")
    bundle = load_bundle(root, dataset, name)
    features = bundle["features"]
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} đặc trưng: " + ", ".join(missing))
    X = frame.loc[:, features].copy()
    categorical = {"proto", "service", "state"} if dataset == "unsw" else set()
    for column in features:
        if column in categorical:
            if X[column].isna().any():
                raise ValueError(f"Cột {column} có giá trị thiếu.")
            X[column] = X[column].astype(str).str.strip()
            if X[column].eq("").any():
                raise ValueError(f"Cột {column} có giá trị trống.")
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            if not np.isfinite(X[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Cột {column} có giá trị thiếu, không phải số hoặc NaN/inf.")
    model = bundle["model"]
    labels = list(bundle["labels"])
    predictions, probabilities = [], []
    for start in range(0, len(X), 1000):
        batch = X.iloc[start:start + 1000]
        if "preprocessor" in bundle:
            batch = bundle["preprocessor"].transform(batch).astype(np.float32)
        elif "scaler" in bundle:
            batch = bundle["scaler"].transform(batch.to_numpy(dtype=np.float32)).astype(np.float32)
        elif "classes" in bundle and not hasattr(model, "feature_names_in_"):
            batch = batch.to_numpy(dtype=np.float32)
        expected = getattr(model, "n_features_in_", getattr(model, "input_dim", None))
        if expected is not None and batch.shape[1] != expected:
            raise ValueError(f"Sau tiền xử lý có {batch.shape[1]} cột, nhưng model cần {expected}. Kiểm tra bundle huấn luyện.")
        batch_probabilities = model.predict_proba(batch) if name in {"TabM", "TabR"} else None
        output = model.predict(batch) if batch_probabilities is None else batch_probabilities.argmax(axis=1)
        if any(isinstance(value, (int, np.integer)) and not 0 <= int(value) < len(labels) for value in output):
            raise ValueError("Model trả mã lớp ngoài danh sách nhãn đã lưu.")
        predictions.extend(labels[int(value)] if isinstance(value, (int, np.integer))
                           else str(value) for value in output)
        if batch_probabilities is None and hasattr(model, "predict_proba"):
            batch_probabilities = model.predict_proba(batch)
        if batch_probabilities is not None:
            probabilities.extend(batch_probabilities.max(axis=1).tolist())
    result = normalize_predictions(predictions)
    result.index = frame.index
    if probabilities:
        result["prediction_probability"] = probabilities
    result["prediction_dataset"] = DATASETS[dataset]
    result["prediction_model"] = MODEL_NAMES[name]
    result = pd.concat([result, frame], axis=1)
    result.attrs["model_warning"] = bundle["_version_warning"]
    return result
