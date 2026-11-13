import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from giao_dien.data import eda_label, load_results, localize_eda_table, per_class


class ResultTests(unittest.TestCase):
    def test_dataset_filter_missing_files_and_real_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "results"
            folder.mkdir()
            pd.DataFrame([
                {"model": "MLP", "accuracy": 0, "dataset": "cic"},
                {"model": "MLP", "accuracy": .9, "dataset": "unsw"},
            ]).to_csv(folder / "mlp_metrics.csv", index=False)
            (folder / "mlp_meta.json").write_text(json.dumps({"train": 80, "test": 20}), encoding="utf-8")
            pd.DataFrame([{"model": "XGBoost", "accuracy": 1}]).to_csv(
                folder / "random_forest_xgboost_all_metrics.csv", index=False)
            frame, errors = load_results(temporary, "cic")
            self.assertEqual(frame["model"].tolist(), ["MLP"])
            self.assertEqual(frame.iloc[0]["accuracy"], 0)
            self.assertIn("80 / 20", frame.iloc[0]["_protocol"])
            self.assertEqual(len(errors), 2)

    def test_malformed_csv_and_class_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "results" / "unsw_nb15"
            folder.mkdir(parents=True)
            (folder / "baselines_metrics.csv").write_text("wrong\n1\n", encoding="utf-8")
            frame, errors = load_results(temporary, "unsw")
            self.assertTrue(frame.empty)
            self.assertTrue(any("thiếu cột" in error for error in errors))
            pd.DataFrame([{"model": "KNN", "class": "0", "precision": .8}]).to_csv(
                folder / "baselines_per_class.csv", index=False)
            report, error = per_class(folder, {"model": "KNN", "_stem": "baselines",
                                              "_meta": {"classes": ["ATTACK", "BENIGN"]}})
            self.assertFalse(error)
            self.assertEqual(report["class"].tolist(), ["ATTACK"])

    def test_eda_labels_and_feature_names(self):
        self.assertEqual(eda_label("class_distribution"), "Phân bố số lượng các lớp")
        categorical = localize_eda_table(
            pd.DataFrame([{"feature": "proto", "value": "tcp", "count": 3, "percent": .5}]),
            "categorical_feature_distribution")
        self.assertEqual(categorical.columns.tolist(),
                         ["Đặc trưng", "Giá trị", "Số lượng", "Phần trăm (%)"])
        self.assertEqual(categorical.iloc[0]["Đặc trưng"], "Giao thức")
        numeric = localize_eda_table(
            pd.DataFrame([{"label": 1, "dur_mean": 2.0}]), "numeric_summary_by_label")
        self.assertEqual(numeric.columns.tolist(), ["Nhãn", "Thời lượng TB (s)"])
        self.assertEqual(numeric.iloc[0]["Nhãn"], "1 · Tấn công")
        summary = localize_eda_table(
            pd.DataFrame([{"Label": "BENIGN", "n_flows": 1}]), "summary")
        self.assertEqual(summary.columns.tolist(), ["Nhãn", "Số luồng"])


if __name__ == "__main__":
    unittest.main()
