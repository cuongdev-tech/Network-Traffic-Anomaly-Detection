"""Optional native-Qt smoke check: python -m unittest giao_dien.test_gui."""
import os
import tempfile
import unittest
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from giao_dien.prediction import normalize_predictions

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    from giao_dien.main import Dashboard, configure_app
except ImportError as error:
    raise unittest.SkipTest("Cần cài giao_dien/requirements.txt để chạy kiểm tra Qt.") from error


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_app(cls.app)

    def test_missing_results_can_render_and_refresh_every_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            window = Dashboard(temporary)
            try:
                window.show()
                for dataset in range(window.dataset.count()):
                    window.dataset.setCurrentIndex(dataset)
                    window.refresh()
                    for page in range(window.navigation.count()):
                        window.navigation.setCurrentRow(page)
                        self.app.processEvents()
                        self.assertFalse(window.grab().isNull())
                    self.assertTrue(window.results.empty)
                    self.assertEqual(window.model_choice.count(), 0)
                    self.assertEqual(window.eda_image_choice.count(), 0)
                    self.assertTrue(window.errors.text())
            finally:
                window.close()

    def test_comparison_groups_for_both_datasets(self):
        window = Dashboard()
        traditional = {"Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"}
        try:
            for dataset in range(window.dataset.count()):
                window.dataset.setCurrentIndex(dataset)
                for index, expected in [(0, traditional | {"TabNet"}), (1, traditional), (2, {"TabNet"})]:
                    window.group.setCurrentIndex(index)
                    window.update_comparison()
                    names = {window.comparison.item(row, 0).text()
                             for row in range(window.comparison.rowCount())}
                    self.assertEqual(names, expected)
                    self.assertEqual(len(window.figure.axes[0].patches), len(expected))
        finally:
            window.close()

    def test_prediction_displays_ten_separate_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            window = Dashboard(temporary)
            try:
                classes = ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
                           "Generic", "Reconnaissance", "Shellcode", "Worms"]
                window.show_prediction(normalize_predictions(classes))
                self.assertEqual(window.prediction_table.horizontalHeaderItem(0).text(), "Lớp dự đoán")
                displayed = {window.prediction_table.item(i, 0).text() for i in range(10)}
                self.assertEqual(displayed, {"Bình thường", *classes[1:]})
            finally:
                window.close()

    def test_predict_export_and_clear_when_dataset_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "models" / "cic"
            folder.mkdir(parents=True)
            X = pd.DataFrame({"x": [0, 1, 10, 11]})
            model = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
            joblib.dump({"model": model, "features": ["x"], "labels": ["ATTACK", "BENIGN"]},
                        folder / "baseline_DecisionTree.joblib")
            source = root / "input.csv"
            X.to_csv(source, index=False)
            window = Dashboard(root)
            try:
                window.show()
                window.navigation.setCurrentRow(4)
                window.prediction_path = source
                window.clear_prediction()
                window.run_prediction.click()
                deadline = time.monotonic() + 30
                while window.prediction_worker is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertIsNone(window.prediction_worker)
                self.assertEqual(len(window.prediction_result), 4)
                self.assertTrue(window.export_prediction.isEnabled())
                output = root / "output.csv"
                with patch("giao_dien.main.QFileDialog.getSaveFileName", return_value=(str(output), "")):
                    window.save_prediction()
                self.assertEqual(pd.read_csv(output).prediction.tolist(), ["ATTACK", "ATTACK", "BENIGN", "BENIGN"])
                exported = pd.read_csv(output)
                self.assertEqual(exported.prediction_code.tolist(), [1, 1, 0, 0])
                self.assertEqual(exported.attack_type.tolist(), ["Tấn công khác", "Tấn công khác", "Không có tấn công", "Không có tấn công"])
                headers = [window.prediction_table.horizontalHeaderItem(i).text()
                           for i in range(window.prediction_table.columnCount())]
                self.assertEqual(headers[0], "Lớp dự đoán")
                self.assertNotIn("Mã nhãn", headers)
                self.assertEqual(exported.prediction_class.tolist(), ["Tấn công khác", "Tấn công khác", "Bình thường", "Bình thường"])
                window.dataset.setCurrentIndex(1)
                self.assertTrue(window.prediction_result.empty)
                self.assertFalse(window.export_prediction.isEnabled())
            finally:
                if window.prediction_worker is not None:
                    window.prediction_worker.wait(30000)
                    self.app.processEvents()
                window.close()


if __name__ == "__main__":
    unittest.main()
