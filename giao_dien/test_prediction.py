import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from giao_dien.prediction import load_bundle, normalize_predictions, predict_file


class PredictionTests(unittest.TestCase):
    def test_normalized_meanings_and_unknown_label(self):
        result = normalize_predictions(["ATTACK", "BENIGN", "DDoS", "PortScan"])
        self.assertEqual(result.prediction_code.tolist(), [1, 0, 1, 1])
        self.assertEqual(result.prediction_status.tolist(), ["Tấn công", "Bình thường", "Tấn công", "Tấn công"])
        self.assertEqual(result.attack_type.tolist(), ["Tấn công khác", "Không có tấn công", "DDoS", "PortScan"])
        with self.assertRaisesRegex(ValueError, "không được hỗ trợ"):
            normalize_predictions(["UNKNOWN"])
        classes = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                   "Normal", "Reconnaissance", "Shellcode", "Worms"]
        result = normalize_predictions(classes)
        self.assertEqual(result.prediction_code.tolist(), [1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
        self.assertEqual(result.attack_type.iloc[0], "Analysis")
        self.assertEqual(result.prediction_class.tolist(), [
            "Bình thường" if value == "Normal" else value for value in classes])
        self.assertEqual(result.prediction_class.nunique(), 10)

    def test_order_decoding_and_invalid_values(self):
        for encoded, label_key in [(True, "labels"), (False, "labels"), (True, "classes"), (False, "classes")]:
            with self.subTest(encoded=encoded), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / "models" / "cic"
                folder.mkdir(parents=True)
                X = pd.DataFrame({"a": [0, 0, 10, 10], "b": [1, 2, 1, 2]})
                y = [0, 0, 1, 1] if encoded else ["ATTACK", "ATTACK", "BENIGN", "BENIGN"]
                model = DecisionTreeClassifier(random_state=42).fit(X, y)
                joblib.dump({"model": model, "features": ["a", "b"],
                             label_key: ["ATTACK", "BENIGN"]}, folder / "baseline_DecisionTree.joblib")
                path = root / "input.csv.gz"
                pd.DataFrame({" b ": [1, 2], " a ": [0, 10],
                              "Label": ["BENIGN", "ATTACK"]}).to_csv(path, index=False)
                original = path.read_bytes()
                result = predict_file(root, "cic", "DecisionTree", path)
                self.assertEqual(result.prediction.tolist(), ["ATTACK", "BENIGN"])
                self.assertEqual(result.prediction_code.tolist(), [1, 0])
                self.assertEqual(result.attack_type.tolist(), ["Tấn công khác", "Không có tấn công"])
                self.assertEqual(result.prediction_probability.tolist(), [1., 1.])
                self.assertEqual(result.Label.tolist(), ["BENIGN", "ATTACK"])
                self.assertEqual(path.read_bytes(), original)
                for content, error in [("a\n1\n", "Thiếu"),
                                       ("a,b\ninf,1\n", "NaN/inf"),
                                       ("a,b\ntext,1\n", "không phải số")]:
                    bad = root / "bad.csv"
                    bad.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        predict_file(root, "cic", "DecisionTree", bad)

    def test_standalone_scaler_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models" / "cic"
            folder.mkdir(parents=True)
            X = pd.DataFrame({"a": [100., 200.]})
            scaler = StandardScaler().fit(X.to_numpy())
            model = DecisionTreeClassifier().fit(scaler.transform(X.to_numpy()), [0, 1])
            joblib.dump({"model": model, "scaler": scaler, "features": ["a"],
                         "classes": ["BENIGN", "DDoS"]}, folder / "baseline_LogisticRegression.joblib")
            X.to_csv(root / "input.csv", index=False)
            result = predict_file(root, "cic", "LogisticRegression", root / "input.csv")
            self.assertEqual(result.prediction.tolist(), ["BENIGN", "DDoS"])

    def test_tabnet_requires_weights_for_metadata_only_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models" / "cic"
            folder.mkdir(parents=True)
            joblib.dump({"classes": ["BENIGN", "DDoS"], "features": ["a"],
                         "scaler": StandardScaler()}, folder / "tabnet_bundle.joblib")
            with self.assertRaisesRegex(ValueError, "tabnet_attack.zip"):
                load_bundle(root, "cic", "TabNet")

    def test_unsw_shared_preprocessor_and_missing_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models" / "unsw_nb15"
            folder.mkdir(parents=True)
            X = pd.DataFrame({"a": [100., 200.], "proto": ["tcp", "udp"]})
            prep = ColumnTransformer([("num", StandardScaler(), ["a"]),
                                      ("cat", OneHotEncoder(sparse_output=False), ["proto"])])
            model = DecisionTreeClassifier().fit(prep.fit_transform(X), [0, 1])
            metadata = {"features": list(X.columns), "classes": ["Analysis", "Normal"]}
            joblib.dump({"model": model, **metadata}, folder / "random_forest_attack.joblib")
            X.to_csv(root / "input.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Thiếu bộ tiền xử lý"):
                predict_file(root, "unsw", "Random Forest", root / "input.csv")
            joblib.dump({"preprocessor": prep, **metadata}, folder / "preprocessor_bundle.joblib")
            result = predict_file(root, "unsw", "Random Forest", root / "input.csv")
            self.assertEqual(result.prediction.tolist(), ["Analysis", "Normal"])
            self.assertEqual(result.prediction_code.tolist(), [1, 0])

    def test_rejects_pcap_empty_and_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "PCAP"):
                predict_file(root, "cic", "MLP", root / "capture.pcap")
            path = root / "input.csv"
            for content, error in [("a, a\n1,2\n", "trùng"), ("a,b\n", "không có dòng")]:
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    predict_file(root, "cic", "MLP", path)


if __name__ == "__main__":
    unittest.main()
