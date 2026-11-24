
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.dual_inference import (
    BRANCHES,
    DualInferenceEngine,
    checked_frame,
    predict_bundle,
)


class DualInferenceTests(unittest.TestCase):

    def test_missing_and_nonfinite_features(self):
        frames = (
            pd.DataFrame({"a": [1]}),
            pd.DataFrame({"a": [1], "b": [np.inf]}),
        )

        for frame in frames:
            with self.assertRaises(ValueError):
                checked_frame(
                    frame, ["a", "b"], "cic"
                )

    def test_preprocessor_class_mapping_and_column_order(self):
        X = np.array([
            [1, 100],
            [2, 200],
            [3, 300],
        ])

        scaler = StandardScaler().fit(X)

        model = DecisionTreeClassifier(
            random_state=42
        ).fit(
            scaler.transform(X),
            [0, 1, 2],
        )

        bundle = {
            "model": model,
            "scaler": scaler,
            "features": ["a", "b"],
            "classes": ["BENIGN", "DDoS", "PortScan"],
            "labels": ["BENIGN", "DDoS", "PortScan"],
        }

        out = predict_bundle(
            bundle,
            "cic",
            "DecisionTree",
            pd.DataFrame({
                "b": [100, 300],
                "a": [1, 3],
            }),
        )

        self.assertEqual(
            [r["label"] for r in out],
            ["Normal", "PortScan"],
        )

        self.assertEqual(
            [r["raw_label"] for r in out],
            ["BENIGN", "PortScan"],
        )

    def test_unsw_categorical_missing(self):
        with self.assertRaises(ValueError):
            checked_frame(
                pd.DataFrame({"proto": [None]}),
                ["proto"],
                "unsw",
            )

    def test_unavailable_and_failed_branches_are_not_normal(self):
        engine = DualInferenceEngine(
            models=["DecisionTree"]
        )

        with patch(
            "src.dual_inference.load_bundle",
            side_effect=FileNotFoundError("test"),
        ):
            result = engine.predict_flow(
                "flow-test",
                cic_features={"a": 1},
            )

        self.assertEqual(
            result["cic"]["status"],
            "ERROR",
        )

        self.assertEqual(
            result["unsw"]["status"],
            "FEATURES_UNAVAILABLE",
        )

        self.assertNotIn(
            "predictions",
            result["cic"]["models"][0],
        )

    def test_cache_and_validate_schema(self):
        engine = DualInferenceEngine(
            models=["DecisionTree"]
        )

        count, classes, _ = BRANCHES["cic"]

        dummy = {
            "features": [
                f"f{i}" for i in range(count)
            ],
            "labels": sorted(classes),
            "model": object(),
        }

        with patch(
            "src.dual_inference.load_bundle",
            return_value=dummy,
        ) as loader:

            self.assertIs(
                engine.get_bundle(
                    "cic", "DecisionTree"
                ),
                engine.get_bundle(
                    "cic", "DecisionTree"
                ),
            )

            self.assertEqual(loader.call_count, 1)

        engine.release("cic", "DecisionTree")

        with patch(
            "src.dual_inference.load_bundle",
            return_value={
                **dummy,
                "features": ["wrong"],
            },
        ):
            with self.assertRaises(ValueError):
                engine.get_bundle(
                    "cic", "DecisionTree"
                )


if __name__ == "__main__":
    unittest.main()
