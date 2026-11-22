import unittest

import numpy as np
import pandas as pd

from src.botnet_graph import analyze_flows
from src.evaluate_botnet_graph import evaluate_cases, synthetic_flows
from src.evaluate_ctu13_graph import attach_truth
from src.gaussian_anomaly import GaussianAnomalyDetector


class GaussianTests(unittest.TestCase):
    def test_scores_constant_features_and_preserves_input(self):
        normal = pd.DataFrame({"constant": [0.0] * 4, "varied": [1.0, 2.0, 3.0, 4.0]})
        input_flows = pd.DataFrame({"constant": [0.0, 100.0, np.inf], "varied": [2.0] * 3})
        scores = GaussianAnomalyDetector().fit(normal).score_samples(input_flows)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[2], scores[0])
        self.assertTrue(np.isinf(input_flows.iloc[2, 0]))


class BotnetGraphTests(unittest.TestCase):
    def test_duplicate_starts_and_separate_captures_do_not_create_beacon(self):
        data = pd.DataFrame({
            "file": ["a"] * 4 + ["b"] * 3,
            "src": ["client"] * 7,
            "dst": ["server"] * 7,
            "start": [0, 10, 20, 20, 30, 40, 50],
        })
        edges, _ = analyze_flows(data)
        self.assertEqual(edges["connections"].tolist(), [3, 3])
        self.assertFalse(edges["periodic"].any())

    def test_connections_in_different_time_windows_are_separate(self):
        data = pd.DataFrame({"src": "client", "dst": "server", "start": [0, 5, 10, 15, 20, 25]})
        edges, _ = analyze_flows(data, window_seconds=15)
        self.assertEqual(edges["window"].tolist(), [0, 1])
        self.assertFalse(edges["periodic"].any())

    def test_topology_and_known_false_positive(self):
        flows, truth = synthetic_flows()
        cases, summary = evaluate_cases(*analyze_flows(flows), truth)
        combined = cases.loc[cases["method"].eq("periodic_topology")].set_index("capture")
        self.assertEqual(combined.loc["centralized_clean", "prediction"], "centralized")
        self.assertEqual(combined.loc["p2p_ring", "prediction"], "p2p")
        self.assertEqual(combined.loc["normal_periodic_service", "prediction"], "centralized")
        self.assertEqual(summary.set_index("method").loc["periodic_topology", "false_positives"], 1)

    def test_ctu_truth_uses_labels_only_after_detection(self):
        data = pd.DataFrame({
            "file": ["CTU-11"] * 5,
            "src": ["x", "x", "y", "z", "z"],
            "dst": ["q"] * 5,
            "start": [0, 10, 20, 30, 40],
            "label": ["normal", "botnet", "normal", "background", "background"],
        })
        edges, _ = analyze_flows(data)
        scored = attach_truth(data, edges).set_index("src")
        self.assertEqual(scored["truth"].to_dict(), {"x": "botnet", "y": "normal", "z": "unknown"})
        original = scored["periodic"].copy()
        data["label"] = "botnet"
        recomputed = attach_truth(data, analyze_flows(data)[0]).set_index("src")
        pd.testing.assert_series_equal(recomputed["periodic"], original)

    def test_empty_and_invalid_input(self):
        edges, components = analyze_flows(pd.DataFrame(columns=["src", "dst", "start"]))
        self.assertTrue(edges.empty and components.empty)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            analyze_flows(pd.DataFrame({"src": ["a"], "dst": ["b"], "start": [np.nan]}))


if __name__ == "__main__":
    unittest.main()
