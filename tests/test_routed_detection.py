import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock

import pandas as pd

from src.cic_feature_extractor import CIC_FEATURES
from src.unsw_feature_extractor import UNSW_FEATURES
from src.dual_inference import DualInferenceEngine
from src.unified_detection import UnifiedDetectionEngine, aggregate
from src.host_window import HostWindow
from test_feature_extractors import event_fixture


def fixture():

    event = event_fixture()
    event["flow_id"] = "flow-001"

    return event


def reference(event, dataset):

    # Synthetic rows for interface tests only.

    schema = (
        CIC_FEATURES
        if dataset == "cic"
        else UNSW_FEATURES
    )

    row = dict.fromkeys(schema, 1)

    if dataset == "unsw":

        row.update(
            proto="tcp",
            service="http",
            state="FIN",
        )

    return {
        "flow_id": event["flow_id"],
        "dataset": dataset,
        "extractor": "fixture-only",
        "features": row,
    }


def branch(label):

    return {
        "status": "OK",
        "models": [
            {
                "model": "TabR",
                "status": "OK",
                "predictions": [
                    {
                        "label": label,
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }


class TabRTests(unittest.TestCase):

    def test_missing_provider(self):

        with TemporaryDirectory() as root:

            engine = UnifiedDetectionEngine(
                root=root
            )

            with patch.object(
                engine.engine, "infer_model"
            ) as infer:

                result = engine.detect(fixture())

            infer.assert_not_called()

        self.assertEqual(
            result["mode"], "tabr_dual"
        )

        self.assertEqual(
            result["decision"]["status"],
            "UNAVAILABLE",
        )

        self.assertEqual(
            result["decision"]["completed_models"],
            0,
        )

    def test_same_flow_exactly_two_tabr_calls(self):

        provider = Mock(side_effect=reference)

        with TemporaryDirectory() as root:

            engine = UnifiedDetectionEngine(
                root=root,
                provider=provider,
            )

            def predict(dataset, model, frame):

                self.assertEqual(
                    model, "TabR"
                )

                self.assertEqual(
                    tuple(frame.columns),
                    CIC_FEATURES
                    if dataset == "cic"
                    else UNSW_FEATURES,
                )

                return branch("Normal")["models"][0]

            with patch.object(
                engine.engine,
                "infer_model",
                side_effect=predict,
            ) as infer:

                result = engine.detect(fixture())

            self.assertEqual(
                infer.call_count, 2
            )

            self.assertEqual(
                {
                    (c.args[0], c.args[1])
                    for c in infer.call_args_list
                },
                {
                    ("cic", "TabR"),
                    ("unsw", "TabR"),
                },
            )

            self.assertEqual(
                {
                    c.args[0]["flow_id"]
                    for c in provider.call_args_list
                },
                {"flow-001"},
            )

        self.assertEqual(
            result["decision"]["status"],
            "NORMAL_CANDIDATE",
        )

        self.assertEqual(
            result["decision"]["completed_models"],
            2,
        )

    def test_partial_attack(self):

        result = aggregate({
            "cic": branch("PortScan"),
            "unsw": {
                "status": "FEATURES_UNAVAILABLE"
            },
        })

        self.assertEqual(
            result["status"], "INCOMPLETE"
        )

        self.assertTrue(
            result["attack_candidate"]
        )

        self.assertEqual(
            result["completed_models"], 1
        )

    def test_conflict_not_fused(self):

        result = aggregate({
            "cic": branch("DDoS"),
            "unsw": branch("DoS"),
        })

        self.assertEqual(
            result["status"], "CONFLICT"
        )

        self.assertEqual(
            result["candidate_labels"],
            ["DDoS", "DoS"],
        )

        self.assertTrue(
            result["attack_candidate"]
        )

    def test_partial_normal_is_not_clear(self):

        result = aggregate({
            "cic": branch("Normal")
        })

        self.assertEqual(
            result["status"], "INCOMPLETE"
        )

        self.assertFalse(
            result["attack_candidate"]
        )

    def test_checkpoint_version_guard(self):

        engine = DualInferenceEngine(
            models=("TabR",),
            strict_versions=True,
        )

        with patch.object(
            engine,
            "get_bundle",
            return_value={
                "_version_warning": "mismatch"
            },
        ):

            result = engine.infer_model(
                "cic",
                "TabR",
                pd.DataFrame({"f": [1]}),
            )

        self.assertEqual(
            result["status"],
            "VERSION_MISMATCH",
        )

    def test_monitor_tabr_uses_gaussian_without_rf(self):

        from io import StringIO
        import json
        import numpy as np

        try:

            from src.stage4_monitor import Monitor

            from src.monitor_flows import (
                FlowTable, Observation
            )

        except ModuleNotFoundError as error:

            if error.name and error.name.startswith("scapy"):

                self.skipTest(
                    "Scapy unavailable"
                )

            raise

        gaussian = Mock()

        gaussian.score_samples.return_value = (
            np.array([0.5])
        )

        detector = Mock()

        detector.detect.return_value = {
            "decision": {
                "attack_candidate": True,
                "completed_models": 2,
            }
        }

        bundle = {
            "gaussian": gaussian,
            "gaussian_threshold": 1.0,
        }

        log = StringIO()

        monitor = Monitor(
            bundle,
            log,
            FlowTable(),
            live=False,
            detector=detector,
        )

        with patch(
            "src.stage4_monitor.classify"
        ) as legacy:

            monitor.accept(
                Observation(
                    10,
                    "192.0.2.1",
                    "192.0.2.2",
                    40000,
                    80,
                    "tcp",
                )
            )

            monitor.finish()

            legacy.assert_not_called()

        event = json.loads(
            log.getvalue().splitlines()[0]
        )

        self.assertEqual(
            event["event"], "flow_alert"
        )

        self.assertEqual(
            event["classification_mode"],
            "tabr_dual",
        )

        self.assertFalse(
            event["ctu_alert"]
        )

        self.assertEqual(
            event["host_window"]["flows"], 1
        )

        self.assertEqual(
            monitor.summary()["dual_completed"],
            1,
        )

    def test_host_window_expiry_and_cap(self):

        w = HostWindow(
            seconds=60,
            limit=2,
        )

        self.assertEqual(
            w.add(0, "a", "b", 80, 3)["flows"],
            1,
        )

        self.assertEqual(
            w.add(1, "a", "c", 443, 4)[
                "unique_dst_ports"
            ],
            2,
        )

        v = w.add(2, "a", "d", 443, 5)

        self.assertEqual(
            v["packets"], 9
        )

        self.assertTrue(
            v["capacity_truncated"]
        )

        v = w.add(62, "a", "b", 80, 1)

        self.assertEqual(
            v["flows"], 1
        )

        self.assertFalse(
            v["capacity_truncated"]
        )


if __name__ == "__main__":
    unittest.main()
