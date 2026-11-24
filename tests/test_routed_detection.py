import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock

import pandas as pd

from src.cic_feature_extractor import CIC_FEATURES
from src.unsw_feature_extractor import UNSW_FEATURES
from src.dual_inference import DualInferenceEngine, MODELS
from src.unified_detection import UnifiedDetectionEngine, aggregate
from test_feature_extractors import event_fixture


def branch(labels):
    return {
        "status": "OK",
        "models": [
            {
                "model": model,
                "status": "OK",
                "predictions": [{"label": label}],
            }
            for model, label in zip(MODELS, labels)
        ],
    }


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
        "extractor": "test-fixture",
        "features": row,
    }


class DualBranchTests(unittest.TestCase):

    def test_missing_provider_blocks_all_models(self):

        with TemporaryDirectory() as root:
            engine = UnifiedDetectionEngine(root=root)

            with patch.object(
                engine.engine, "infer_model"
            ) as infer:

                result = engine.detect(fixture())

            infer.assert_not_called()

        self.assertEqual(
            result["mode"], "dual_independent"
        )

        self.assertEqual(
            result["route"], "both"
        )

        self.assertEqual(
            result["decision"]["status"],
            "UNAVAILABLE",
        )

        self.assertEqual(
            result["decision"]["completed_models"],
            0,
        )


    def test_same_flow_runs_all_twelve_models(self):

        with TemporaryDirectory() as root:

            provider = Mock(side_effect=reference)

            engine = UnifiedDetectionEngine(
                root=root,
                provider=provider,
            )

            def predict(dataset, model, frame):

                expected = (
                    CIC_FEATURES
                    if dataset == "cic"
                    else UNSW_FEATURES
                )

                self.assertEqual(
                    tuple(frame.columns),
                    expected,
                )

                return {
                    "model": model,
                    "status": "OK",
                    "predictions": [
                        {"label": "Normal"}
                    ],
                }

            with patch.object(
                engine.engine,
                "infer_model",
                side_effect=predict,
            ) as infer:

                result = engine.detect(fixture())

            self.assertEqual(
                infer.call_count, 12
            )

            self.assertEqual(
                {
                    (c.args[0], c.args[1])
                    for c in infer.call_args_list
                },
                {
                    (ds, m)
                    for ds in ("cic", "unsw")
                    for m in MODELS
                },
            )

            self.assertEqual(
                [
                    c.args[1]
                    for c in provider.call_args_list
                ],
                ["cic", "unsw"],
            )

            self.assertEqual(
                {
                    c.args[0]["flow_id"]
                    for c in provider.call_args_list
                },
                {"flow-001"},
            )

        self.assertEqual(
            result["decision"]["completed_models"],
            12,
        )

        self.assertEqual(
            result["decision"]["status"],
            "NORMAL_CANDIDATE",
        )


    def test_invalid_one_branch_does_not_block_other(self):

        def provider(event, dataset):

            if dataset == "unsw":
                raise ValueError("missing reference")

            return reference(event, dataset)

        with TemporaryDirectory() as root:

            engine = UnifiedDetectionEngine(
                root=root,
                provider=provider,
            )

            with patch.object(
                engine.engine,
                "infer_branch",
                side_effect=lambda ds, frame:
                    branch(["Normal"] * 6)
                    if frame is not None
                    else {
                        "status": "FEATURES_UNAVAILABLE",
                        "models": [],
                    },
            ):

                result = engine.detect(fixture())

        self.assertEqual(
            result["feature_audit"]["unsw"]["status"],
            "PROVIDER_ERROR",
        )

        self.assertEqual(
            result["decision"]["completed_models"],
            6,
        )

        self.assertEqual(
            result["decision"]["status"],
            "INCOMPLETE",
        )


    def test_conflicting_labels_remain_separate(self):

        dual = {
            "cic": branch(["DDoS"] * 6),
            "unsw": branch(["DoS"] * 6),
        }

        result = aggregate(dual)

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


    def test_disagreement_inside_branch(self):

        dual = {
            "cic": branch(
                ["Normal"] * 3 + ["DDoS"] * 3
            ),
            "unsw": branch(["Normal"] * 6),
        }

        self.assertEqual(
            aggregate(dual)["status"],
            "DISAGREEMENT",
        )


    def test_checkpoint_version_guard(self):

        engine = DualInferenceEngine(
            strict_versions=True
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
                MODELS[0],
                pd.DataFrame({"f": [1]}),
            )

        self.assertEqual(
            result["status"],
            "VERSION_MISMATCH",
        )

        self.assertNotIn(
            "predictions", result
        )


if __name__ == "__main__":
    unittest.main()
