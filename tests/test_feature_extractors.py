
import csv
import unittest

from copy import deepcopy
from pathlib import Path

from src.cic_feature_extractor import (
    CIC_FEATURES,
    extract_cic_features,
    validate_cic_row,
)

from src.unsw_feature_extractor import (
    UNSW_FEATURES,
    extract_unsw_features,
    validate_unsw_row,
)


ROOT = Path(__file__).resolve().parents[1]


def moments(
    n,
    total=0,
    minimum=None,
    maximum=None,
    mean=None,
    std=None,
):
    return dict(
        count=n,
        total=total,
        min=minimum,
        max=maximum,
        mean=mean,
        std=std,
    )


def direction(packets, payload, window):

    return {

        "packets": packets,

        "payload_length": moments(
            packets,
            payload * packets,
            payload,
            payload,
            payload,
            0,
        ),

        "transport_header": moments(
            packets,
            20 * packets,
            20, 20, 20, 0,
        ),

        "ip_length": moments(
            packets,
            (40 + payload) * packets,
            40 + payload,
            40 + payload,
            40 + payload,
            0,
        ),

        "iat_seconds": moments(
            packets - 1,
            0.5 * (packets - 1),
            0.5, 0.5, 0.5, 0,
        ),

        "tcp_flags": {
            "ACK": packets
        },

        "initial_window": window,
    }


def event_fixture():

    return {

        "event": "flow_decision",
        "protocol": "tcp",

        "src": "192.0.2.1",
        "dst": "192.0.2.2",

        "sport": 40000,
        "dport": 8765,

        "duration_s": 1.0,
        "packets": 3,

        "capture_features": {

            "forward": direction(
                2, 10, 4096
            ),

            "backward": direction(
                1, 20, 8192
            ),

            "flow_iat_seconds": moments(
                2, 1, 0.5, 0.5, 0.5, 0
            ),

            "missing_metadata": 0,
            "out_of_order": 0,
        },
    }


class FeatureExtractorTests(unittest.TestCase):

    def test_schema_and_sample_rows(self):

        for ds, schema, validator, size in (

            (
                "cic",
                CIC_FEATURES,
                validate_cic_row,
                77,
            ),

            (
                "unsw",
                UNSW_FEATURES,
                validate_unsw_row,
                34,
            ),
        ):

            path = (
                ROOT / "giao_dien"
                / f"mau_du_doan_{ds}.csv"
            )

            with path.open(
                encoding="utf-8-sig",
                newline="",
            ) as file:

                reader = csv.DictReader(file)

                self.assertEqual(
                    tuple(reader.fieldnames),
                    schema,
                )

                self.assertEqual(
                    len(schema), size
                )

                row = next(reader)

            self.assertEqual(
                tuple(validator(row)),
                schema,
            )

            row.pop(schema[0])

            with self.assertRaises(ValueError):
                validator(row)


    def test_cic_units_and_missing(self):

        r = extract_cic_features(
            event_fixture()
        )

        v = r["candidate_features"]

        self.assertEqual(
            r["status"], "UNVERIFIED"
        )

        self.assertEqual(
            v["Flow Duration"], 1_000_000
        )

        self.assertEqual(
            v["Flow IAT Mean"], 500_000
        )

        self.assertEqual(
            v["Flow Bytes/s"], 40
        )

        self.assertEqual(
            v["ACK Flag Count"], 3
        )

        self.assertIn(
            "Active Mean",
            r["missing_features"],
        )

        self.assertIsNone(
            r["model_features"]
        )

        self.assertEqual(
            r["available_count"], 51
        )

        self.assertEqual(
            len(v) + len(r["missing_features"]),
            77,
        )


    def test_unsw_units_and_missing(self):

        r = extract_unsw_features(
            event_fixture()
        )

        v = r["candidate_features"]

        self.assertEqual(
            v["sinpkt"], 500
        )

        self.assertEqual(
            v["spkts"], 2
        )

        self.assertEqual(
            v["dpkts"], 1
        )

        self.assertIn(
            "service",
            r["missing_features"],
        )

        self.assertNotIn("sloss", v)

        self.assertIsNone(
            r["model_features"]
        )

        self.assertEqual(
            r["available_count"], 7
        )

        self.assertEqual(
            len(v) + len(r["missing_features"]),
            34,
        )


    def test_invalid_metadata_and_counts(self):

        original = event_fixture()

        for change in (
            "missing_metadata",
            "out_of_order",
            "packet_count",
            "icmp",
        ):

            event = deepcopy(original)

            if change == "packet_count":
                event["packets"] += 1

            elif change == "icmp":
                event["protocol"] = "icmp"

            else:
                event["capture_features"][
                    change
                ] = 1

            for extractor in (
                extract_cic_features,
                extract_unsw_features,
            ):

                r = extractor(event)

                self.assertEqual(
                    r["status"],
                    "INVALID_FLOW",
                )

                self.assertIsNone(
                    r["model_features"]
                )


    def test_zero_duration_and_invalid_reference(self):

        event = event_fixture()

        event["duration_s"] = 0

        stats = event["capture_features"]

        stats["flow_iat_seconds"] = moments(
            2, 0, 0, 0, 0, 0
        )

        stats["forward"]["iat_seconds"] = moments(
            1, 0, 0, 0, 0, 0
        )

        r = extract_cic_features(event)

        self.assertIn(
            "Flow Bytes/s",
            r["missing_features"],
        )

        row = dict.fromkeys(
            CIC_FEATURES, 1
        )

        row["Flow Duration"] = float("inf")

        with self.assertRaises(ValueError):
            validate_cic_row(row)


if __name__ == "__main__":
    unittest.main()
