import csv
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_flow_stats import (
    FlowStats, PacketMeta, read_metadata
)

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


def make_observation(t, meta, backward=False, protocol="tcp"):

    src, dst = "192.0.2.1", "192.0.2.2"
    sport, dport = 40000, 8765

    if backward:

        src, dst, sport, dport = (
            dst, src, dport, sport
        )

    return SimpleNamespace(
        timestamp=t,
        src=src,
        dst=dst,
        sport=sport,
        dport=dport,
        protocol=protocol,
        meta=meta,
    )


def tcp(flags, seq, ack=0, size=0):

    return PacketMeta(
        40 + size,
        20,
        20,
        size,
        flags,
        4096,
        seq,
        ack,
    )


def make_event(observations):

    first = observations[0]
    stats = FlowStats()

    for obs in observations:

        stats.add(obs, first)

    stats.history = dict(
        scope="local_last_100_closed_flows",
        observed_connections=1,
        ct_src_dport_ltm=1,
        ct_dst_sport_ltm=1,
    )

    return {
        "event": "flow_decision",
        "flow_id": "fixture",
        "protocol": first.protocol,
        "src": first.src,
        "dst": first.dst,
        "sport": first.sport,
        "dport": first.dport,
        "packets": len(observations),

        "duration_s": (
            max(o.timestamp for o in observations)
            - first.timestamp
        ),

        "capture_features": stats.snapshot(),
    }


class ExtendedFeatureTests(unittest.TestCase):

    def test_training_csv_schema(self):

        for dataset, schema, validator, count in (

            ("cic", CIC_FEATURES, validate_cic_row, 77),

            ("unsw", UNSW_FEATURES, validate_unsw_row, 34),

        ):

            path = (
                ROOT
                / "giao_dien"
                / f"mau_du_doan_{dataset}.csv"
            )

            with path.open(
                encoding="utf-8-sig",
                newline="",
            ) as f:

                reader = csv.DictReader(f)

                self.assertEqual(
                    tuple(reader.fieldnames), schema
                )

                self.assertEqual(
                    len(validator(next(reader))), count
                )

    def test_raw_packet_lengths_and_truncation(self):

        def ipv4(proto, transport):

            return struct.pack(
                "!BBHHHBBH4s4s",
                0x45,
                0,
                20 + len(transport),
                0,
                0,
                64,
                proto,
                0,
                b"\xc0\x00\x02\x01",
                b"\xc0\x00\x02\x02",
            ) + transport

        segment = struct.pack(
            "!HHIIBBHHH",
            40000,
            8765,
            10,
            20,
            0x50,
            0x18,
            4096,
            0,
            0,
        ) + b"abc"

        raw = ipv4(6, segment)

        self.assertEqual(
            read_metadata(raw),
            tcp(0x18, 10, 20, 3),
        )

        self.assertIsNone(
            read_metadata(raw[:-1])
        )

        udp = ipv4(
            17,
            struct.pack(
                "!HHHH",
                40000,
                8765,
                11,
                0,
            ) + b"abc",
        )

        self.assertEqual(
            read_metadata(udp),
            PacketMeta(31, 20, 8, 3),
        )

        self.assertIsNone(
            read_metadata(udp[:-1])
        )

    def test_tcp_handshake_bulk_and_cic_candidates(self):

        obs = [
            make_observation(
                10.0, tcp(2, 100)
            ),

            make_observation(
                10.05, tcp(18, 200, 101), True
            ),

            make_observation(
                10.10, tcp(16, 101, 201)
            ),
        ]

        obs += [
            make_observation(
                10.2 + i * 0.1,
                tcp(24, 101 + i * 10, 201, 10),
            )
            for i in range(4)
        ]

        event = make_event(obs)

        c = extract_cic_features(event)
        u = extract_unsw_features(event)

        self.assertEqual(
            c["available_count"], 77
        )

        self.assertEqual(
            c["candidate_features"]["Fwd Avg Bytes/Bulk"],
            40,
        )

        self.assertEqual(
            c["candidate_features"]["Fwd Avg Packets/Bulk"],
            4,
        )

        self.assertEqual(
            c["candidate_features"]["act_data_pkt_fwd"],
            4,
        )

        self.assertEqual(
            c["candidate_features"]["SYN Flag Count"],
            1,
        )

        self.assertEqual(
            u["available_count"], 21
        )

        self.assertAlmostEqual(
            u["candidate_features"]["tcprtt"],
            0.10,
        )

        self.assertEqual(
            u["candidate_features"]["stcpb"],
            100,
        )

        self.assertEqual(
            u["candidate_features"]["dtcpb"],
            200,
        )

        self.assertEqual(
            u["candidate_features"]["sbytes"],
            280,
        )

        self.assertIn(
            "service", u["missing_features"]
        )

        self.assertIsNone(
            c["model_features"]
        )

        self.assertIsNone(
            u["model_features"]
        )

    def test_udp_legacy_statistics(self):

        event = make_event([

            make_observation(
                0,
                PacketMeta(99, 20, 8, 71),
                protocol="udp",
            ),

            make_observation(
                0.05,
                PacketMeta(183, 20, 8, 155),
                True,
                "udp",
            ),

        ])

        c = extract_cic_features(event)
        u = extract_unsw_features(event)

        self.assertEqual(
            c["available_count"], 77
        )

        self.assertEqual(
            c["candidate_features"]["Init_Win_bytes_forward"],
            -1,
        )

        self.assertEqual(
            c["candidate_features"]["Packet Length Mean"],
            99,
        )

        self.assertEqual(
            c["candidate_features"]["Average Packet Size"],
            148.5,
        )

        self.assertEqual(
            u["available_count"], 21
        )

        self.assertEqual(
            u["candidate_features"]["swin"],
            0,
        )

    def test_invalid_flow_and_zero_duration(self):

        for obs in (

            [
                make_observation(1, None)
            ],

            [
                make_observation(1, tcp(2, 1)),
                make_observation(0.9, tcp(16, 2)),
            ],

        ):

            event = make_event(obs)

            self.assertEqual(
                extract_cic_features(event)["status"],
                "INVALID_FLOW",
            )

            self.assertEqual(
                extract_unsw_features(event)["status"],
                "INVALID_FLOW",
            )

        event = make_event([
            make_observation(1, tcp(2, 1))
        ])

        self.assertIn(
            "Flow Bytes/s",
            extract_cic_features(event)["missing_features"],
        )

        self.assertIn(
            "rate",
            extract_unsw_features(event)["missing_features"],
        )

    def test_ambiguous_handshake_not_filled(self):

        obs = [

            make_observation(
                0, tcp(2, 100)
            ),

            make_observation(
                0.1, tcp(2, 100)
            ),

            make_observation(
                0.2, tcp(18, 200, 101), True
            ),

            make_observation(
                0.3, tcp(16, 101, 201)
            ),

        ]

        event = make_event(obs)

        self.assertIsNone(
            event["capture_features"]["handshake"]
        )

        self.assertIn(
            "tcprtt",
            extract_unsw_features(event)["missing_features"],
        )

    def test_flow_table_history(self):

        try:

            from src.monitor_flows import FlowTable

        except ModuleNotFoundError as error:

            if error.name and error.name.startswith("scapy"):

                self.skipTest(
                    "Scapy not installed in this environment"
                )

            raise

        table = FlowTable(
            idle_seconds=1,
            active_seconds=10,
        )

        for t in range(3):

            table.ingest(
                make_observation(t * 2, tcp(2, 100))
            )

            [(flow, reason)] = table.expire(t * 2 + 1)

            self.assertEqual(
                reason, "idle"
            )

            self.assertEqual(
                flow.stats.snapshot()["history"]["ct_src_dport_ltm"],
                t + 1,
            )

        self.assertLessEqual(
            len(table.recent), 99
        )


if __name__ == "__main__":
    unittest.main()
