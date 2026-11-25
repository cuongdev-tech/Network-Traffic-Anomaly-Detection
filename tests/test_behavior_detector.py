import unittest

from types import SimpleNamespace as NS

from src.behavior_detector import (
    BehaviorWindow,
    unsw_candidate,
)


def packet(
    t,
    src="a",
    dst="b",
    port=80,
    flags=2,
):

    return NS(

        timestamp=t,

        src=src,
        dst=dst,

        dport=port,

        protocol="tcp",

        meta=NS(
            tcp_flags=flags
        ),
    )


class BehaviorTests(unittest.TestCase):

    def test_normal_and_retransmissions(self):

        w = BehaviorWindow()

        for i in range(20):

            self.assertEqual(
                w.observe(packet(i)), []
            )

        result = w.advance(60)

        self.assertEqual(

            [r["event"] for r in result],

            ["behavior_window"],
        )

        self.assertEqual(
            result[0]["alerts"], 0
        )

    def test_scan_only_at_end(self):

        w = BehaviorWindow()

        for i in range(10):

            self.assertEqual(

                w.observe(
                    packet(i, port=100+i)
                ),

                [],
            )

        self.assertEqual(
            w.advance(59), []
        )

        r = w.advance(60)

        self.assertEqual(

            [a.get("label") for a in r[1:]],

            ["PortScan"],
        )

        self.assertEqual(
            w.advance(60), []
        )

    def test_distributed_vs_single_source(self):

        for distributed in (False, True):

            w = BehaviorWindow(

                seconds=2,
                flood_pps=5,
                distributed_sources=3,
            )

            for i in range(12):

                w.observe(

                    packet(

                        0,

                        src=(
                            str(i % 3)
                            if distributed
                            else "a"
                        ),
                    )
                )

            r = w.advance(2)

            self.assertEqual(

                any(
                    x.get("label") == "DDoS"
                    for x in r
                ),

                distributed,
            )

    def test_ack_and_pair_isolation(self):

        w = BehaviorWindow(
            scan_ports=3
        )

        for i in range(3):

            w.observe(
                packet(i, port=100+i, flags=18)
            )

            w.observe(
                packet(i, dst=str(i), port=100+i)
            )

        self.assertEqual(
            w.advance(60)[0]["alerts"], 0
        )

    def test_capacity_and_late(self):

        w = BehaviorWindow(
            max_keys=1
        )

        w.observe(
            packet(0)
        )

        w.observe(
            packet(1, port=81)
        )

        self.assertTrue(
            w.advance(60)[0]["truncated"]
        )

        w.observe(
            packet(59)
        )

        self.assertEqual(
            w.late, 1
        )

    def test_unsw_advisory_gate(self):

        p = {

            "label": "Exploits",

            "confidence": 0.749,
        }

        b = {

            "ood": {
                "status": "IN_RANGE"
            },

            "models": [

                {
                    "model": "TabR",
                    "status": "OK",
                    "predictions": [p],
                }
            ],
        }

        e = {
            "dual_inference": {
                "unsw": b
            }
        }

        self.assertIsNone(
            unsw_candidate(e)
        )

        p["confidence"] = 0.95

        self.assertEqual(

            unsw_candidate(e)["event"],

            "model_candidate",
        )

        b["ood"]["status"] = "OOD"

        self.assertIsNone(
            unsw_candidate(e)
        )

        b["ood"]["status"] = "IN_RANGE"

        p["label"] = "Normal"

        self.assertIsNone(
            unsw_candidate(e)
        )


if __name__ == "__main__":

    unittest.main()
