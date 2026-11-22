from io import StringIO
import json
import unittest
from unittest.mock import patch

import pandas as pd
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import ARP

from src.evaluate_botnet_graph import synthetic_flows
from src.monitor_flows import FlowTable, Observation, packet_observation
from src.stage4_monitor import Monitor


def observation(t, reply=False):
    origin, target = ("192.0.2.1", 40250), ("192.0.2.2", 8080)
    src, dst = (target, origin) if reply else (origin, target)
    return Observation(t, src[0], dst[0], src[1], dst[1], "tcp")


class FlowTableTests(unittest.TestCase):
    def test_bidirectional_flow_uses_initiator_and_idle_deadline(self):
        table = FlowTable(idle_seconds=2, active_seconds=5)
        self.assertEqual(table.ingest(observation(10)), [])
        self.assertEqual(table.ingest(observation(11, reply=True)), [])
        self.assertEqual(table.expire(12), [])
        [(flow, reason)] = table.expire(13)
        self.assertEqual((flow.first.src, flow.packets, flow.duration_s, reason),
                         ("192.0.2.1", 2, 1, "idle"))

    def test_active_limit_and_capacity(self):
        table = FlowTable(idle_seconds=3, active_seconds=4, max_flows=1)
        table.ingest(observation(10))
        table.ingest(Observation(11, "192.0.2.4", "192.0.2.5", 10, 80, "tcp"))
        table.ingest(observation(12, reply=True))
        [(flow, reason)] = table.ingest(observation(14))
        self.assertEqual((flow.packets, reason, table.capacity_drops), (2, "active", 1))
        self.assertEqual(table.finish()[0][0].packets, 1)

    def test_supported_transport_and_ignored_fragments(self):
        packet = IP(src="192.0.2.1", dst="192.0.2.2") / TCP(sport=80, dport=1000)
        self.assertEqual(packet_observation(packet).dport, 1000)
        packet[IP].flags = "MF"
        self.assertIsNone(packet_observation(packet))
        self.assertIsNone(packet_observation(ARP()))


class MonitorTests(unittest.TestCase):
    def test_offline_aggregation_writes_flow_alert_without_live_timing(self):
        results = pd.DataFrame({
            "label": ["normal"], "confidence": [0.9],
            "gaussian_score": [3.0], "gaussian_flag": [True],
        })
        log = StringIO()
        monitor = Monitor(None, log, FlowTable(), live=False)
        with patch("src.stage4_monitor.classify", return_value=results):
            monitor.accept(observation(10))
            monitor.finish()
        event = json.loads(log.getvalue())
        self.assertEqual(event["event"], "flow_alert")
        self.assertIsNone(event["post_last_packet_ms"])
        self.assertEqual(monitor.summary()["flow_alerts"], 1)

    def test_sliding_graph_uses_recent_periodic_endpoints(self):
        flows, _ = synthetic_flows()
        star = flows.loc[flows["file"].eq("centralized_clean"), ["src", "dst", "start"]]
        log = StringIO()
        table = FlowTable()
        table.expire(260)
        monitor = Monitor(None, log, table, live=False)
        monitor.history.extend(star.itertuples(index=False, name=None))
        monitor.graph_dirty = True
        monitor.scan_graph(force=True)
        event = json.loads(log.getvalue())
        self.assertEqual((event["event"], event["topology"], event["hub"]),
                         ("graph_alert", "centralized", "hub"))
        table.expire(2000)
        monitor.graph_dirty = True
        monitor.scan_graph(force=True)
        self.assertEqual(len(log.getvalue().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
