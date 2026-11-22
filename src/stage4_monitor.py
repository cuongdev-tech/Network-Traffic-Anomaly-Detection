"""Monitor a lab interface passively, or process a saved PCAP offline."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from time import monotonic, perf_counter, time

import numpy as np
import pandas as pd
from scapy.sendrecv import AsyncSniffer
from scapy.utils import PcapReader

from src.botnet_graph import analyze_flows
from src.monitor_flows import FlowTable, packet_observation
from src.monitor_model import alert_decision, classify, load_monitor


GRAPH_WINDOW = 900
GRAPH_INTERVAL = 30
BUFFER_SIZE = 8192
STATS_SAMPLES = 10000


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")


def percentiles(numbers):
    if not numbers:
        return None
    median, p95 = np.percentile(numbers, [50, 95])
    return {"samples": len(numbers), "p50_ms": round(float(median), 3), "p95_ms": round(float(p95), 3)}


class Monitor:
    def __init__(self, bundle, log, table, *, live):
        self.bundle, self.log, self.table, self.live = bundle, log, table, live
        self.counts = {"flow_decisions": 0, "flow_alerts": 0, "graph_alerts": 0, "graph_records_evicted": 0}
        self.classification_ms = deque(maxlen=STATS_SAMPLES)
        self.live_decision_ms = deque(maxlen=STATS_SAMPLES)
        self.live_alert_ms = deque(maxlen=STATS_SAMPLES)
        self.history = deque(maxlen=20000)
        self.next_graph_scan = None
        self.graph_dirty = False

    def emit(self, event):
        self.log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.log.flush()

    def accept(self, observation):
        self.handle(self.table.ingest(observation))
        self.scan_graph()

    def tick(self, timestamp):
        self.handle(self.table.expire(timestamp))
        self.scan_graph()

    def finish(self):
        self.handle(self.table.finish())
        self.scan_graph(force=True)

    def handle(self, completed):
        if not completed:
            return
        flows = pd.DataFrame([
            {"duration_s": flow.duration_s, "packets": flow.packets, "protocol": flow.first.protocol}
            for flow, _ in completed
        ])
        started = perf_counter()
        predictions = classify(self.bundle, flows)
        self.classification_ms.append(1000 * (perf_counter() - started))
        for (flow, reason), result in zip(completed, predictions.itertuples(index=False)):
            first = flow.first
            alert = bool(alert_decision(result.label, result.gaussian_flag))
            now = time()
            delay = None
            if self.live:
                delay = round(1000 * max(0, now - flow.last), 3)
                self.live_decision_ms.append(delay)
                if alert:
                    self.live_alert_ms.append(delay)
            self.emit({
                "event": "flow_alert" if alert else "flow_decision",
                "logged_utc": utc(now), "flow_start_utc": utc(first.timestamp), "flow_end_utc": utc(flow.last),
                "src": first.src, "dst": first.dst, "sport": first.sport, "dport": first.dport,
                "protocol": first.protocol, "packets": flow.packets, "duration_s": round(flow.duration_s, 6),
                "close_reason": reason, "label": result.label, "confidence": round(result.confidence, 6),
                "gaussian_flag": bool(result.gaussian_flag), "gaussian_score": round(result.gaussian_score, 6),
                "post_last_packet_ms": delay,
            })
            self.counts["flow_decisions"] += 1
            self.counts["flow_alerts"] += int(alert)
            self.counts["graph_records_evicted"] += int(len(self.history) == self.history.maxlen)
            self.history.append((first.src, first.dst, first.timestamp))
            self.graph_dirty = True

    def scan_graph(self, *, force=False):
        if not self.graph_dirty:
            return
        now = self.table.now
        if self.next_graph_scan is None:
            self.next_graph_scan = now + GRAPH_INTERVAL
        if now < self.next_graph_scan and not force:
            return
        self.graph_dirty = False
        self.next_graph_scan = now + GRAPH_INTERVAL
        rows = [item for item in self.history if now - GRAPH_WINDOW < item[2] <= now]
        self.history = deque(rows, maxlen=self.history.maxlen)
        if not rows:
            return
        _, components = analyze_flows(pd.DataFrame(rows, columns=["src", "dst", "start"]))
        candidates = components.loc[
            components["method"].eq("periodic_topology") & components["topology"].ne("unclassified")
        ]
        for candidate in candidates.itertuples(index=False):
            self.emit({
                "event": "graph_alert", "logged_utc": utc(time()), "window_end_utc": utc(now),
                "window_seconds": GRAPH_WINDOW, "topology": candidate.topology, "hub": candidate.hub or None,
                "nodes": candidate.nodes, "edges": candidate.edges,
            })
            self.counts["graph_alerts"] += 1

    def summary(self):
        return {
            **self.counts, "flow_capacity_drops": self.table.capacity_drops,
            "late_packets": self.table.late_packets,
            "classification_batch_timing": percentiles(self.classification_ms),
            "last_packet_to_decision": percentiles(self.live_decision_ms),
            "last_packet_to_alert": percentiles(self.live_alert_ms),
        }


def read_pcap(path, monitor):
    counters = {"packets_observed": 0, "packets_supported": 0, "queue_drops": 0}
    with PcapReader(str(path)) as packets:
        for packet in packets:
            counters["packets_observed"] += 1
            item = packet_observation(packet)
            if item is not None:
                counters["packets_supported"] += 1
                monitor.accept(item)
    return counters


def capture_live(interface, capture_seconds, bpf, monitor):
    ready = Event()
    observations = Queue(maxsize=BUFFER_SIZE)
    counters = {"packets_observed": 0, "packets_supported": 0, "queue_drops": 0}

    def enqueue(packet):
        counters["packets_observed"] += 1
        item = packet_observation(packet)
        if item is not None:
            counters["packets_supported"] += 1
            try:
                observations.put_nowait(item)
            except Full:
                counters["queue_drops"] += 1

    options = {"iface": interface, "store": False, "prn": enqueue, "started_callback": ready.set}
    if bpf:
        options["filter"] = bpf
    sniffer = AsyncSniffer(**options)
    sniffer.start()
    try:
        if not ready.wait(timeout=5):
            sniffer.join(timeout=0)
            raise RuntimeError("Could not start packet capture. Check the selected interface and capture permissions.")
        print(f"Listening on {interface}", flush=True)
        deadline = monotonic() + capture_seconds
        while (remaining := deadline - monotonic()) > 0 and sniffer.running:
            try:
                item = observations.get(timeout=min(0.2, remaining))
            except Empty:
                monitor.tick(time())
            else:
                monitor.accept(item)
    except KeyboardInterrupt:
        print("Stopping capture and writing remaining flows.", flush=True)
    finally:
        if ready.is_set() and sniffer.running:
            sniffer.stop()
        else:
            sniffer.join(timeout=1)

    while not observations.empty():
        monitor.accept(observations.get_nowait())
    monitor.tick(time())
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pcap", type=Path, help="Read an existing capture without transmitting packets.")
    mode.add_argument("--interface", help="Interface selected for passive capture.")
    parser.add_argument("--seconds", type=float, default=60, help="Live capture duration (default: 60 seconds).")
    parser.add_argument("--bpf", help="Optional live BPF capture filter.")
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--active-seconds", type=float, default=60)
    parser.add_argument("--max-flows", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive.")
    if args.pcap and not args.pcap.is_file():
        parser.error(f"PCAP file not found: {args.pcap}")
    if args.pcap and args.bpf:
        parser.error("--bpf applies to live capture.")
    try:
        table = FlowTable(args.idle_seconds, args.active_seconds, args.max_flows)
        bundle = load_monitor()
        args.output.mkdir(parents=True, exist_ok=True)
        live = args.interface is not None
        start = perf_counter()
        with (args.output / "events.jsonl").open("x", encoding="utf-8") as log:
            monitor = Monitor(bundle, log, table, live=live)
            counters = (capture_live(args.interface, args.seconds, args.bpf, monitor)
                        if live else read_pcap(args.pcap, monitor))
            monitor.finish()
        result = {
            "mode": "live" if live else "pcap",
            "source": args.interface if live else args.pcap.name,
            "seconds_elapsed": round(perf_counter() - start, 3),
            "flow_idle_seconds": table.idle_seconds, "flow_active_seconds": table.active_seconds,
            **counters, **monitor.summary(),
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except FileExistsError:
        parser.exit(2, "Choose a new output directory to preserve prior event logs.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
