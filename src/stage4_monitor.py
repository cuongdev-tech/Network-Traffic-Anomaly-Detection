
"""Monitor a lab interface passively, or process a saved PCAP offline."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import shutil
import signal
import subprocess
import tempfile
from uuid import uuid4
from math import isfinite
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic, perf_counter, time

import numpy as np
import pandas as pd
from scapy.sendrecv import AsyncSniffer
from scapy.utils import PcapReader

from src.botnet_graph import analyze_flows
from src.monitor_flows import FlowTable, packet_observation
from src.monitor_model import alert_decision, classify, load_monitor, features
from src.host_window import HostWindow
from src.realtime_detection import (
    RealtimeBehaviorWindow as BehaviorWindow,
    netflow_candidate as unsw_candidate,
)
from src.realtime_detection import RealtimeDetectionEngine as UnifiedDetectionEngine
from src.unified_detection import load_provider


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
    def __init__(self, bundle, log, table, *, live, detector=None):
        self.bundle, self.log, self.table, self.live = bundle, log, table, live
        self.detector = detector
        self.host_window = HostWindow()
        self.behavior = BehaviorWindow() if detector is not None else None
        self.counts = {"flow_decisions": 0, "flow_alerts": 0, "graph_alerts": 0, "graph_records_evicted": 0}
        self.counts.update(ctu_alerts=0, gaussian_alerts=0, dual_alerts=0, dual_completed=0)
        self.counts.update(behavior_alerts=0, behavior_windows=0, behavior_truncated=0, unsw_advisories=0)
        self.classification_ms = deque(maxlen=STATS_SAMPLES)
        self.live_decision_ms = deque(maxlen=STATS_SAMPLES)
        self.live_alert_ms = deque(maxlen=STATS_SAMPLES)
        self.history = deque(maxlen=20000)
        self.next_graph_scan = None
        self.graph_dirty = False

    def emit(self, event):
        self.log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.log.flush()

    def emit_behavior(self, events):
        for event in events:
            event["logged_utc"] = utc(time())
            self.emit(event)
            self.counts["behavior_alerts"] += int(event["event"] == "behavior_alert")
            self.counts["behavior_windows"] += int(event["event"] == "behavior_window")
            self.counts["behavior_truncated"] += int(event.get("truncated", False))

    def accept(self, observation):
        if self.behavior is not None:
            self.emit_behavior(self.behavior.observe(observation))
        self.handle(self.table.ingest(observation))
        self.scan_graph()

    def tick(self, timestamp):
        if self.behavior is not None:
            self.emit_behavior(self.behavior.advance(timestamp))
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
        if self.detector is None:
            predictions = classify(self.bundle, flows)  # historical CTU-only CLI
        else:
            # Gaussian uses the five-feature CTU schema.
            scores = self.bundle["gaussian"].score_samples(features(flows))
            predictions = pd.DataFrame({"label": [None] * len(flows),
                "confidence": [None] * len(flows), "gaussian_score": scores,
                "gaussian_flag": scores > self.bundle["gaussian_threshold"]})
        self.classification_ms.append(1000 * (perf_counter() - started))
        for (flow, reason), result in zip(completed, predictions.itertuples(index=False)):
            first = flow.first
            gaussian_alert = bool(result.gaussian_flag)
            ctu_alert = (bool(alert_decision(result.label, gaussian_alert))
                         if self.detector is None else False)
            event = {
                "flow_id": uuid4().hex,
                "event": "flow_alert" if ctu_alert or gaussian_alert else "flow_decision",
                "flow_start_utc": utc(first.timestamp), "flow_end_utc": utc(flow.last),
                "flow_start_epoch": first.timestamp, "flow_end_epoch": flow.last,
                "src": first.src, "dst": first.dst, "sport": first.sport, "dport": first.dport,
                "protocol": first.protocol, "packets": flow.packets, "duration_s": round(flow.duration_s, 6),
                "close_reason": reason, "label": result.label,
                "confidence": round(result.confidence, 6) if result.confidence is not None else None,
                "classification_mode": "tabr_dual" if self.detector is not None else "ctu_legacy",
                "gaussian_flag": bool(result.gaussian_flag), "gaussian_score": round(result.gaussian_score, 6),
                "ctu_alert": ctu_alert, "capture_features": flow.stats.snapshot(),
                "host_window": self.host_window.add(self.table.now, first.src, first.dst,
                                                     first.dport, flow.packets),
            }
            dual_alert = False
            if self.detector is not None:
                try:
                    dual = self.detector.detect(event)
                    event["dual_inference"] = dual
                    dual_alert = bool(dual["decision"]["attack_candidate"])
                    self.counts["dual_completed"] += int(dual["decision"]["completed_models"] == 2)
                except Exception as exc:
                    event["dual_inference"] = {"error": f"{type(exc).__name__}: {exc}"}
            alert = gaussian_alert or ctu_alert or dual_alert
            now = time()
            delay = None
            if self.live:
                delay = round(1000 * max(0, now - flow.last), 3)
                self.live_decision_ms.append(delay)
                if alert:
                    self.live_alert_ms.append(delay)
            event.update(event="flow_alert" if alert else "flow_decision",
                         dual_alert=dual_alert, logged_utc=utc(now), post_last_packet_ms=delay)
            self.emit(event)
            if self.behavior is not None:
                candidate = unsw_candidate(event)
                if candidate is not None:
                    candidate["logged_utc"] = utc(now)
                    self.emit(candidate)
                    self.counts["unsw_advisories"] += 1
            self.counts["flow_decisions"] += 1
            self.counts["flow_alerts"] += int(alert)
            self.counts["ctu_alerts"] += int(ctu_alert)
            self.counts["gaussian_alerts"] += int(gaussian_alert)
            self.counts["dual_alerts"] += int(dual_alert)
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
            "behavior_late_packets": self.behavior.late if self.behavior else 0,
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


def capture_live(interface, capture_seconds, bpf, monitor, *, stop=None, stop_file=None,
                 backend="scapy"):
    """Capture only the selected, authorized interface. No packet transmission."""
    stop = stop if stop is not None else Event()
    ready, reader_done = Event(), Event()
    observations = Queue(maxsize=BUFFER_SIZE)
    counters = {"packets_observed": 0, "packets_supported": 0, "queue_drops": 0}
    errors = []
    sniffer = proc = thread = stderr = None

    def should_stop():
        return stop.is_set() or (stop_file is not None and stop_file.exists())

    def enqueue(packet):
        counters["packets_observed"] += 1
        item = packet_observation(packet)
        if item is not None:
            counters["packets_supported"] += 1
            try:
                observations.put_nowait(item)
            except Full:
                counters["queue_drops"] += 1

    def read_pipe():
        try:
            with PcapReader(proc.stdout) as reader:
                ready.set()
                for packet in reader:
                    enqueue(packet)
        except Exception as exc:
            if not should_stop():
                errors.append(str(exc))
        finally:
            reader_done.set()

    def running():
        return sniffer.running if sniffer is not None else not reader_done.is_set()

    failure = None
    try:
        if should_stop():
            return counters
        if backend == "dumpcap":
            executable = shutil.which("dumpcap")
            if executable is None:
                raise RuntimeError("Install wireshark-common and configure non-root dumpcap permissions")
            # Only dumpcap has capture privileges; the Python worker and GUI are normal users.
            command = [executable, "-i", interface, "-p", "-F", "pcap", "-w", "-", "-q"]
            if bpf:
                command.extend(["-f", bpf])
            stderr = tempfile.TemporaryFile()
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=stderr, start_new_session=True)
            thread = Thread(target=read_pipe, daemon=True)
            thread.start()
        else:
            options = {"iface": interface, "store": False, "prn": enqueue,
                       "started_callback": ready.set, "promisc": False}
            if bpf:
                options["filter"] = bpf
            sniffer = AsyncSniffer(**options)
            sniffer.start()

        startup = monotonic() + 10
        while not ready.wait(0.1):
            if should_stop():
                break
            if monotonic() >= startup or not running():
                raise RuntimeError("Could not start capture; check interface and capture privileges")
        if ready.is_set() and not should_stop():
            print(f"Listening on {interface}", flush=True)
            deadline = monotonic() + capture_seconds if capture_seconds else float("inf")
            while not should_stop() and monotonic() < deadline:
                if not running():
                    raise RuntimeError("Capture ended unexpectedly")
                try:
                    item = observations.get(timeout=min(0.2, max(0.001, deadline - monotonic())))
                except Empty:
                    monitor.tick(time())
                else:
                    monitor.accept(item)
    except KeyboardInterrupt:
        stop.set()
    except Exception as exc:
        failure = exc
    finally:
        stop.set()
        if sniffer is not None:
            try:
                if ready.is_set() and sniffer.running:
                    sniffer.stop()
                else:
                    sniffer.join(timeout=1)
            except Exception as exc:
                failure = failure or exc
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if thread is not None:
                thread.join(timeout=5)
                if thread.is_alive():
                    failure = failure or RuntimeError("Capture reader did not terminate")
            if proc.stdout is not None:
                proc.stdout.close()
        if stderr is not None:
            stderr.seek(0)
            message = stderr.read(8192).decode("utf-8", errors="replace").strip()
            stderr.close()
            if failure and message:
                failure = RuntimeError(f"{failure}: {message}")
    while not observations.empty():
        monitor.accept(observations.get_nowait())
    monitor.tick(time())
    if failure:
        raise RuntimeError(str(failure)) from failure
    if errors:
        raise RuntimeError("Capture reader: " + errors[0])
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pcap", type=Path, help="Read an existing capture without transmitting packets.")
    mode.add_argument("--interface", help="Interface selected for passive capture.")
    parser.add_argument("--seconds", type=float, default=60, help="Live duration in seconds (0 = until stopped).")
    parser.add_argument("--bpf", help="Optional live BPF capture filter.")
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--active-seconds", type=float, default=60)
    parser.add_argument("--max-flows", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-backend", choices=("scapy", "dumpcap"), default="scapy")
    parser.add_argument("--stop-file", type=Path, help="Stop safely when this session marker exists")
    parser.add_argument("--tabr", "--dual", dest="dual", action="store_true",
                        help="Enable Gaussian + both TabR checkpoints; --dual is a legacy alias")
    parser.add_argument("--tabr-cache", type=int, choices=(1, 2), default=2)
    parser.add_argument("--feature-provider", help="Trusted validated adapter module:function")
    args = parser.parse_args()
    if not isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative.")
    if args.pcap and not args.pcap.is_file():
        parser.error(f"PCAP file not found: {args.pcap}")
    if args.pcap and args.bpf:
        parser.error("--bpf applies to live capture.")
    if args.pcap and args.stop_file:
        parser.error("--stop-file is for live capture")
    if args.feature_provider and not args.dual:
        parser.error("Feature provider requires --dual")
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        table = FlowTable(args.idle_seconds, args.active_seconds, args.max_flows)
        bundle = load_monitor()
        if args.dual:
            bundle.pop("classifier", None)  # Gaussian only in TabR mode
        detector = (UnifiedDetectionEngine(provider=load_provider(args.feature_provider),
                    max_cached_models=args.tabr_cache) if args.dual else None)
        args.output.mkdir(parents=True, exist_ok=True)
        live = args.interface is not None
        start = perf_counter()
        with (args.output / "events.jsonl").open("x", encoding="utf-8") as log:
            monitor = Monitor(bundle, log, table, live=live, detector=detector)
            try:
                counters = (capture_live(args.interface, args.seconds, args.bpf, monitor, stop=stop,
                            stop_file=args.stop_file, backend=args.capture_backend)
                            if live else read_pcap(args.pcap, monitor))
            finally:
                monitor.finish()
        result = {
            "mode": "live" if live else "pcap",
            "dual_enabled": args.dual,
            "classification_mode": "tabr_dual" if args.dual else "ctu_legacy",
            "capture_backend": args.capture_backend if live else "pcap",
            "source": args.interface if live else args.pcap.name,
            "seconds_elapsed": round(perf_counter() - start, 3),
            "flow_idle_seconds": table.idle_seconds, "flow_active_seconds": table.active_seconds,
            **counters, **monitor.summary(),
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except FileExistsError:
        parser.exit(2, "Choose a new output directory to preserve prior event logs.\n")
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
