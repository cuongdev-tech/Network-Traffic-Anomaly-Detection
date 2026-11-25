"""Passive, bounded 60-second packet-window heuristics; not a trained model."""

from collections import Counter, defaultdict
from math import isfinite


class BehaviorWindow:

    def __init__(
        self,
        seconds=60,
        scan_ports=10,
        flood_pps=500,
        distributed_sources=5,
        max_keys=50000,
    ):

        if (
            not isfinite(seconds)
            or seconds <= 0
            or scan_ports < 2
            or flood_pps <= 0
            or distributed_sources < 2
            or max_keys < 1
        ):

            raise ValueError(
                "Invalid behavior window settings"
            )

        self.seconds = seconds
        self.scan_ports = scan_ports
        self.flood_pps = flood_pps
        self.distributed_sources = distributed_sources
        self.max_keys = max_keys

        self.start = None
        self.late = 0

        self._reset()

    def _reset(self):

        self.packets = 0
        self.by_dst = Counter()
        self.sources = defaultdict(set)
        self.ports = defaultdict(set)

        self.source_keys = 0
        self.port_keys = 0
        self.truncated = False

    def advance(self, timestamp):

        if not isfinite(timestamp):
            return []

        if self.start is None:

            self.start = timestamp
            return []

        end = self.start + self.seconds

        if timestamp < end:
            return []

        alerts = []

        # Evaluate complete windows only.

        if self.packets and not self.truncated:

            # ==================================
            # PORTSCAN DETECTION
            # ==================================

            for (src, dst), ports in self.ports.items():

                if len(ports) >= self.scan_ports:

                    alerts.append(dict(

                        event="behavior_alert",
                        label="PortScan",
                        source="Host behavior (heuristic)",

                        src=src,
                        dst=dst,

                        window_start_epoch=self.start,
                        window_end_epoch=end,
                        window_seconds=self.seconds,

                        evidence=(
                            f"{len(ports)} distinct TCP "
                            "destination ports with initial SYN"
                        ),
                    ))

            # ==================================
            # DDOS DETECTION
            # ==================================

            for dst, count in self.by_dst.items():

                rate = count / self.seconds

                n = len(self.sources[dst])

                if (
                    rate >= self.flood_pps
                    and n >= self.distributed_sources
                ):

                    alerts.append(dict(

                        event="behavior_alert",
                        label="DDoS",
                        source="Network behavior (heuristic)",

                        src=None,
                        dst=dst,

                        window_start_epoch=self.start,
                        window_end_epoch=end,
                        window_seconds=self.seconds,

                        evidence=(
                            f"{rate:.1f} packets/s; "
                            f"{n} observed source IPs"
                        ),
                    ))

        summary = dict(

            event="behavior_window",

            window_start_epoch=self.start,
            window_end_epoch=end,

            window_seconds=self.seconds,
            packets=self.packets,

            truncated=self.truncated,

            alerts=len(alerts),
        )

        self.start += int(
            (timestamp - self.start) // self.seconds
        ) * self.seconds

        self._reset()

        return [summary, *alerts]

    def observe(self, obs):

        if not isfinite(obs.timestamp):
            return []

        events = self.advance(
            obs.timestamp
        )

        if obs.timestamp < self.start:

            self.late += 1

            return events

        self.packets += 1

        if self.truncated:
            return events

        if obs.src not in self.sources.get(obs.dst, ()):

            if self.source_keys >= self.max_keys:

                self.truncated = True
                return events

            self.sources[obs.dst].add(
                obs.src
            )

            self.source_keys += 1

        self.by_dst[obs.dst] += 1

        # Count distinct initial TCP SYN ports.
        # Ignore response SYN-ACK and duplicate ports.

        flags = getattr(
            obs.meta,
            "tcp_flags",
            0,
        )

        if (
            obs.protocol == "tcp"
            and flags & 2
            and not flags & 16
        ):

            key = (
                obs.src,
                obs.dst,
            )

            if obs.dport not in self.ports.get(key, ()):

                if self.port_keys >= self.max_keys:

                    self.truncated = True
                    return events

                self.ports[key].add(
                    obs.dport
                )

                self.port_keys += 1

        return events


# ==========================================
# UNSW BACKGROUND CLASSIFICATION
# ==========================================

def unsw_candidate(event, min_confidence=0.9):

    """Experimental advisory, not a confirmed attack."""

    dual = event.get(
        "dual_inference"
    ) or {}

    branch = dual.get("unsw") or {}

    if branch.get(
        "ood", {}
    ).get("status") != "IN_RANGE":

        return None

    threshold = max(

        min_confidence,

        branch.get(
            "min_confidence", 0
        ),
    )

    for model in branch.get("models", []):

        predictions = model.get(
            "predictions", []
        )

        if (
            model.get("model") != "TabR"
            or model.get("status") != "OK"
            or len(predictions) != 1
        ):

            continue

        p = predictions[0]

        label = p.get(

            "candidate_label",

            p.get("label"),
        )

        if isinstance(label, str):

            label = label.removeprefix(
                "[DEMO] "
            )

        confidence = p.get(
            "confidence"
        )

        attacks = {

            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        }

        if label not in attacks:

            continue

        if (
            not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not threshold <= confidence <= 1
        ):

            continue

        return dict(

            event="model_candidate",

            label=label,

            source="TabR UNSW (unverified)",

            flow_id=event.get("flow_id"),

            src=event.get("src"),

            dst=event.get("dst"),

            confidence=confidence,

            evidence=(
                "OOD in range; experimental "
                "prediction, not confirmed"
            ),

            feature_quality=branch.get(
                "feature_quality",
                "UNKNOWN",
            ),
        )

    return None
