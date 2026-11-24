"""Bounded sliding window of completed flows by initiating host, for logging only."""
from collections import Counter, deque
from math import isfinite


class HostWindow:

    def __init__(self, seconds=60, limit=20000):

        if (
            not isfinite(seconds)
            or seconds <= 0
            or limit < 1
        ):

            raise ValueError(
                "Positive window and limit required"
            )

        self.seconds = seconds
        self.limit = limit

        self.records = deque()
        self.hosts = {}

        self.now = float("-inf")
        self.last_evicted = float("-inf")

    def _remove(self):

        t, src, dst, port, packets = (
            self.records.popleft()
        )

        host = self.hosts[src]

        host["flows"] -= 1
        host["packets"] -= packets

        for key, value in (
            ("targets", dst),
            ("ports", port),
        ):

            host[key][value] -= 1

            if not host[key][value]:

                del host[key][value]

        if not host["flows"]:

            del self.hosts[src]

        return t

    def add(self, timestamp, src, dst, port, packets):

        if not isfinite(timestamp) or packets < 1:

            raise ValueError(
                "Invalid completed flow"
            )

        self.now = max(self.now, timestamp)

        while (
            self.records
            and self.records[0][0]
            <= self.now - self.seconds
        ):

            self._remove()

        if len(self.records) == self.limit:

            self.last_evicted = self._remove()

        host = self.hosts.setdefault(
            src,
            {
                "flows": 0,
                "packets": 0,
                "targets": Counter(),
                "ports": Counter(),
            },
        )

        host["flows"] += 1
        host["packets"] += packets

        host["targets"][dst] += 1
        host["ports"][port] += 1

        self.records.append(
            (self.now, src, dst, port, packets)
        )

        return {
            "scope": "completed_flows_by_initiator",
            "window_seconds": self.seconds,
            "flows": host["flows"],
            "packets": host["packets"],
            "unique_destinations": len(
                host["targets"]
            ),
            "unique_dst_ports": len(
                host["ports"]
            ),
            "capacity_truncated": (
                self.last_evicted
                > self.now - self.seconds
            ),
        }
