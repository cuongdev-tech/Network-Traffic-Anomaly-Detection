"""Bounded aggregation of IPv4 TCP, UDP and ICMP packet metadata."""
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import isfinite

from scapy.layers.inet import ICMP, IP, TCP, UDP

from src.pcap_to_flows import canon_key
from src.live_flow_stats import FlowStats, PacketMeta, read_metadata

PROTOCOLS = ("tcp", "udp", "icmp")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    src: str
    dst: str
    sport: int
    dport: int
    protocol: str
    meta: PacketMeta | None = None


@dataclass
class Flow:
    first: Observation
    last: float
    packets: int = 1
    stats: FlowStats = field(default_factory=FlowStats)

    def __post_init__(self):
        self.stats.add(self.first, self.first)

    @property
    def duration_s(self):
        return self.last - self.first.timestamp


def packet_observation(packet):

    ip = packet.getlayer(IP, 1)

    if ip is None or int(ip.frag) != 0 or "MF" in str(ip.flags):
        return None

    transport = ip.payload
    timestamp = float(packet.time)

    if not isfinite(timestamp):
        return None

    if isinstance(transport, (TCP, UDP)):

        protocol = "tcp" if isinstance(transport, TCP) else "udp"
        sport, dport = int(transport.sport), int(transport.dport)

    elif isinstance(transport, ICMP):

        protocol, sport, dport = "icmp", 0, 0

    else:
        return None

    return Observation(
        timestamp,
        ip.src,
        ip.dst,
        sport,
        dport,
        protocol,
        read_metadata(ip),
    )


class FlowTable:
    """Expire flows and maintain local closure-order history."""

    def __init__(self, idle_seconds=5.0, active_seconds=60.0, max_flows=10000):

        if (
            not (0 < idle_seconds <= active_seconds)
            or not isfinite(active_seconds)
            or max_flows < 1
        ):
            raise ValueError(
                "Use 0 < idle seconds <= active seconds and a positive flow limit."
            )

        self.idle_seconds = idle_seconds
        self.active_seconds = active_seconds
        self.max_flows = max_flows

        self.states = {}
        self.deadlines = []

        self.now = float("-inf")

        self.capacity_drops = self.late_packets = 0

        self.recent = deque(maxlen=99)

    def _finalize(self, ready):

        for flow, _ in ready:

            first = flow.first

            src_dport = (first.src, first.dport)
            dst_sport = (first.dst, first.sport)

            flow.stats.history = {
                "scope": "local_last_100_closed_flows",
                "observed_connections": len(self.recent) + 1,

                "ct_src_dport_ltm": 1 + sum(
                    a == src_dport for a, _ in self.recent
                ),

                "ct_dst_sport_ltm": 1 + sum(
                    b == dst_sport for _, b in self.recent
                ),
            }

            self.recent.append((src_dport, dst_sport))

        return ready

    def expire(self, now):

        self.now = max(self.now, now)
        ready = []

        while self.deadlines and self.deadlines[0][0] <= self.now:

            _, key = heappop(self.deadlines)

            flow = self.states[key]

            idle = flow.last + self.idle_seconds
            active = flow.first.timestamp + self.active_seconds

            deadline = min(idle, active)

            if deadline > self.now:

                heappush(self.deadlines, (deadline, key))

            else:

                ready.append((
                    self.states.pop(key),
                    "idle" if idle <= active else "active",
                ))

        return self._finalize(ready)

    def ingest(self, observation):

        if observation.timestamp < self.now:
            self.late_packets += 1

        ready = self.expire(observation.timestamp)

        key = canon_key(
            observation.src,
            observation.dst,
            observation.sport,
            observation.dport,
            observation.protocol,
        )

        flow = self.states.get(key)

        if flow is not None:

            flow.last = max(flow.last, observation.timestamp)
            flow.packets += 1

            flow.stats.add(observation, flow.first)

        elif len(self.states) >= self.max_flows:

            self.capacity_drops += 1

        else:

            self.states[key] = Flow(
                observation,
                observation.timestamp,
            )

            heappush(
                self.deadlines,
                (observation.timestamp + self.idle_seconds, key),
            )

        return ready

    def finish(self):

        ready = [
            (flow, "end_of_input")
            for flow in sorted(
                self.states.values(),
                key=lambda f: f.last,
            )
        ]

        self.states.clear()
        self.deadlines.clear()

        return self._finalize(ready)
