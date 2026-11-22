"""Bounded aggregation of IPv4 TCP, UDP and ICMP packet metadata."""
from dataclasses import dataclass
from heapq import heappop, heappush
from math import isfinite

from scapy.layers.inet import ICMP, IP, TCP, UDP

from src.pcap_to_flows import canon_key


PROTOCOLS = ("tcp", "udp", "icmp")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    src: str
    dst: str
    sport: int
    dport: int
    protocol: str


@dataclass
class Flow:
    first: Observation
    last: float
    packets: int = 1

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
    return Observation(timestamp, ip.src, ip.dst, sport, dport, protocol)


class FlowTable:
    """Expire at inactivity or maximum lifetime without storing packet contents."""

    def __init__(self, idle_seconds=5.0, active_seconds=60.0, max_flows=10000):
        if not (0 < idle_seconds <= active_seconds) or not isfinite(active_seconds) or max_flows < 1:
            raise ValueError("Use 0 < idle seconds <= active seconds and a positive flow limit.")
        self.idle_seconds = idle_seconds
        self.active_seconds = active_seconds
        self.max_flows = max_flows
        self.states = {}
        self.deadlines = []
        self.now = float("-inf")
        self.capacity_drops = 0
        self.late_packets = 0

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
                ready.append((self.states.pop(key), "idle" if idle <= active else "active"))
        return ready

    def ingest(self, observation):
        if observation.timestamp < self.now:
            self.late_packets += 1
        ready = self.expire(observation.timestamp)
        key = canon_key(observation.src, observation.dst, observation.sport,
                        observation.dport, observation.protocol)
        flow = self.states.get(key)
        if flow is not None:
            flow.last = max(flow.last, observation.timestamp)
            flow.packets += 1
        elif len(self.states) >= self.max_flows:
            self.capacity_drops += 1
        else:
            self.states[key] = Flow(observation, observation.timestamp)
            heappush(self.deadlines, (observation.timestamp + self.idle_seconds, key))
        return ready

    def finish(self):
        ready = [(flow, "end_of_input") for flow in self.states.values()]
        self.states.clear()
        self.deadlines.clear()
        return ready
