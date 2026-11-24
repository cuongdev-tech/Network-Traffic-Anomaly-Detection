
from collections import Counter
from dataclasses import dataclass
from math import sqrt

from scapy.layers.inet import IP, TCP, UDP


@dataclass(frozen=True)
class PacketMeta:
    ip_bytes: int
    ip_header_bytes: int
    transport_header_bytes: int
    payload_bytes: int
    tcp_flags: int = 0
    tcp_window: int | None = None


def read_metadata(ip):
    transport = ip.payload

    if not isinstance(transport, (TCP, UDP)):
        return None

    try:
        # Resolve unset header fields in constructed packets.
        if (
            ip.len is None
            or ip.ihl is None
            or (
                isinstance(transport, TCP)
                and transport.dataofs is None
            )
            or (
                isinstance(transport, UDP)
                and transport.len is None
            )
        ):
            ip = IP(bytes(ip))
            transport = ip.payload

        total = int(ip.len)
        ihl = int(ip.ihl) * 4

        captured = (
            len(ip.original)
            if ip.original
            else len(ip)
        )

        if isinstance(transport, TCP):
            header = int(transport.dataofs) * 4
            flags = int(transport.flags)
            window = int(transport.window)

            if not 20 <= header <= 60:
                return None

        else:
            header = 8
            flags = 0
            window = None

            if int(transport.len) != total - ihl:
                return None

        if not (
            20 <= ihl <= 60
            and ihl + header <= total <= captured
        ):
            return None

        return PacketMeta(
            ip_bytes=total,
            ip_header_bytes=ihl,
            transport_header_bytes=header,
            payload_bytes=total - ihl - header,
            tcp_flags=flags,
            tcp_window=window,
        )

    except (ValueError, TypeError, OverflowError):
        return None


class Moments:
    """Constant-memory online statistics."""

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        self.n += 1
        self.total += value

        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

        self.minimum = (
            value if self.minimum is None
            else min(self.minimum, value)
        )

        self.maximum = (
            value if self.maximum is None
            else max(self.maximum, value)
        )

    def snapshot(self):
        return {
            "count": self.n,
            "total": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean if self.n else None,
            "std": (
                sqrt(max(0.0, self.m2 / (self.n - 1)))
                if self.n > 1
                else 0.0 if self.n == 1 else None
            ),
        }


FLAGS = {
    "FIN": 0x01,
    "SYN": 0x02,
    "RST": 0x04,
    "PSH": 0x08,
    "ACK": 0x10,
    "URG": 0x20,
    "ECE": 0x40,
    "CWR": 0x80,
}


class DirectionStats:

    def __init__(self):
        self.packets = 0
        self.last_timestamp = None

        self.ip_length = Moments()
        self.payload_length = Moments()
        self.ip_header = Moments()
        self.transport_header = Moments()
        self.iat = Moments()

        self.flags = Counter()
        self.initial_window = None

    def add(self, observation):
        self.packets += 1
        timestamp = observation.timestamp

        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp

            if delta >= 0:
                self.iat.add(delta)

        self.last_timestamp = (
            timestamp
            if self.last_timestamp is None
            else max(self.last_timestamp, timestamp)
        )

        meta = observation.meta

        if meta is None:
            return

        self.ip_length.add(meta.ip_bytes)
        self.payload_length.add(meta.payload_bytes)
        self.ip_header.add(meta.ip_header_bytes)
        self.transport_header.add(
            meta.transport_header_bytes
        )

        if meta.tcp_window is not None:
            if self.initial_window is None:
                self.initial_window = meta.tcp_window

            for name, mask in FLAGS.items():
                if meta.tcp_flags & mask:
                    self.flags[name] += 1

    def snapshot(self):
        return {
            "packets": self.packets,
            "ip_length": self.ip_length.snapshot(),
            "payload_length": self.payload_length.snapshot(),
            "ip_header": self.ip_header.snapshot(),
            "transport_header": self.transport_header.snapshot(),
            "iat_seconds": self.iat.snapshot(),
            "tcp_flags": dict(self.flags),
            "initial_window": self.initial_window,
        }


class FlowStats:

    def __init__(self):
        self.forward = DirectionStats()
        self.backward = DirectionStats()
        self.iat = Moments()
        self.last_timestamp = None
        self.missing_metadata = 0
        self.out_of_order = 0

    def add(self, observation, first):
        timestamp = observation.timestamp

        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp

            if delta >= 0:
                self.iat.add(delta)
            else:
                self.out_of_order += 1

        self.last_timestamp = (
            timestamp
            if self.last_timestamp is None
            else max(self.last_timestamp, timestamp)
        )

        if observation.meta is None:
            self.missing_metadata += 1

        is_forward = (
            observation.src,
            observation.sport,
        ) == (
            first.src,
            first.sport,
        )

        direction = (
            self.forward if is_forward
            else self.backward
        )

        direction.add(observation)

    def snapshot(self):
        return {
            "forward": self.forward.snapshot(),
            "backward": self.backward.snapshot(),
            "flow_iat_seconds": self.iat.snapshot(),
            "missing_metadata": self.missing_metadata,
            "out_of_order": self.out_of_order,
        }
