"""Bounded, payload-free flow statistics. Local feature definitions, not Argus parity."""
from collections import Counter
from dataclasses import dataclass
from math import sqrt

FLAGS = {"FIN": 1, "SYN": 2, "RST": 4, "PSH": 8, "ACK": 16,
         "URG": 32, "ECE": 64, "CWR": 128}


@dataclass(frozen=True)
class PacketMeta:
    ip_bytes: int
    ip_header_bytes: int
    transport_header_bytes: int
    payload_bytes: int
    tcp_flags: int = 0
    tcp_window: int | None = None
    tcp_seq: int | None = None
    tcp_ack: int | None = None


def read_metadata(ip):
    """Read actual IPv4 lengths, excluding Ethernet padding. No payload retained."""
    try:
        raw = bytes(ip)
        if len(raw) < 20 or raw[0] >> 4 != 4:
            return None
        ihl = (raw[0] & 15) * 4
        total = int.from_bytes(raw[2:4], "big")
        if not 20 <= ihl <= total <= len(raw):
            return None
        if int.from_bytes(raw[6:8], "big") & 0x3fff:
            return None
        proto = raw[9]
        if proto == 6:
            if total < ihl + 20:
                return None
            header = (raw[ihl + 12] >> 4) * 4
            if not 20 <= header <= total - ihl:
                return None
            return PacketMeta(
                total, ihl, header, total - ihl - header,
                raw[ihl + 13],
                int.from_bytes(raw[ihl+14:ihl+16], "big"),
                int.from_bytes(raw[ihl+4:ihl+8], "big"),
                int.from_bytes(raw[ihl+8:ihl+12], "big")
            )
        if proto == 17:
            if total < ihl + 8:
                return None
            length = int.from_bytes(raw[ihl+4:ihl+6], "big")
            if not 8 <= length == total - ihl:
                return None
            return PacketMeta(total, ihl, 8, length - 8)
    except (TypeError, ValueError, IndexError, OverflowError, OSError):
        return None
    return None


class Moments:
    def __init__(self):
        self.n = 0
        self.total = self.mean = self.m2 = 0.0
        self.minimum = self.maximum = None

    def add(self, value):
        value = float(value)
        self.n += 1
        self.total += value
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def snapshot(self):
        return {
            "count": self.n,
            "total": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean if self.n else None,
            "std": sqrt(max(0, self.m2 / (self.n - 1)))
                   if self.n > 1 else (0.0 if self.n else None),
        }


class BulkStats:
    """Local bulk: >=4 same-direction payload packets, gaps <=1 s."""

    def __init__(self):
        self.groups = self.packets = self.bytes = 0
        self.duration_s = 0.0
        self.reset()

    def reset(self):
        self.start = self.last = None
        self.pending_packets = self.pending_bytes = 0

    def add(self, timestamp, size):
        if self.last is None or timestamp - self.last > 1.0:
            self.reset()
            self.start = timestamp

        self.pending_packets += 1
        self.pending_bytes += size

        if self.pending_packets == 4:
            self.groups += 1
            self.packets += 4
            self.bytes += self.pending_bytes
            self.duration_s += timestamp - self.start

        elif self.pending_packets > 4:
            self.packets += 1
            self.bytes += size
            self.duration_s += timestamp - self.last

        self.last = timestamp

    def snapshot(self):
        return dict(
            groups=self.groups,
            packets=self.packets,
            bytes=self.bytes,
            duration_s=self.duration_s,
        )


class DirectionStats:
    def __init__(self):
        self.packets = self.data_packets = 0
        self.first_timestamp = self.last_timestamp = None
        self.ip_length = Moments()
        self.payload_length = Moments()
        self.ip_header = Moments()
        self.transport_header = Moments()
        self.iat = Moments()
        self.flags = Counter()
        self.first_flags = None
        self.initial_window = None
        self.initial_syn_seq = None
        self.bulk = BulkStats()

    def add(self, timestamp, meta):
        self.packets += 1

        if self.last_timestamp is not None:
            self.iat.add(max(0, timestamp - self.last_timestamp))
        else:
            self.first_timestamp = timestamp

        self.last_timestamp = (
            timestamp if self.last_timestamp is None
            else max(timestamp, self.last_timestamp)
        )

        if meta is None:
            return

        self.ip_length.add(meta.ip_bytes)
        self.payload_length.add(meta.payload_bytes)
        self.ip_header.add(meta.ip_header_bytes)
        self.transport_header.add(meta.transport_header_bytes)
        self.data_packets += int(meta.payload_bytes > 0)

        if self.packets == 1:
            self.first_flags = meta.tcp_flags
            self.initial_window = meta.tcp_window

            if meta.tcp_flags & FLAGS["SYN"]:
                self.initial_syn_seq = meta.tcp_seq

        for name, bit in FLAGS.items():
            if meta.tcp_flags & bit:
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
            "first_flags": self.first_flags,
            "initial_window": self.initial_window,
            "initial_syn_seq": self.initial_syn_seq,
            "data_packets": self.data_packets,
            "bulk": self.bulk.snapshot(),
        }


class FlowStats:
    def __init__(self):
        self.forward = DirectionStats()
        self.backward = DirectionStats()

        self.iat = Moments()
        self.payload = Moments()
        self.legacy_payload = Moments()

        self.active = Moments()
        self.idle = Moments()

        self.last_timestamp = self.active_start = None
        self.first_payload = None

        self.missing_metadata = self.out_of_order = 0

        self.subflows = 1
        self.history = None

        self.syn = self.synack = self.final_ack = None
        self.handshake_ambiguous = False

    def _handshake(self, timestamp, meta, forward, first):

        if meta.tcp_seq is None or meta.tcp_ack is None:
            return

        syn = bool(meta.tcp_flags & 2)
        ack = bool(meta.tcp_flags & 16)

        if self.final_ack is not None:
            return

        if forward and syn and not ack:

            if first:
                self.syn = (timestamp, meta.tcp_seq)

            elif self.syn is not None:
                self.handshake_ambiguous = True

        elif not forward and syn and ack and self.syn is not None:

            if self.synack is not None:
                self.handshake_ambiguous = True

            elif meta.tcp_ack == (self.syn[1] + 1) % (2**32):
                self.synack = (timestamp, meta.tcp_seq)

        elif (
            forward and ack and not syn
            and not (meta.tcp_flags & 5)
            and self.synack is not None
        ):

            if meta.tcp_ack == (self.synack[1] + 1) % (2**32):

                if meta.tcp_seq == (self.syn[1] + 1) % (2**32):
                    self.final_ack = timestamp

    def add(self, observation, first):

        timestamp, meta = observation.timestamp, observation.meta

        is_first = self.last_timestamp is None

        forward = (
            observation.src, observation.sport
        ) == (
            first.src, first.sport
        )

        direction = self.forward if forward else self.backward
        other = self.backward if forward else self.forward

        if is_first:
            self.active_start = timestamp

        else:
            gap = timestamp - self.last_timestamp

            if gap < 0:
                self.out_of_order += 1

            self.iat.add(max(0, gap))

            if gap > 1.0:
                self.subflows += 1

            if gap > 5.0:

                active_duration = self.last_timestamp - self.active_start

                if active_duration > 0:
                    self.active.add(active_duration)

                self.idle.add(gap)
                self.active_start = timestamp

        self.last_timestamp = (
            timestamp if is_first
            else max(timestamp, self.last_timestamp)
        )

        direction.add(timestamp, meta)

        if meta is None:
            self.missing_metadata += 1
            return

        self.payload.add(meta.payload_bytes)
        self.legacy_payload.add(meta.payload_bytes)

        if is_first:
            self.first_payload = meta.payload_bytes
            self.legacy_payload.add(meta.payload_bytes)

        if meta.payload_bytes > 0 and not self.out_of_order:

            other.bulk.reset()
            direction.bulk.add(timestamp, meta.payload_bytes)

        self._handshake(timestamp, meta, forward, is_first)

    def snapshot(self):

        handshake = None

        if (
            not self.handshake_ambiguous
            and not self.out_of_order
            and self.syn is not None
            and self.synack is not None
            and self.final_ack is not None
        ):

            a = self.synack[0] - self.syn[0]
            b = self.final_ack - self.synack[0]

            if a >= 0 and b >= 0:
                handshake = dict(
                    synack=a,
                    ackdat=b,
                    tcprtt=a+b,
                )

        active_tail = (
            None if self.active_start is None
            else self.last_timestamp - self.active_start
        )

        return {
            "schema_version": 2,
            "forward": self.forward.snapshot(),
            "backward": self.backward.snapshot(),
            "flow_iat_seconds": self.iat.snapshot(),
            "flow_payload_length": self.payload.snapshot(),
            "legacy_payload_length": self.legacy_payload.snapshot(),
            "first_payload_bytes": self.first_payload,
            "subflows": self.subflows,
            "active_seconds": self.active.snapshot(),
            "idle_seconds": self.idle.snapshot(),
            "active_tail_seconds": active_tail,
            "handshake": handshake,
            "history": self.history,
            "missing_metadata": self.missing_metadata,
            "out_of_order": self.out_of_order,
        }
