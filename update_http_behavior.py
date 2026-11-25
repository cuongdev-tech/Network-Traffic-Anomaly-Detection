
import ast
import shutil
import tempfile
from pathlib import Path


MODULE = r'''
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from time import time


WINDOW_SECONDS = 60
MAX_GROUPS = 4096
MAX_REQUESTS = 20000

STANDARD = frozenset(
    (
        "GET HEAD POST PUT DELETE CONNECT OPTIONS TRACE PATCH "
        "PROPFIND PROPPATCH MKCOL COPY MOVE LOCK UNLOCK REPORT "
        "MKACTIVITY CHECKOUT MERGE SEARCH ACL BIND UNBIND REBIND "
        "MKCALENDAR LINK UNLINK PURGE"
    ).split()
)

REQUEST = re.compile(
    rb"^([A-Z][A-Z0-9_-]{0,31}) [^ \r\n]{1,2048} HTTP/1\.[01]\r?\n"
)

RESPONSE = re.compile(
    rb"^HTTP/1\.[01] ([0-9]{3})[^\r\n]*\r?\n"
)


def utc(timestamp):

    return datetime.fromtimestamp(
        timestamp,
        timezone.utc
    ).isoformat(timespec="milliseconds")


@dataclass
class Group:

    requests: int = 0

    heads: int = 0

    options: int = 0

    probe_connections: set = field(default_factory=set)

    rejected_methods: set = field(default_factory=set)

    rejected_connections: set = field(default_factory=set)

    rejected: int = 0


class HttpBehavior:

    def __init__(self, seconds=WINDOW_SECONDS):

        if not isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid window duration")

        self.seconds = seconds

        self.start = None

        self.windows = 0
        self.alerts = 0
        self.truncated_windows = 0
        self.late = 0

        self._reset()

    def _reset(self):

        self.groups = {}

        self.seen = set()

        self.pending = {}

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

        events = []

        self.windows += 1

        if self.truncated:

            self.truncated_windows += 1

        else:

            for (src, dst, port), g in sorted(
                self.groups.items()
            ):

                base = {

                    "event": "behavior_alert",

                    "source": "HTTP Behavior (heuristic)",

                    "src": src,

                    "dst": dst,

                    "dport": port,

                    "window_seconds": self.seconds,

                    "window_start_epoch": self.start,

                    "window_end_epoch": end,

                    "logged_utc": utc(time()),

                    "feature_quality":
                        "HTTP_PLAINTEXT_HEURISTIC",
                }

                # ==========================
                # ANALYSIS-LIKE DETECTION
                # ==========================

                probes = g.heads + g.options

                if (

                    g.heads >= 2

                    and g.options >= 2

                    and len(g.probe_connections) >= 4

                    and probes / max(g.requests, 1) >= 0.75

                ):

                    events.append({

                        **base,

                        "label":
                            "Analysis-like (HTTP probing)",

                        "evidence": (

                            f"HEAD={g.heads}; "

                            f"OPTIONS={g.options}; "

                            f"{len(g.probe_connections)} connections; "

                            f"Probe share={probes / g.requests:.0%}; "

                            f"Port={port}"
                        ),
                    })

                # ==========================
                # FUZZERS-LIKE DETECTION
                # ==========================

                if (

                    len(g.rejected_methods) >= 3

                    and g.rejected >= 3

                    and len(g.rejected_connections) >= 3

                ):

                    events.append({

                        **base,

                        "label":
                            "Fuzzers-like (HTTP methods)",

                        "evidence": (

                            f"{len(g.rejected_methods)} "

                            "distinct unsupported methods; "

                            f"{g.rejected} rejected responses; "

                            f"{len(g.rejected_connections)} connections; "

                            f"Port={port}"
                        ),
                    })

        self.alerts += len(events)

        steps = int(
            (timestamp - self.start) // self.seconds
        )

        self.start += steps * self.seconds

        self._reset()

        return events

    def observe(self, obs):

        timestamp = float(obs.timestamp)

        if not isfinite(timestamp):

            return []

        if (

            self.start is not None

            and timestamp < self.start

        ):

            self.late += 1

            return []

        events = self.advance(timestamp)

        if self.truncated or obs.protocol != "tcp":

            return events

        payload = getattr(
            obs,
            "app_payload",
            None
        )

        meta = getattr(
            obs,
            "meta",
            None
        )

        seq = getattr(
            meta,
            "tcp_seq",
            None
        )

        if not payload or seq is None:

            return events

        key = (

            obs.src,

            obs.dst,

            obs.sport,

            obs.dport,
        )

        # ==========================
        # HTTP REQUEST ANALYSIS
        # ==========================

        request = REQUEST.match(payload)

        if request:

            method = request.group(1).decode("ascii")

            dedup = (*key, int(seq))

            if dedup in self.seen:

                return events

            if len(self.seen) >= MAX_REQUESTS:

                self.truncated = True

                return events

            self.seen.add(dedup)

            group_key = (

                obs.src,

                obs.dst,

                obs.dport,
            )

            if group_key not in self.groups:

                if len(self.groups) >= MAX_GROUPS:

                    self.truncated = True

                    return events

                self.groups[group_key] = Group()

            group = self.groups[group_key]

            group.requests += 1

            self.pending.pop(key, None)

            if method in {"HEAD", "OPTIONS"}:

                group.heads += int(
                    method == "HEAD"
                )

                group.options += int(
                    method == "OPTIONS"
                )

                group.probe_connections.add(
                    obs.sport
                )

            elif method not in STANDARD:

                self.pending[key] = (
                    group_key,
                    method
                )

            return events

        # ==========================
        # HTTP RESPONSE ANALYSIS
        # ==========================

        response = RESPONSE.match(payload)

        if response:

            reverse = (

                obs.dst,

                obs.src,

                obs.dport,

                obs.sport,
            )

            status = int(
                response.group(1)
            )

            if status < 200:

                return events

            pending = self.pending.pop(
                reverse,
                None
            )

            if (

                pending is not None

                and status in {400, 405, 501}

            ):

                group_key, method = pending

                group = self.groups[group_key]

                group.rejected += 1

                group.rejected_methods.add(
                    method
                )

                group.rejected_connections.add(
                    obs.dport
                )

        return events


# ==========================
# MONITOR INTEGRATION
# ==========================

def with_http_behavior(BaseMonitor):

    class HttpMonitor(BaseMonitor):

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

            self.http_behavior = HttpBehavior()

        def _http_publish(self, events):

            for event in events:

                self.emit(event)

        def accept(self, observation):

            if self.detector is not None:

                events = self.http_behavior.observe(
                    observation
                )

                self._http_publish(events)

            return super().accept(observation)

        def tick(self, timestamp):

            if (

                self.detector is not None

                and self.http_behavior.start is not None

            ):

                events = self.http_behavior.advance(
                    timestamp
                )

                self._http_publish(events)

            return super().tick(timestamp)

        def summary(self):

            result = super().summary()

            h = self.http_behavior

            result.update({

                "http_behavior_windows":
                    h.windows,

                "http_behavior_alerts":
                    h.alerts,

                "http_behavior_truncated_windows":
                    h.truncated_windows,

                "http_behavior_late_packets":
                    h.late,

                "http_behavior_partial_requests":
                    len(h.seen),
            })

            return result

    return HttpMonitor
'''


# ==========================
# AUTOMATIC PROJECT UPDATE
# ==========================

def update(root):

    monitor = root / "src/stage4_monitor.py"

    module = root / "src/http_behavior.py"

    if not monitor.is_file():

        raise SystemExit(
            "Run from Network-IDS project root"
        )

    if not (root / "src/window_alerts.py").is_file():

        raise SystemExit(
            "Existing window_alerts.py not found"
        )

    source = monitor.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    matches = [

        node for node in tree.body

        if isinstance(node, ast.ClassDef)

        and node.name == "Monitor"
    ]

    if len(matches) != 1:

        raise SystemExit(
            "Monitor class mismatch"
        )

    node = matches[0]

    methods = {

        n.name for n in node.body

        if isinstance(n, ast.FunctionDef)
    }

    required = {

        "__init__",

        "emit",

        "accept",

        "tick",

        "summary",
    }

    if not required.issubset(methods):

        raise SystemExit(
            "Monitor API mismatch"
        )

    if "@with_http_behavior" in source:

        if module.is_file():

            print("Already updated")

            return

        raise SystemExit(
            "HTTP module missing"
        )

    if module.exists():

        raise SystemExit(
            "http_behavior.py already exists"
        )

    first = min(

        [node.lineno]

        + [d.lineno for d in node.decorator_list]

    ) - 1

    lines = source.splitlines(
        keepends=True
    )

    lines.insert(

        first,

        "from src.http_behavior import with_http_behavior\n\n"
        "@with_http_behavior\n"
    )

    updated = "".join(lines)

    compile(
        MODULE,
        str(module),
        "exec"
    )

    compile(
        updated,
        str(monitor),
        "exec"
    )

    backup = Path(

        tempfile.mkdtemp(
            prefix="attt_http_behavior_"
        )
    )

    shutil.copy2(
        monitor,
        backup / monitor.name
    )

    try:

        module.write_text(
            MODULE,
            encoding="utf-8"
        )

        monitor.write_text(
            updated,
            encoding="utf-8"
        )

    except BaseException:

        shutil.copy2(
            backup / monitor.name,
            monitor
        )

        module.unlink(
            missing_ok=True
        )

        raise

    print("\n===== UPDATE SUCCESS =====")

    print("Backup:", backup)

    print("Analysis-like detection: ENABLED")

    print("Fuzzers-like detection: ENABLED")

    print("Window: 60 seconds")

    print("Existing models unchanged")


if __name__ == "__main__":

    update(Path.cwd())
