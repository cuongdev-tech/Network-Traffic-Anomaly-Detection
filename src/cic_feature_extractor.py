"""Audit live CIC features; do not infer with unverified semantics."""
from math import isfinite, isclose

CIC_FEATURES = tuple("""
Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
Min Packet Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
CWE Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Avg Fwd Segment Size
Avg Bwd Segment Size
Fwd Avg Bytes/Bulk
Fwd Avg Packets/Bulk
Fwd Avg Bulk Rate
Bwd Avg Bytes/Bulk
Bwd Avg Packets/Bulk
Bwd Avg Bulk Rate
Subflow Fwd Packets
Subflow Fwd Bytes
Subflow Bwd Packets
Subflow Bwd Bytes
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
""".strip().splitlines())


def number(value):

    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric feature")

    n = float(value)

    if not isfinite(n):
        raise ValueError("Nonfinite feature")

    return n


def validate_row(row, schema, categorical=()):

    missing = set(schema) - row.keys()
    extra = row.keys() - set(schema)

    if missing or extra:

        raise ValueError(
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )

    result = {}

    for name in schema:

        value = row[name]

        if name in categorical:

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid category: {name}")

            result[name] = value.strip()

        else:

            result[name] = number(value)

    return result


def validate_cic_row(row):

    return validate_row(row, CIC_FEATURES)


def audit_result(schema, values, error=None):

    values = {
        name: values[name]
        for name in schema
        if name in values
    }

    missing = [
        name for name in schema
        if name not in values
    ]

    return {
        "status": "INVALID_FLOW" if error else "UNVERIFIED",
        "candidate_features": values,
        "available_count": len(values),
        "required_count": len(schema),
        "missing_features": missing,
        "error": error,
        "model_features": None,
    }


def flow_parts(event):

    if event["event"] not in ("flow_decision", "flow_alert"):
        raise ValueError("Not a flow event")

    if event["protocol"] not in ("tcp", "udp"):
        raise ValueError("Extended metadata supports TCP/UDP only")

    stats = event["capture_features"]

    if stats["missing_metadata"] or stats["out_of_order"]:
        raise ValueError("Incomplete or out-of-order metadata")

    f, b = stats["forward"], stats["backward"]

    n = event["packets"]
    duration = number(event["duration_s"])

    counts = (n, f["packets"], b["packets"])

    if any(type(count) is not int for count in counts):
        raise ValueError("Packet counts must be integers")

    if f["packets"] < 1 or b["packets"] < 0:
        raise ValueError("Invalid directional counts")

    if f["packets"] + b["packets"] != n or duration < 0:
        raise ValueError("Inconsistent flow counts or duration")

    for d in (f, b):

        for key in (
            "payload_length",
            "transport_header",
            "ip_length",
        ):

            if d[key]["count"] != d["packets"]:
                raise ValueError(f"Incomplete directional {key}")

        if d["iat_seconds"]["count"] != max(0, d["packets"] - 1):
            raise ValueError("Inconsistent directional IAT")

    if stats["flow_iat_seconds"]["count"] != n - 1:
        raise ValueError("Inconsistent flow IAT")

    return stats, f, b, duration, n


def stat_value(stats, key):

    return 0.0 if stats["count"] == 0 else number(stats[key])


def extract_cic_features(event, reference=None):

    values = {}

    try:

        stats, f, b, duration, n = flow_parts(event)

        values.update({
            "Destination Port": event["dport"],
            "Flow Duration": round(duration * 1_000_000),
            "Total Fwd Packets": f["packets"],
            "Total Backward Packets": b["packets"],
            "Down/Up Ratio": b["packets"] // f["packets"],
        })

        for prefix, d in (("Fwd", f), ("Bwd", b)):

            payload = d["payload_length"]

            values[f"Total Length of {prefix} Packets"] = number(
                payload["total"]
            )

            for label in ("Max", "Min", "Mean", "Std"):

                values[f"{prefix} Packet Length {label}"] = stat_value(
                    payload, label.lower()
                )

            values[f"{prefix} Header Length"] = number(
                d["transport_header"]["total"]
            )

            values[f"Avg {prefix} Segment Size"] = stat_value(
                payload, "mean"
            )

            for label in ("Total", "Mean", "Std", "Max", "Min"):

                values[f"{prefix} IAT {label}"] = stat_value(
                    d["iat_seconds"], label.lower()
                ) * 1_000_000

            if duration > 0:

                values[f"{prefix} Packets/s"] = (
                    d["packets"] / duration
                )

        for label in ("Mean", "Std", "Max", "Min"):

            values[f"Flow IAT {label}"] = stat_value(
                stats["flow_iat_seconds"], label.lower()
            ) * 1_000_000

        if duration > 0:

            values["Flow Bytes/s"] = (
                f["payload_length"]["total"]
                + b["payload_length"]["total"]
            ) / duration

            values["Flow Packets/s"] = n / duration

        for flag in (
            "FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE"
        ):

            values[f"{flag} Flag Count"] = (
                f["tcp_flags"].get(flag, 0)
                + b["tcp_flags"].get(flag, 0)
            )

        for prefix, d, direction in (
            ("Fwd", f, "forward"),
            ("Bwd", b, "backward"),
        ):

            for flag in ("PSH", "URG"):

                values[f"{prefix} {flag} Flags"] = (
                    d["tcp_flags"].get(flag, 0)
                )

            if d["initial_window"] is not None:

                values[f"Init_Win_bytes_{direction}"] = number(
                    d["initial_window"]
                )

        values["min_seg_size_forward"] = stat_value(
            f["transport_header"], "min"
        )

        if stats.get("schema_version") == 2:

            add_extended_cic(
                values, event, stats, f, b, n
            )

        for name in values:
            number(values[name])

    except (
        KeyError, TypeError, ValueError, OverflowError
    ) as exc:

        return audit_result(
            CIC_FEATURES, {}, str(exc)
        )

    result = audit_result(CIC_FEATURES, values)

    if stats.get("schema_version") == 2:

        result["candidate_definition"] = (
            "local_v2_with_legacy_cic_indicators_unverified"
        )

    return attach_reference(
        result, event, "cic", reference, validate_cic_row
    )


def add_extended_cic(values, event, stats, f, b, n):

    from src.live_flow_stats import FLAGS

    payload = stats["legacy_payload_length"]

    if payload["count"] != n + 1:
        raise ValueError("Legacy payload statistics incomplete")

    for key, field in (
        ("Min Packet Length", "min"),
        ("Max Packet Length", "max"),
        ("Packet Length Mean", "mean"),
        ("Packet Length Std", "std"),
    ):

        values[key] = stat_value(payload, field)

    values["Packet Length Variance"] = (
        values["Packet Length Std"] ** 2
    )

    values["Average Packet Size"] = payload["total"] / n

    first_flags = f["first_flags"]

    for flag, bit in FLAGS.items():

        name = "CWE" if flag == "CWR" else flag

        values[f"{name} Flag Count"] = int(
            bool(first_flags & bit)
        )

    values["act_data_pkt_fwd"] = (
        f["data_packets"]
        - int(stats["first_payload_bytes"] > 0)
    )

    groups = stats["subflows"]

    if type(groups) is not int or groups < 1:
        raise ValueError("Invalid local subflow count")

    for prefix, d, direction in (
        ("Fwd", f, "forward"),
        ("Bwd", b, "backward"),
    ):

        bulk = d["bulk"]
        count = bulk["groups"]

        values[f"{prefix} Avg Bytes/Bulk"] = (
            bulk["bytes"] / count if count else 0
        )

        values[f"{prefix} Avg Packets/Bulk"] = (
            bulk["packets"] / count if count else 0
        )

        if not count or bulk["duration_s"] > 0:

            values[f"{prefix} Avg Bulk Rate"] = (
                bulk["bytes"] / bulk["duration_s"]
                if count else 0
            )

        values[f"Subflow {prefix} Packets"] = (
            d["packets"] // groups
        )

        values[f"Subflow {prefix} Bytes"] = (
            int(d["payload_length"]["total"]) // groups
        )

        if d["initial_window"] is None:

            values[f"Init_Win_bytes_{direction}"] = -1

    for prefix, key in (
        ("Active", "active_seconds"),
        ("Idle", "idle_seconds"),
    ):

        for label in ("Mean", "Std", "Max", "Min"):

            values[f"{prefix} {label}"] = stat_value(
                stats[key], label.lower()
            ) * 1_000_000


def compare_reference(candidate, reference):

    matched = []
    different = {}
    missing = []

    for name, expected in reference.items():

        if name not in candidate:

            missing.append(name)
            continue

        actual = candidate[name]

        if isinstance(expected, str):

            equal = actual == expected

        else:

            equal = isclose(
                number(actual),
                number(expected),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )

        if equal:

            matched.append(name)

        else:

            different[name] = {
                "candidate": actual,
                "reference": expected,
            }

    return {
        "matched": matched,
        "different": different,
        "missing": missing,
        "all_match": not different and not missing,
    }


def attach_reference(result, event, dataset, reference, validator):
    """Accept only trusted, independently parity-tested reference features."""

    if reference is None or result["status"] == "INVALID_FLOW":
        return result

    try:

        if (
            not event.get("flow_id")
            or reference["flow_id"] != event["flow_id"]
            or reference["dataset"] != dataset
        ):

            raise ValueError(
                "Reference does not describe this flow and dataset"
            )

        extractor = reference["extractor"]

        if not isinstance(extractor, str) or not extractor.strip():
            raise ValueError("Missing reference extractor identity")

        features = validator(reference["features"])

    except (
        KeyError, TypeError, ValueError,
        OverflowError, AttributeError
    ) as error:

        return {
            **result,
            "status": "REFERENCE_INVALID",
            "error": str(error),
        }

    return {
        **result,
        "status": "REFERENCE_SCHEMA_OK",
        "model_features": features,
        "reference_extractor": extractor,
        "reference_comparison": compare_reference(result["candidate_features"], features),
    }
