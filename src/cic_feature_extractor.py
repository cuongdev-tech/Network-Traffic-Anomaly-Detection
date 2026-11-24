
"""Audit live CIC features; do not infer with unverified semantics."""
from math import isfinite

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
    """Check a reference row's schema, not its semantic provenance."""
    missing = set(schema) - row.keys()
    extra = row.keys() - set(schema)
    if missing or extra:
        raise ValueError(f"Missing: {sorted(missing)}; extra: {sorted(extra)}")
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
    values = {name: values[name] for name in schema if name in values}
    missing = [name for name in schema if name not in values]
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
    """Check basic integrity of a stage4_monitor flow event."""
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
        for key in ("payload_length", "transport_header", "ip_length"):
            if d[key]["count"] != d["packets"]:
                raise ValueError(f"Incomplete directional {key}")
        if d["iat_seconds"]["count"] != max(0, d["packets"] - 1):
            raise ValueError("Inconsistent directional IAT")
    if stats["flow_iat_seconds"]["count"] != n - 1:
        raise ValueError("Inconsistent flow IAT")
    return stats, f, b, duration, n


def stat_value(stats, key):
    # Empty observations have no measured min/mean/std.
    # Zero is a candidate convention for absent intervals/directions only.
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
            values[f"Total Length of {prefix} Packets"] = number(payload["total"])
            for label in ("Max", "Min", "Mean", "Std"):
                values[f"{prefix} Packet Length {label}"] = stat_value(payload, label.lower())
            values[f"{prefix} Header Length"] = number(d["transport_header"]["total"])
            values[f"Avg {prefix} Segment Size"] = stat_value(payload, "mean")
            for label in ("Total", "Mean", "Std", "Max", "Min"):
                values[f"{prefix} IAT {label}"] = stat_value(d["iat_seconds"], label.lower()) * 1_000_000
            if duration > 0:
                values[f"{prefix} Packets/s"] = d["packets"] / duration
        for label in ("Mean", "Std", "Max", "Min"):
            values[f"Flow IAT {label}"] = stat_value(stats["flow_iat_seconds"], label.lower()) * 1_000_000
        if duration > 0:
            values["Flow Bytes/s"] = (f["payload_length"]["total"] + b["payload_length"]["total"]) / duration
            values["Flow Packets/s"] = n / duration
        for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE"):
            values[f"{flag} Flag Count"] = f["tcp_flags"].get(flag, 0) + b["tcp_flags"].get(flag, 0)
        for prefix, d, direction in (("Fwd", f, "forward"), ("Bwd", b, "backward")):
            for flag in ("PSH", "URG"):
                values[f"{prefix} {flag} Flags"] = d["tcp_flags"].get(flag, 0)
            if d["initial_window"] is not None:
                values[f"Init_Win_bytes_{direction}"] = number(d["initial_window"])
        values["min_seg_size_forward"] = stat_value(f["transport_header"], "min")
        # No guessed Bulk/Subflow/Active/Idle or legacy aggregate size fields.
        for name in values:
            number(values[name])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return audit_result(CIC_FEATURES, {}, str(exc))
    result = audit_result(CIC_FEATURES, values)
    return attach_reference(result, event, "cic", reference, validate_cic_row)


def attach_reference(result, event, dataset, reference, validator):
    """A trusted, separately parity-tested extractor can supply an exact row.

    This verifies identity and schema only. It cannot prove semantic equivalence.
    Do not construct references from candidate_features or fill missing fields.
    """
    if reference is None or result["status"] == "INVALID_FLOW":
        return result
    try:
        if (not event.get("flow_id") or reference["flow_id"] != event["flow_id"]
                or reference["dataset"] != dataset):
            raise ValueError("Reference does not describe this flow and dataset")
        extractor = reference["extractor"]
        if not isinstance(extractor, str) or not extractor.strip():
            raise ValueError("Missing reference extractor identity")
        features = validator(reference["features"])
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
        return {**result, "status": "REFERENCE_INVALID", "error": str(error)}
    return {**result, "status": "REFERENCE_SCHEMA_OK", "model_features": features,
            "reference_extractor": extractor}
