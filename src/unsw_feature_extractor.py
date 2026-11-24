"""Audit live UNSW features without inventing Argus/Bro fields."""
import argparse
import json
from pathlib import Path

from src.cic_feature_extractor import (
    attach_reference,
    audit_result,
    extract_cic_features,
    flow_parts,
    number,
    stat_value,
    validate_row,
)

UNSW_FEATURES = tuple("""
dur proto service state spkts dpkts sbytes dbytes rate sload dload
sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt
synack ackdat smean dmean trans_depth response_body_len ct_src_dport_ltm
ct_dst_sport_ltm is_ftp_login ct_ftp_cmd ct_flw_http_mthd is_sm_ips_ports
""".split())


def validate_unsw_row(row):

    return validate_row(
        row,
        UNSW_FEATURES,
        {"proto", "service", "state"},
    )


def extract_unsw_features(event, reference=None):

    try:

        stats, f, b, duration, n = flow_parts(event)

        values = {
            "dur": duration,
            "proto": event["protocol"],
            "spkts": f["packets"],
            "dpkts": b["packets"],

            "sinpkt": stat_value(
                f["iat_seconds"], "mean"
            ) * 1000,

            "dinpkt": stat_value(
                b["iat_seconds"], "mean"
            ) * 1000,

            "is_sm_ips_ports": int(
                event["src"] == event["dst"]
                and event["sport"] == event["dport"]
            ),
        }

        if stats.get("schema_version") == 2:

            for prefix, direction in (("s", f), ("d", b)):

                total = number(
                    direction["ip_length"]["total"]
                )

                packets = direction["packets"]

                values[f"{prefix}bytes"] = total

                values[f"{prefix}mean"] = (
                    int(total / packets + 0.5)
                    if packets else 0
                )

                if event["protocol"] == "tcp":

                    if direction["initial_window"] is not None:

                        values[f"{prefix}win"] = number(
                            direction["initial_window"]
                        )

                    if direction["initial_syn_seq"] is not None:

                        values[f"{prefix}tcpb"] = number(
                            direction["initial_syn_seq"]
                        )

                else:

                    values[f"{prefix}win"] = 0
                    values[f"{prefix}tcpb"] = 0

            if duration > 0:

                values["rate"] = (n - 1) / duration

            if event["protocol"] == "udp":

                values.update(
                    tcprtt=0,
                    synack=0,
                    ackdat=0,
                )

            elif stats["handshake"] is not None:

                for key in ("tcprtt", "synack", "ackdat"):

                    values[key] = number(
                        stats["handshake"][key]
                    )

            history = stats.get("history")

            if (
                history
                and history["scope"] == "local_last_100_closed_flows"
            ):

                for key in (
                    "ct_src_dport_ltm",
                    "ct_dst_sport_ltm",
                ):

                    values[key] = number(history[key])

            # Do not substitute local heuristics for Argus state,
            # service, loss, load, jitter or HTTP/FTP statistics.

    except (
        KeyError, TypeError, ValueError, OverflowError
    ) as exc:

        return audit_result(
            UNSW_FEATURES, {}, str(exc)
        )

    local = stats.get("unsw_local", {})

    allowed = {
        "service", "state",
        "sload", "dload",
        "sloss", "dloss",
        "sjit", "djit",
        "trans_depth",
        "response_body_len",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
    }

    for name, value in local.items():

        if name in allowed:

            if name in {"service", "state"}:

                if isinstance(value, str) and value:
                    values[name] = value

            else:

                try:

                    values[name] = number(value)

                except (TypeError, ValueError, OverflowError):

                    pass
    result = audit_result(UNSW_FEATURES, values)

    if stats.get("schema_version") == 2:

        result["candidate_definition"] = (
            "local_v2_ip_bytes_handshake_history_unverified"
        )

    sources = {
        "service": "Application protocol identification",
        "state": "Argus connection state",
        "sloss": "Argus loss accounting",
        "dloss": "Argus loss accounting",
        "sload": "Argus byte and load definition",
        "dload": "Argus byte and load definition",
        "sjit": "Argus jitter definition",
        "djit": "Argus jitter definition",
        "trans_depth": "HTTP transaction parsing",
        "response_body_len": "HTTP response parsing",
        "is_ftp_login": "FTP session parsing",
        "ct_ftp_cmd": "FTP command history",
        "ct_flw_http_mthd": "HTTP method history",
    }

    result["missing_sources"] = {
        name: sources.get(
            name,
            "Missing packet or timing evidence",
        )
        for name in result["missing_features"]
    }

    if local:

        result["candidate_definition"] = (
            "local_v3_unverified_not_argus_bro"
        )

        result["local_estimates"] = sorted(
            set(local) & allowed & values.keys()
        )
    return attach_reference(
        result,
        event,
        "unsw",
        reference,
        validate_unsw_row,
    )


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--events",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--details",
        action="store_true",
    )

    args = parser.parse_args()

    with args.events.open(encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            event = json.loads(line)

            if event.get("event") not in (
                "flow_decision", "flow_alert"
            ):
                continue

            result = {
                "src": event.get("src"),
                "dst": event.get("dst"),
                "flow_start_utc": event.get("flow_start_utc"),

                "cic": extract_cic_features(event),

                "unsw": extract_unsw_features(event),
            }

            if not args.details:

                for branch in ("cic", "unsw"):

                    result[branch].pop(
                        "candidate_features"
                    )

            print(json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
            ))


if __name__ == "__main__":
    main()
