
"""Audit live UNSW features without inventing Argus/Bro fields."""

import argparse
import json

from pathlib import Path

from src.cic_feature_extractor import (
    audit_result,
    extract_cic_features,
    flow_parts,
    stat_value,
    validate_row,
)


UNSW_FEATURES = tuple("""
dur proto service state spkts dpkts sbytes dbytes
rate sload dload sloss dloss sinpkt dinpkt sjit djit
swin stcpb dtcpb dwin tcprtt synack ackdat
smean dmean trans_depth response_body_len
ct_src_dport_ltm ct_dst_sport_ltm
is_ftp_login ct_ftp_cmd ct_flw_http_mthd
is_sm_ips_ports
""".split())


def validate_unsw_row(row):

    return validate_row(
        row,
        UNSW_FEATURES,
        {"proto", "service", "state"},
    )


def extract_unsw_features(event):

    try:

        _, f, b, duration, _ = flow_parts(event)

        values = {

            "dur": duration,

            "proto": event["protocol"],

            "spkts": f["packets"],

            "dpkts": b["packets"],

            # UNSW inter-packet arrival time
            # is expressed in milliseconds.
            "sinpkt": stat_value(
                f["iat_seconds"],
                "mean",
            ) * 1000,

            "dinpkt": stat_value(
                b["iat_seconds"],
                "mean",
            ) * 1000,

            "is_sm_ips_ports": int(
                event["src"] == event["dst"]
                and event["sport"] == event["dport"]
            ),
        }

    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:

        return audit_result(
            UNSW_FEATURES, {}, str(exc)
        )

    return audit_result(
        UNSW_FEATURES, values
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

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

    with args.events.open(
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            event = json.loads(line)

            if event.get("event") not in (
                "flow_decision",
                "flow_alert",
            ):
                continue

            result = {

                "src": event.get("src"),

                "dst": event.get("dst"),

                "flow_start_utc": event.get(
                    "flow_start_utc"
                ),

                "cic": extract_cic_features(
                    event
                ),

                "unsw": extract_unsw_features(
                    event
                ),
            }

            if not args.details:
                for branch in ("cic", "unsw"):
                    result[branch].pop(
                        "candidate_features"
                    )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )


if __name__ == "__main__":
    main()
