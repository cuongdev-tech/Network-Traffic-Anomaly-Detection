"""Match independent reference features to monitor flow events."""

import argparse
import csv
import json
import os

from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

from src.cic_feature_extractor import (
    CIC_FEATURES,
    extract_cic_features,
    validate_cic_row,
)

from src.unsw_feature_extractor import (
    UNSW_FEATURES,
    extract_unsw_features,
    validate_unsw_row,
)

SPECS = {
    "cic": (
        CIC_FEATURES,
        extract_cic_features,
        validate_cic_row,
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Duration",
        1_000_000,
    ),
    "unsw": (
        UNSW_FEATURES,
        extract_unsw_features,
        validate_unsw_row,
        "spkts",
        "dpkts",
        "dur",
        1,
    ),
}

IDENTITY = (
    "src",
    "dst",
    "sport",
    "dport",
    "protocol",
    "flow_start_epoch",
    "flow_end_epoch",
)


def integer(value):

    n = Decimal(str(value))

    if not n.is_finite() or n != n.to_integral_value():
        raise ValueError(f"Not an integer: {value}")

    return int(n)


def micros(value):

    n = Decimal(str(value))

    if not n.is_finite():
        raise ValueError("Invalid timestamp")

    return int(
        (n * 1_000_000).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def flow_identity(row):

    src = ip_address(row["src"])
    dst = ip_address(row["dst"])

    sport = integer(row["sport"])
    dport = integer(row["dport"])

    protocol = str(row["protocol"]).lower()

    proto = {
        "6": "tcp",
        "17": "udp",
    }.get(protocol, protocol)

    if (
        src.version != 4
        or dst.version != 4
        or proto not in {"tcp", "udp"}
    ):
        raise ValueError(
            "Only IPv4 TCP/UDP supported"
        )

    if not (
        0 <= sport <= 65535
        and 0 <= dport <= 65535
    ):
        raise ValueError("Invalid port")

    start = micros(row["flow_start_epoch"])
    end = micros(row["flow_end_epoch"])

    if end < start:
        raise ValueError("End precedes start")

    return (
        (str(src), str(dst), sport, dport, proto),
        start,
        end,
    )


class ReferenceTable:

    def __init__(self, dataset, path, extractor):

        if not extractor or not extractor.strip():
            raise ValueError(
                "Specify extractor name and version"
            )

        self.dataset = dataset
        self.extractor = extractor
        self.index = defaultdict(list)

        schema, _, validator, fwd, bwd, duration, scale = (
            SPECS[dataset]
        )

        with Path(path).open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:

            reader = csv.DictReader(stream)

            fields = [
                name.strip()
                for name in (reader.fieldnames or [])
            ]

            if len(fields) != len(set(fields)):
                raise ValueError(
                    "Duplicate CSV columns"
                )

            missing = (
                set(IDENTITY + schema) - set(fields)
            )

            if missing:
                raise ValueError(
                    f"Missing reference columns: {sorted(missing)}"
                )

            reader.fieldnames = fields

            for row in reader:

                if None in row:
                    raise ValueError(
                        f"Malformed CSV line {reader.line_num}"
                    )

                key, start, end = flow_identity(row)

                values = validator({
                    name: row[name]
                    for name in schema
                })

                a = integer(values[fwd])
                b = integer(values[bwd])

                if a < 1 or b < 0:
                    raise ValueError(
                        "Invalid packet counts"
                    )

                if dataset == "cic":

                    if integer(
                        values["Destination Port"]
                    ) != key[3]:

                        raise ValueError(
                            "CIC destination port mismatch"
                        )

                if dataset == "unsw":

                    if values["proto"] != key[4]:

                        raise ValueError(
                            "UNSW protocol mismatch"
                        )

                reference_duration = micros(
                    Decimal(str(values[duration])) / scale
                )

                if abs(
                    reference_duration - (end - start)
                ) > 2:

                    raise ValueError(
                        "Duration/timestamp mismatch"
                    )

                self.index[key, start].append(
                    (end, a, b, values)
                )

    def match(self, event):

        key, start, end = flow_identity(event)

        stats = event["capture_features"]

        a = integer(
            stats["forward"]["packets"]
        )

        b = integer(
            stats["backward"]["packets"]
        )

        if a + b != integer(event["packets"]):
            raise ValueError(
                "Inconsistent packet counts"
            )

        matches = []

        for offset in (-1, 0, 1):

            records = self.index.get(
                (key, start + offset), ()
            )

            for stop, fwd, bwd, values in records:

                if (
                    abs(stop - end) <= 1
                    and (fwd, bwd) == (a, b)
                ):

                    matches.append(values)

        if len(matches) > 1:
            raise ValueError(
                "Ambiguous reference flow"
            )

        if not matches:
            return None

        return {
            "flow_id": event["flow_id"],
            "dataset": self.dataset,
            "extractor": self.extractor,
            "features": matches[0],
        }


@lru_cache(maxsize=2)
def reference_table(dataset, path, extractor):

    return ReferenceTable(
        dataset,
        path,
        extractor,
    )


def provide_features(event, dataset):
    """Static reference provider, not a live packet extractor."""

    if dataset not in SPECS:
        raise ValueError("Unknown dataset")

    if os.environ.get(
        "ATTT_ENABLE_REFERENCE_INFERENCE"
    ) != "1":

        return None

    prefix = f"ATTT_{dataset.upper()}"

    path = os.environ.get(
        f"{prefix}_REFERENCE_CSV"
    )

    if not path:
        return None

    extractor = os.environ.get(
        f"{prefix}_REFERENCE_EXTRACTOR",
        "",
    )

    if event.get("protocol") not in {"tcp", "udp"}:
        return None

    return reference_table(
        dataset,
        path,
        extractor,
    ).match(event)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--events",
        type=Path,
    )

    parser.add_argument(
        "--schema",
        choices=tuple(SPECS),
    )

    parser.add_argument(
        "--cic-csv",
        type=Path,
    )

    parser.add_argument(
        "--unsw-csv",
        type=Path,
    )

    parser.add_argument(
        "--cic-extractor",
        default="",
    )

    parser.add_argument(
        "--unsw-extractor",
        default="",
    )

    args = parser.parse_args()

    if args.schema:

        print(",".join(
            IDENTITY + SPECS[args.schema][0]
        ))

        return

    if not args.events:
        parser.error(
            "--events is required for audit"
        )

    tables = {}

    for dataset in SPECS:

        path = getattr(
            args,
            f"{dataset}_csv",
        )

        if path:

            tables[dataset] = ReferenceTable(
                dataset,
                path,
                getattr(
                    args,
                    f"{dataset}_extractor",
                ),
            )

    totals = {
        dataset: Counter()
        for dataset in SPECS
    }

    missing = {
        dataset: Counter()
        for dataset in SPECS
    }

    differences = {
        dataset: Counter()
        for dataset in SPECS
    }

    with args.events.open(
        encoding="utf-8"
    ) as stream:

        for line in stream:

            if not line.strip():
                continue

            event = json.loads(line)

            if event.get("event") not in {
                "flow_decision",
                "flow_alert",
            }:

                continue

            for dataset, (
                schema, extract, *_
            ) in SPECS.items():

                count = totals[dataset]

                count["flows"] += 1

                reference = None

                if (
                    dataset in tables
                    and event.get("protocol") in {"tcp", "udp"}
                ):

                    try:

                        reference = tables[
                            dataset
                        ].match(event)

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        ArithmeticError,
                    ):

                        count["match_errors"] += 1

                result = extract(
                    event,
                    reference,
                )

                count["invalid_flows"] += int(
                    result["status"] == "INVALID_FLOW"
                )

                count["complete_candidate_rows"] += int(
                    result["available_count"] == len(schema)
                )

                count["matched_reference_rows"] += int(
                    result["model_features"] is not None
                )

                missing[dataset].update(
                    result["missing_features"]
                )

                parity = result.get(
                    "reference_comparison"
                )

                if parity:

                    count["all_features_match"] += int(
                        parity["all_match"]
                    )

                    differences[dataset].update(
                        parity["different"].keys()
                    )

    report = {

        dataset: {

            **totals[dataset],

            "required_features": len(
                SPECS[dataset][0]
            ),

            "missing_features": dict(
                missing[dataset]
            ),

            "different_features": dict(
                differences[dataset]
            ),
        }

        for dataset in SPECS
    }

    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ))

    print(
        "Audit only: no model was called; "
        "no extractor was automatically approved."
    )


if __name__ == "__main__":
    main()
