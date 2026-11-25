
import argparse
from pathlib import Path
from collections import Counter

import faiss
import joblib
import numpy as np
import pandas as pd
import torch

from scapy.utils import PcapReader

from src.monitor_flows import (
    FlowTable,
    packet_observation,
)

from src.cic_feature_extractor import flow_parts
from src.live_flow_stats import FLAGS

from src.tabr_training import (
    TabR,
    TabRClassifier,
    tensors,
)


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "models/unsw_realtime/tabr_checkpoint.joblib"
)

FEATURES = [
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]


# ==========================================
# NETFLOW FEATURE EXTRACTION
# ==========================================

def extract_netflow_features(event):

    stats, forward, backward, duration, n = (
        flow_parts(event)
    )

    if stats.get("schema_version") != 2:
        raise ValueError("Unsupported flow schema")

    protocol = {
        "tcp": 6,
        "udp": 17,
    }[event["protocol"]]

    tcp_flags = 0

    if protocol == 6:

        for name, bit in FLAGS.items():

            if (
                forward["tcp_flags"].get(name, 0)
                or backward["tcp_flags"].get(name, 0)
            ):

                tcp_flags |= bit

    return {
        "PROTOCOL": protocol,

        "IN_BYTES": int(
            forward["ip_length"]["total"]
        ),

        "OUT_BYTES": int(
            backward["ip_length"]["total"]
        ),

        "IN_PKTS": forward["packets"],

        "OUT_PKTS": backward["packets"],

        "TCP_FLAGS": tcp_flags,

        "FLOW_DURATION_MILLISECONDS": int(
            round(duration * 1000)
        ),
    }


# ==========================================
# LOAD REALTIME TABR
# ==========================================

class NetFlowTabR:

    def __init__(self, threads=2):

        torch.set_num_threads(threads)
        faiss.omp_set_num_threads(threads)

        bundle = joblib.load(CHECKPOINT)

        if (
            bundle.get("schema_version") != "netflow_v1_7"
            or list(bundle["features"]) != FEATURES
        ):
            raise ValueError("Checkpoint schema mismatch")

        self.classes = bundle["classes"]

        self.preprocessor = bundle["preprocessor"]

        reference = tensors(
            bundle["reference_x"]
        )

        labels = torch.as_tensor(
            bundle["reference_y"],
            dtype=torch.long,
        )

        model = TabR(
            bundle["input_width"],
            len(self.classes),
        )

        model.load_state_dict(
            bundle["weights"]
        )

        self.classifier = TabRClassifier(
            model,
            reference,
            labels,
        )

    def predict(self, features):

        frame = pd.DataFrame(
            [features],
            columns=FEATURES,
        )

        query = np.asarray(
            self.preprocessor.transform(frame),
            dtype=np.float32,
        )

        probabilities = (
            self.classifier.predict_proba(query)[0]
        )

        index = int(
            np.argmax(probabilities)
        )

        return {
            "label": self.classes[index],
            "confidence": float(probabilities[index]),
        }


# ==========================================
# OFFLINE PCAP TEST
# ==========================================

def test_pcap(path, src, dst, port):

    model = NetFlowTabR()

    table = FlowTable(
        idle_seconds=5,
        active_seconds=60,
        max_flows=10000,
    )

    counts = Counter()

    total = 0
    invalid = 0

    def consume(completed):

        nonlocal total, invalid

        for flow, reason in completed:

            first = flow.first

            # Evaluate only the controlled
            # normal HTTP test traffic.

            if (
                first.src != src
                or first.dst != dst
                or first.dport != port
                or first.protocol != "tcp"
            ):
                continue

            total += 1

            event = {
                "event": "flow_decision",
                "src": first.src,
                "dst": first.dst,
                "sport": first.sport,
                "dport": first.dport,
                "protocol": first.protocol,
                "packets": flow.packets,
                "duration_s": round(
                    flow.duration_s, 6
                ),
                "capture_features":
                    flow.stats.snapshot(),
            }

            try:

                features = extract_netflow_features(
                    event
                )

            except (ValueError, KeyError, TypeError) as exc:

                invalid += 1

                print("Invalid flow:", exc)

                continue

            result = model.predict(features)

            label = result["label"]

            counts[label] += 1

            print(
                f"Flow {total:02d}: "
                f"{label} "
                f"({result['confidence']:.2%})"
            )

    with PcapReader(str(path)) as packets:

        for packet in packets:

            observation = packet_observation(
                packet
            )

            if observation is not None:

                consume(
                    table.ingest(observation)
                )

    consume(table.finish())

    predicted = sum(counts.values())

    print("\n===== REALTIME MODEL TEST =====")

    print("Matched flows:", total)

    print("Predicted flows:", predicted)

    print("Invalid flows:", invalid)

    print("Predictions:", dict(counts))

    print("Capacity drops:", table.capacity_drops)

    print("Late packets:", table.late_packets)

    if predicted:

        print(
            f"Normal recall: "
            f"{counts['Normal']/predicted:.2%}"
        )

        print(
            f"False positive rate: "
            f"{1-counts['Normal']/predicted:.2%}"
        )

    else:

        raise SystemExit("No valid matching flows")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pcap",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--src",
        default="192.168.83.1",
    )

    parser.add_argument(
        "--dst",
        default="192.168.83.129",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    args = parser.parse_args()

    test_pcap(
        args.pcap,
        args.src,
        args.dst,
        args.port,
    )
