
import argparse
import copy
import json
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    QuantileTransformer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.tabr_training import (
    TabR,
    TabRClassifier,
    NeighborSearch,
    encode_keys,
    tensors,
    CONTEXT_SIZE,
    WIDTH,
    BATCH_SIZE,
)


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/unsw_realtime/netflow_v1"

MODELS = ROOT / "models/unsw_realtime"

RESULTS = ROOT / "results/unsw_realtime"

FEATURES = [
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

CLASSES = {
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
}

SEED = 42


# ==========================================
# DATA LOADING
# ==========================================

def load_data(name):

    df = pd.read_csv(DATA / f"{name}.csv")

    if set(df["target"]) != CLASSES:
        raise ValueError(
            f"{name}: expected all 10 classes"
        )

    x = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise ValueError(
            f"{name}: invalid numeric features"
        )

    if (x < 0).any():
        raise ValueError(
            f"{name}: negative features"
        )

    return df


# ==========================================
# VALIDATION SAMPLING
# ==========================================

def sample_validation(df):

    samples = []

    for label, group in df.groupby("target"):

        limit = 6000 if label == "Normal" else 1500

        n = min(limit, len(group))

        sample = group.sample(
            n=n,
            random_state=SEED,
        ).copy()

        # Restore original validation class proportions
        # when computing validation metrics.
        sample["eval_weight"] = len(group) / n

        samples.append(sample)

    return pd.concat(
        samples,
        ignore_index=True,
    )


# ==========================================
# PREPROCESSING
# ==========================================

def make_preprocessor():

    numeric = [
        name for name in FEATURES
        if name != "PROTOCOL"
    ]

    return ColumnTransformer([
        (
            "numeric",
            QuantileTransformer(
                output_distribution="normal",
                random_state=SEED,
            ),
            numeric,
        ),
        (
            "protocol",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            ["PROTOCOL"],
        ),
    ])


# ==========================================
# MODEL EVALUATION
# ==========================================

def predict_batches(classifier, query):

    output = []

    for batch in query.split(4096):

        predicted = (
            classifier.logits(batch)
            .argmax(dim=1)
            .numpy()
        )

        output.append(predicted)

    return np.concatenate(output)


# ==========================================
# TABR TRAINING
# ==========================================

def train_realtime(
    x_train,
    y_train,
    x_val,
    y_val,
    val_weights,
    normal_index,
    epochs,
    threads,
    weight_cap,
):

    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)

    torch.manual_seed(SEED)

    generator = torch.Generator().manual_seed(SEED)

    n_classes = len(CLASSES)

    model = TabR(
        x_train.shape[1],
        n_classes,
    )

    # Moderate class weighting.
    # Avoid extremely large weights for rare classes.

    counts = np.bincount(
        y_train.numpy(),
        minlength=n_classes,
    )

    weights = np.sqrt(
        counts.max() / counts
    )

    weights = np.clip(
        weights, 1.0, weight_cap
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0003,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            epochs,
            eta_min=0.0003,
        )
    )

    best = None
    best_weights = None
    history = []

    for epoch in range(1, epochs + 1):

        start = time.perf_counter()

        # Refresh retrieval contexts during
        # the first seven epochs.

        if epoch <= 7 and epoch % 2 == 1:

            model.eval()

            keys = encode_keys(
                model,
                x_train,
            )

            contexts = NeighborSearch(
                keys.numpy()
            ).search(
                keys.numpy(),
                exclude_self=True,
            )

        model.train()

        total_loss = 0.0
        total_samples = 0

        order = torch.randperm(
            len(x_train),
            generator=generator,
        )

        for batch in order.split(BATCH_SIZE):

            indices = contexts[batch]

            unique, inverse = (
                indices.flatten().unique(
                    return_inverse=True
                )
            )

            neighbor_keys = (
                model.keys(x_train[unique])[inverse]
                .reshape(
                    len(batch),
                    CONTEXT_SIZE,
                    WIDTH,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model.predict_logits(
                x_train[batch],
                neighbor_keys,
                y_train[indices],
            )

            loss = criterion(
                logits,
                y_train[batch],
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(batch)
            )

            total_samples += len(batch)

        scheduler.step()

        # Validation retrieves only from training.

        classifier = TabRClassifier(
            model,
            x_train,
            y_train,
        )

        predicted = predict_batches(
            classifier,
            x_val,
        )

        normal = y_val == normal_index

        false_positive_rate = float(
            np.average(
                predicted[normal] != normal_index,
                weights=val_weights[normal],
            )
        )

        row = {
            "epoch": epoch,

            "loss": total_loss / total_samples,

            "accuracy": float(
                accuracy_score(
                    y_val,
                    predicted,
                    sample_weight=val_weights,
                )
            ),

            "f1_macro": float(
                f1_score(
                    y_val,
                    predicted,
                    labels=np.arange(n_classes),
                    average="macro",
                    sample_weight=val_weights,
                    zero_division=0,
                )
            ),

            "false_positive_rate":
                false_positive_rate,

            "seconds": round(
                time.perf_counter() - start,
                2,
            ),
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss={row['loss']:.5f} | "
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro F1={row['f1_macro']:.4f} | "
            f"FPR={false_positive_rate:.2%} | "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

        # Select by Macro F1.
        # Lower FPR breaks ties.

        score = (
            row["f1_macro"],
            -row["false_positive_rate"],
        )

        if best is None or score > (
            best["f1_macro"],
            -best["false_positive_rate"],
        ):

            best = row.copy()

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            print(
                "  -> New best checkpoint",
                flush=True,
            )

        if epoch - best["epoch"] >= 5:

            print("Early stopping", flush=True)
            break

    model.load_state_dict(best_weights)

    return model.eval(), best, history, weights


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--weight-cap",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.threads < 1
        or not np.isfinite(args.weight_cap)
        or args.weight_cap < 1
    ):
        parser.error("Invalid training settings")

    checkpoint = MODELS / "tabr_checkpoint.joblib"

    if checkpoint.exists():
        raise SystemExit(
            "Realtime checkpoint already exists. "
            "Back it up before training again."
        )

    print("Loading NetFlow dataset...")

    train_df = load_data("train")

    validation_full = load_data("validation")

    val_df = sample_validation(validation_full)

    print("Train:", len(train_df))

    print(
        "Full validation:",
        len(validation_full),
    )

    print(
        "Sampled validation:",
        len(val_df),
    )

    del validation_full

    encoder = LabelEncoder()

    y_train = torch.as_tensor(
        encoder.fit_transform(
            train_df["target"]
        ),
        dtype=torch.long,
    )

    y_val = encoder.transform(
        val_df["target"]
    )

    classes = encoder.classes_.tolist()

    normal_index = classes.index("Normal")

    preprocessor = make_preprocessor()

    x_train = tensors(
        preprocessor.fit_transform(
            train_df[FEATURES]
        )
    )

    x_val = tensors(
        preprocessor.transform(
            val_df[FEATURES]
        )
    )

    print("\nFeatures:", FEATURES)
    print("Input width:", x_train.shape[1])
    print("Classes:", classes)

    print("\n===== START TRAINING =====")

    started = time.perf_counter()

    model, best, history, weights = (
        train_realtime(
            x_train,
            y_train,
            x_val,
            y_val,
            val_df["eval_weight"].to_numpy(),
            normal_index,
            args.epochs,
            args.threads,
            args.weight_cap,
        )
    )

    train_time = (
        time.perf_counter() - started
    )

    # Save model and reference data.

    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "architecture": "TabR",
        "dataset": "NF-UNSW-NB15-V1",
        "schema_version": "netflow_v1_7",
        "weights": model.state_dict(),
        "preprocessor": preprocessor,
        "classes": classes,
        "features": FEATURES,
        "input_width": x_train.shape[1],
        "reference_x": x_train.numpy(),
        "reference_y": y_train.numpy(),
    }

    joblib.dump(
        bundle,
        checkpoint,
        compress=3,
    )

    pd.DataFrame(history).to_csv(
        RESULTS / "training_history.csv",
        index=False,
    )

    print("\nCheckpoint saved:", checkpoint)

    # Evaluate untouched test set.

    print("\n===== FULL TEST EVALUATION =====")

    test_df = load_data("test")

    y_test = encoder.transform(
        test_df["target"]
    )

    classifier = TabRClassifier(
        model,
        x_train,
        y_train,
    )

    predictions = []

    for start in range(
        0, len(test_df), 4096
    ):

        frame = test_df.iloc[
            start:start + 4096
        ]

        query = tensors(
            preprocessor.transform(
                frame[FEATURES]
            )
        )

        predictions.append(
            classifier.logits(query)
            .argmax(dim=1)
            .numpy()
        )

        if start % 40960 == 0:

            print(
                f"Evaluated "
                f"{min(start + 4096, len(test_df))}"
                f"/{len(test_df)}",
                flush=True,
            )

    predicted = np.concatenate(
        predictions
    )

    report = classification_report(
        y_test,
        predicted,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        RESULTS / "classification_report.csv"
    )

    cm = confusion_matrix(
        y_test,
        predicted,
        labels=np.arange(len(classes)),
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        RESULTS / "confusion_matrix.csv"
    )

    normal = y_test == normal_index

    fpr = float(
        np.mean(
            predicted[normal] != normal_index
        )
    )

    metrics = {
        "dataset": "NF-UNSW-NB15-V1",

        "model": "TabR Realtime",

        "train_samples": len(train_df),

        "validation_samples": len(val_df),

        "test_samples": len(test_df),

        "features": FEATURES,

        "classes": classes,

        "selected_epoch": best["epoch"],

        "selection_metric": "validation_macro_f1",

        "class_weights": dict(
            zip(classes, weights.tolist())
        ),

        "accuracy": float(
            accuracy_score(
                y_test,
                predicted,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_test,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0,
            )
        ),

        "false_positive_rate": fpr,

        "normal_recall": 1.0 - fpr,

        "training_seconds": round(
            train_time, 2
        ),

        "best_validation": best,

        "checkpoint": str(checkpoint),
    }

    (
        RESULTS / "metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n===== TRAINING FINISHED =====")

    print(
        classification_report(
            y_test,
            predicted,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0,
        )
    )

    print(
        json.dumps(metrics, indent=2)
    )

    print("\nSaved:", checkpoint)


if __name__ == "__main__":
    main()
