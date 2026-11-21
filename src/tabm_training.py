"""TabM training and probability averaging for network-flow features."""
import copy
import time

import numpy as np
import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from sklearn.metrics import accuracy_score, f1_score
from tabm import TabM


class TabMClassifier:
    """Expose the fitted CPU network through the dashboard's prediction API."""

    def __init__(self, network, numeric_indices, numeric_width, n_features):
        self.network = network.eval()
        self.numeric_indices = numeric_indices
        self.numeric_width = numeric_width
        self.n_features_in_ = n_features

    @torch.inference_mode()
    def predict_proba(self, frame):
        features = torch.as_tensor(np.ascontiguousarray(frame, dtype=np.float32))
        numeric = features[:, self.numeric_indices]
        categorical = features[:, self.numeric_width:].long()
        return self.probabilities(numeric, categorical).numpy()

    def probabilities(self, numeric, categorical):
        probabilities = []
        for numbers, categories in zip(numeric.split(2048), categorical.split(2048)):
            logits = self.network(numbers, categories if categories.shape[1] else None)
            probabilities.append(logits.softmax(dim=-1).mean(dim=1))
        return torch.cat(probabilities)

    def predict(self, frame):
        return self.predict_proba(frame).argmax(axis=1)


def train_candidates(training, y_train, validation, y_val, numeric_width,
                     cardinalities, max_epochs, selection_metric, threads):
    """Train two class-weighting options and select using validation only."""
    if threads < 1 or max_epochs < 1:
        raise ValueError("Epochs and threads must be positive.")
    torch.set_num_threads(threads)
    x_train = torch.from_numpy(np.ascontiguousarray(training, dtype=np.float32))
    x_val = torch.from_numpy(np.ascontiguousarray(validation, dtype=np.float32))
    labels = torch.as_tensor(y_train, dtype=torch.long)
    n_classes = int(labels.max()) + 1
    numeric_indices = torch.where(
        x_train[:, :numeric_width].amax(0) != x_train[:, :numeric_width].amin(0)
    )[0]
    categories = x_train[:, numeric_width:].long()
    val_categories = x_val[:, numeric_width:].long()
    numeric = x_train[:, numeric_indices].contiguous()
    val_numeric = x_val[:, numeric_indices].contiguous()
    bins = compute_bins(
        numeric, n_bins=32, y=labels, regression=False,
        tree_kwargs={"min_samples_leaf": 48, "random_state": 42},
    )
    counts = torch.bincount(labels).float()
    batch_size = 1024
    summaries = []
    best_candidate = None
    best_model = None

    def score(record):
        secondary = "f1_macro" if selection_metric == "accuracy" else "accuracy"
        return record[selection_metric], record[secondary]

    for power in (0.0, 0.25):
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)
        network = TabM.make(
            n_num_features=numeric.shape[1], cat_cardinalities=cardinalities,
            num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=8, activation=False, version="B"),
            d_out=n_classes, arch_type="tabm-mini", k=4, n_blocks=3,
            d_block=128, dropout=0.05,
        )
        model = TabMClassifier(network, numeric_indices, numeric_width, x_train.shape[1])
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.0003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=0.0002)
        loss_fn = torch.nn.CrossEntropyLoss(weight=counts.pow(-power))
        started = time.perf_counter()
        best_epoch = None
        best_state = None
        for epoch in range(1, max_epochs + 1):
            network.train()
            for batch in torch.randperm(len(labels), generator=generator).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = network(numeric[batch], categories[batch] if cardinalities else None)
                targets = labels[batch].repeat_interleave(network.k)
                loss = loss_fn(logits.flatten(0, 1), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            network.eval()
            with torch.inference_mode():
                predictions = model.probabilities(val_numeric, val_categories).argmax(dim=1).numpy()
            record = {
                "epoch": epoch, "class_weight_power": power,
                "accuracy": accuracy_score(y_val, predictions),
                "f1_macro": f1_score(y_val, predictions, average="macro", zero_division=0),
            }
            print(
                f"weight={power:g} epoch={epoch:02d} "
                f"val accuracy={record['accuracy']:.5f} val macro F1={record['f1_macro']:.5f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if best_epoch is None or score(record) > score(best_epoch):
                best_epoch = record
                best_state = copy.deepcopy(network.state_dict())
            if epoch - best_epoch["epoch"] >= 6:
                break
        best_epoch.update(epochs_run=epoch, train_time_s=round(time.perf_counter() - started, 2))
        summaries.append(best_epoch)
        if best_candidate is None or score(best_epoch) > score(best_candidate):
            network.load_state_dict(best_state)
            best_model = model
            best_candidate = best_epoch
    return best_model, best_candidate, summaries
