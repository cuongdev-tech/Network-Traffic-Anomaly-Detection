"""CPU TabR adapted from the architecture in https://arxiv.org/abs/2307.14338."""
import copy
import json
import time

import faiss
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

from src.cic_data import load_cleaned_cic
from src.unsw_data import load_and_balance_unsw, split_for_tabr, test_index_hash


CONTEXT_SIZE = 24
WIDTH = 64
BATCH_SIZE = 512


class TabR(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.encode = nn.Linear(n_features, WIDTH)
        self.key = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH))
        self.label_embedding = nn.Embedding(n_classes, WIDTH)
        nn.init.uniform_(self.label_embedding.weight, -1, 1)
        self.position = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH, bias=False))
        self.context_dropout = nn.Dropout(0.1)
        self.residual = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH * 2), nn.ReLU(),
                                      nn.Dropout(0.1), nn.Linear(WIDTH * 2, WIDTH))
        self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.ReLU(), nn.Linear(WIDTH, n_classes))

    def keys(self, features):
        return self.key(self.encode(features))

    def predict_logits(self, features, neighbor_keys, neighbor_labels):
        encoded = self.encode(features)
        difference = self.key(encoded)[:, None] - neighbor_keys
        weights = self.context_dropout((-difference.square().sum(dim=-1)).softmax(dim=1))
        values = self.label_embedding(neighbor_labels) + self.position(difference)
        combined = encoded + torch.bmm(weights[:, None], values).squeeze(1)
        return self.head(combined + self.residual(combined))


class NeighborSearch:
    def __init__(self, keys):
        self.keys = np.ascontiguousarray(keys, dtype=np.float32)
        width = self.keys.shape[1]
        self.index = faiss.IndexFlatL2(width)
        if len(keys) >= 4096:
            partitions = min(256, max(16, len(keys) // 512))
            self.index = faiss.IndexIVFFlat(self.index, width, partitions)
            self.index.cp.niter = 8
            self.index.train(self.keys)
            self.index.nprobe = min(8, partitions)
        self.index.add(self.keys)

    def search(self, queries, exclude_self=False):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        count = CONTEXT_SIZE + int(exclude_self)
        if len(self.keys) < count:
            raise ValueError(f"At least {count} reference keys are required.")
        distance, ids = self.index.search(queries, count)
        missing = (ids < 0).any(axis=1)
        if missing.any():
            exact = faiss.IndexFlatL2(self.keys.shape[1])
            exact.add(self.keys)
            distance[missing], ids[missing] = exact.search(queries[missing], count)
        if exclude_self:
            distance[ids == np.arange(len(queries))[:, None]] = np.inf
            ids = np.take_along_axis(ids, distance.argsort(axis=1, kind="stable")[:, :CONTEXT_SIZE], axis=1)
        return torch.from_numpy(ids)


@torch.inference_mode()
def encode_keys(model, features):
    return torch.cat([model.keys(batch) for batch in features.split(8192)])


def tensors(values):
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


class TabRClassifier:
    def __init__(self, model, reference, labels):
        self.model = model.eval()
        self.n_features_in_ = reference.shape[1]
        self.keys = encode_keys(self.model, reference)
        self.labels = labels
        self.search = NeighborSearch(self.keys.numpy())

    @torch.inference_mode()
    def logits(self, query):
        neighbors = self.search.search(encode_keys(self.model, query).numpy())
        output = []
        for batch, indices in zip(query.split(BATCH_SIZE), neighbors.split(BATCH_SIZE)):
            output.append(self.model.predict_logits(batch, self.keys[indices], self.labels[indices]))
        return torch.cat(output)

    def predict_proba(self, query):
        return self.logits(tensors(query)).softmax(dim=1).numpy()


def sample_reference(features, labels, size):
    if size == len(features):
        return features, labels
    sample, _, y_sample, _ = train_test_split(
        features, labels, train_size=size, stratify=labels, random_state=42,
    )
    return sample, y_sample


def load_classifier(root, dataset, bundle):
    results_dir = root / "results"
    raw_data = root / "data" / "raw"
    if dataset == "cic":
        frame = load_cleaned_cic(data_dir=raw_data / "cicids2017", verbose=False)
        labels = frame.pop("Label")
        features = frame.select_dtypes(include=[np.number])
    else:
        results_dir /= "unsw_nb15"
        features, labels = load_and_balance_unsw(data_dir=raw_data / "unsw_nb15", verbose=False)

    metadata = json.loads((results_dir / "tabr_meta.json").read_text(encoding="utf-8"))
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    reference, _, test, y_reference, _, _ = split_for_tabr(features, encoded)
    if (list(features) != bundle["features"] or encoder.classes_.tolist() != bundle["classes"]
            or not 0 < metadata["fit_train"] <= len(reference)
            or test_index_hash(test) != metadata["test_indices_sha256"]):
        raise ValueError("Dữ liệu tham chiếu không khớp với checkpoint TabR.")
    reference, y_reference = sample_reference(reference, y_reference, metadata["fit_train"])

    torch.set_num_threads(metadata["threads"])
    faiss.omp_set_num_threads(metadata["threads"])
    transformed = tensors(bundle["preprocessor"].transform(reference))
    model = TabR(bundle["input_width"], len(bundle["classes"]))
    model.load_state_dict(bundle["weights"])
    return TabRClassifier(model, transformed, torch.as_tensor(y_reference, dtype=torch.long))


def predict(model, features, labels, query):
    classifier = TabRClassifier(model, features, labels)
    return classifier.logits(query).argmax(dim=1).numpy()


def train(training, labels, validation, y_validation, *, epochs, threads, refresh_epochs=7):
    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(42)
    model = TabR(training.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0003)
    criterion = nn.CrossEntropyLoss()
    best = None
    best_weights = None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        # Refresh the cached training contexts with learned keys, then freeze their IDs.
        if epoch <= refresh_epochs and epoch % 2 == 1:
            model.eval()
            keys = encode_keys(model, training)
            contexts = NeighborSearch(keys.numpy()).search(keys.numpy(), exclude_self=True)
        model.train()
        for batch in torch.randperm(len(training), generator=order).split(BATCH_SIZE):
            indices = contexts[batch]
            unique, inverse = indices.flatten().unique(return_inverse=True)
            neighbor_keys = model.keys(training[unique])[inverse].reshape(len(batch), CONTEXT_SIZE, WIDTH)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.predict_logits(training[batch], neighbor_keys, labels[indices]), labels[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
        predicted = predict(model, training, labels, validation)
        row = {
            "epoch": epoch, "accuracy": accuracy_score(y_validation, predicted),
            "f1_macro": f1_score(y_validation, predicted, average="macro", zero_division=0),
            "seconds": round(time.perf_counter() - started, 2),
        }
        history.append(row)
        print(f"TabR epoch {epoch:02d}: validation accuracy={row['accuracy']:.5f}, "
              f"macro F1={row['f1_macro']:.5f}, {row['seconds']:.1f}s", flush=True)
        if best is None or (row["accuracy"], row["f1_macro"]) > (best["accuracy"], best["f1_macro"]):
            best = row
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best["epoch"] >= 6:
            break
    model.load_state_dict(best_weights)
    return model.eval(), best, history
