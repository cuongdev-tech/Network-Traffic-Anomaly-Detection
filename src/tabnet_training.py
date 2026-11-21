"""TabNet preprocessing and training; selection uses validation, never test."""
import time

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, QuantileTransformer


def reserve_unknown_category(values):
    """Category zero is reserved for values not seen during fit."""
    return values + 1


def make_preprocessor(frame):
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [name for name in frame if name not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("quantile", QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                             subsample=100000, random_state=42)),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("reserve_unknown", FunctionTransformer(reserve_unknown_category)),
        ]), categorical),
    ])


def split_validation(frame, labels):
    return train_test_split(frame, labels, test_size=0.1, random_state=42, stratify=labels)


def validation_scores(model, preprocessor, frame, labels):
    predicted = model.predict(preprocessor.transform(frame).astype(np.float32))
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0))}


def train_tabnet(frame, labels, max_epochs=80, threads=4):
    if max_epochs < 1 or threads < 1:
        raise ValueError("max_epochs and threads must be positive")
    torch.set_num_threads(threads)
    training, validation, y_training, y_validation = split_validation(frame, labels)
    preprocessor = make_preprocessor(training)
    X_training = preprocessor.fit_transform(training).astype(np.float32)
    X_validation = preprocessor.transform(validation).astype(np.float32)
    numeric_count = len(preprocessor.transformers_[0][2])
    categories = preprocessor.named_transformers_["cat"].named_steps["encoder"].categories_
    model = TabNetClassifier(
        n_d=24, n_a=24, n_steps=4, gamma=1.3, lambda_sparse=1e-5,
        cat_idxs=list(range(numeric_count, X_training.shape[1])),
        cat_dims=[len(values) + 1 for values in categories], cat_emb_dim=4,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": 0.015, "weight_decay": 1e-5},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.5},
        mask_type="entmax", seed=42, verbose=1, device_name="cpu",
    )
    started = time.perf_counter()
    model.fit(
        X_training, y_training,
        eval_set=[(X_validation, y_validation)], eval_name=["val"],
        eval_metric=["balanced_accuracy", "accuracy"],
        max_epochs=max_epochs, patience=15, batch_size=2048, virtual_batch_size=256,
        num_workers=0, drop_last=False,
    )
    metadata = {
        "fit_train": len(training), "validation": len(validation),
        "preprocessing": "Numeric quantiles + categorical embeddings; fitted on inner train only",
        "selection_metric": "validation accuracy", "max_epochs": max_epochs,
        "best_epoch": int(model.best_epoch), "epochs_run": len(model.history["loss"]),
        "batch_size": 2048, "virtual_batch_size": 256,
        "n_d": 24, "n_a": 24, "n_steps": 4,
        "validation_scores": validation_scores(model, preprocessor, validation, y_validation),
        "train_time_s": round(time.perf_counter() - started, 2),
    }
    return model, preprocessor, metadata
