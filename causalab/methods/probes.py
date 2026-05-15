"""Linear probe utilities for feature-geometry analyses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass
class ProbeResult:
    weight: Tensor
    train_indices: Tensor
    test_indices: Tensor
    metrics: dict[str, Any]


def stratified_split(
    labels: Tensor,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return train/test indices with each class split independently."""
    labels_np = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for cls in sorted(set(labels_np.tolist())):
        idx = np.flatnonzero(labels_np == cls)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        if len(idx) > 1:
            n_train = min(max(n_train, 1), len(idx) - 1)
        train.extend(idx[:n_train].tolist())
        test.extend(idx[n_train:].tolist())
    rng.shuffle(train)
    rng.shuffle(test)
    return torch.tensor(train, dtype=torch.long), torch.tensor(test, dtype=torch.long)


def labels_from_examples(examples: list[Any], task: Any, n: int | None = None) -> Tensor:
    """Map counterfactual examples to task class indices."""
    rows = examples if n is None else examples[:n]
    return torch.tensor(
        [int(task.intervention_value_index(ex)) for ex in rows],
        dtype=torch.long,
    )


def train_multiclass_probe(
    features: Tensor,
    labels: Tensor,
    *,
    n_classes: int | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    batch_size: int | None = None,
    shuffle_labels: bool = False,
) -> ProbeResult:
    """Train a no-bias multiclass logistic regression probe."""
    X = features.detach().float().cpu()
    y = labels.detach().long().cpu()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features/labels length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if n_classes is None:
        n_classes = int(y.max().item()) + 1
    train_idx, test_idx = stratified_split(y, train_frac=train_frac, seed=seed)
    if shuffle_labels:
        gen = torch.Generator().manual_seed(seed + 17)
        y = y[torch.randperm(y.numel(), generator=gen)]

    torch.manual_seed(seed)
    model = torch.nn.Linear(X.shape[1], n_classes, bias=False)
    torch.nn.init.normal_(model.weight, mean=0.0, std=0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if batch_size is None or batch_size <= 0:
        batch_size = max(1, train_idx.numel())

    for _ in range(int(epochs)):
        perm = train_idx[torch.randperm(train_idx.numel())]
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    W = model.weight.detach().clone()
    metrics = probe_metrics(W, X, y, train_idx=train_idx, test_idx=test_idx)
    return ProbeResult(W, train_idx, test_idx, metrics)


def probe_metrics(
    weight: Tensor,
    features: Tensor,
    labels: Tensor,
    *,
    train_idx: Tensor | None = None,
    test_idx: Tensor | None = None,
) -> dict[str, Any]:
    """Accuracy, per-class accuracy, and cross entropy for a trained probe."""
    X = features.detach().float().cpu()
    y = labels.detach().long().cpu()
    W = weight.detach().float().cpu()
    logits = X @ W.T
    pred = logits.argmax(dim=-1)
    n_classes = W.shape[0]

    def _acc(idx: Tensor | None) -> float:
        if idx is None or idx.numel() == 0:
            return float("nan")
        return float((pred[idx] == y[idx]).float().mean().item())

    eval_idx = test_idx if test_idx is not None and test_idx.numel() else torch.arange(y.numel())
    per_class: dict[str, float] = {}
    for cls in range(n_classes):
        mask = eval_idx[y[eval_idx] == cls]
        per_class[str(cls)] = _acc(mask)
    return {
        "train_accuracy": _acc(train_idx),
        "test_accuracy": _acc(test_idx),
        "accuracy": float((pred == y).float().mean().item()),
        "per_class_accuracy": per_class,
        "cross_entropy": float(torch.nn.functional.cross_entropy(logits, y).item()),
        "n_train": int(train_idx.numel()) if train_idx is not None else 0,
        "n_test": int(test_idx.numel()) if test_idx is not None else 0,
    }


def save_probe(
    output_dir: str,
    result: ProbeResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save probe tensors and sibling JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    save_file(
        {
            "weight": result.weight.contiguous().float(),
            "train_indices": result.train_indices.contiguous(),
            "test_indices": result.test_indices.contiguous(),
        },
        os.path.join(output_dir, "probe.safetensors"),
    )
    meta = dict(metadata or {})
    meta["metrics"] = result.metrics
    with open(os.path.join(output_dir, "probe.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_probe(path_or_dir: str) -> tuple[Tensor, dict[str, Any]]:
    """Load a saved probe weight and metadata from a directory or tensor path."""
    tensor_path = (
        path_or_dir
        if path_or_dir.endswith(".safetensors")
        else os.path.join(path_or_dir, "probe.safetensors")
    )
    W = load_file(tensor_path)["weight"]
    meta_path = os.path.join(os.path.dirname(tensor_path), "probe.meta.json")
    metadata: dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return W, metadata


def find_probe_dir(
    root: str,
    subspace_sub: str,
    *,
    target_variable: str | None = None,
    layer: int | None = None,
    token_position: str | None = None,
    feature_space: str | None = "activation",
    feature_geometry_subdir: str | None = None,
) -> str | None:
    """Find the feature_geometry probe directory matching a path-steering cell."""
    base = os.path.join(root, "feature_geometry")
    if feature_geometry_subdir:
        base = os.path.join(base, feature_geometry_subdir)
    base = os.path.join(base, subspace_sub)
    if target_variable:
        base = os.path.join(base, target_variable)
    if not os.path.isdir(base):
        return None

    candidates: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(base):
        if "probe.safetensors" not in filenames:
            continue
        meta_path = os.path.join(dirpath, "probe.meta.json")
        meta: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        if layer is not None and meta.get("layer") not in (None, layer):
            continue
        if token_position is not None and meta.get("token_position") not in (
            None,
            token_position,
        ):
            continue
        if feature_space is not None and meta.get("feature_space", "activation") not in (
            None,
            feature_space,
        ):
            continue
        candidates.append(dirpath)
    return sorted(candidates)[0] if candidates else None
