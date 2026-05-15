"""Geometry utilities for linear probe domain embeddings."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from safetensors.torch import save_file
from torch import Tensor

from causalab.io.plots.mds import mds_embed


def gram_matrix(weight: Tensor, *, normalize: bool = False) -> Tensor:
    """Return W W^T, optionally after row normalization."""
    W = weight.detach().float()
    if normalize:
        W = torch.nn.functional.normalize(W, dim=-1)
    return W @ W.T


def eigendecompose_symmetric(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues/eigenvectors sorted descending."""
    evals, evecs = torch.linalg.eigh(matrix.detach().float().cpu())
    order = torch.argsort(evals, descending=True)
    return evals[order], evecs[:, order]


def dft_real_basis(n: int) -> Tensor:
    """Real orthonormal Fourier basis ordered DC, cos/sin pairs by frequency."""
    x = torch.arange(n, dtype=torch.float64)
    cols = [torch.ones(n, dtype=torch.float64) / math.sqrt(n)]
    for k in range(1, n // 2 + 1):
        cos = torch.cos(2 * math.pi * k * x / n)
        sin = torch.sin(2 * math.pi * k * x / n)
        if k == n / 2:
            cols.append(cos / cos.norm())
        else:
            cols.append(cos / cos.norm())
            cols.append(sin / sin.norm())
    return torch.stack(cols[:n], dim=1).float()


def dct_basis(n: int) -> Tensor:
    """DCT-II orthonormal basis for interval-like domains."""
    i = torch.arange(n, dtype=torch.float64)
    cols = []
    for k in range(n):
        col = torch.cos(math.pi * (i + 0.5) * k / n)
        cols.append(col / col.norm())
    return torch.stack(cols, dim=1).float()


def grid_laplacian_basis(n_nodes: int) -> Tensor:
    """Graph-Laplacian eigenbasis for a square grid, low-frequency first."""
    side = int(round(math.sqrt(n_nodes)))
    if side * side != n_nodes:
        raise ValueError(f"Grid basis requires square node count, got {n_nodes}")
    graph = nx.grid_2d_graph(side, side)
    mapping = {(r, c): r * side + c for r, c in graph.nodes}
    graph = nx.relabel_nodes(graph, mapping)
    L = nx.laplacian_matrix(graph, nodelist=list(range(n_nodes))).toarray()
    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    return torch.from_numpy(evecs[:, order]).float()


def predicted_basis(task_name: str, values: list[Any], task_config: dict[str, Any]) -> Tensor:
    """Choose a topology-predicted basis from task/domain metadata."""
    n = len(values)
    if task_name == "graph_walk":
        return grid_laplacian_basis(n)
    domain = task_config.get("domain_type")
    if domain in {"weekdays", "months", "hours"}:
        return dft_real_basis(n)
    return dct_basis(n)


def subspace_overlap(empirical: Tensor, predicted: Tensor, k: int) -> float:
    """Mean squared singular-value overlap between two top-k subspaces."""
    k = min(k, empirical.shape[1], predicted.shape[1])
    if k <= 0:
        return float("nan")
    s = torch.linalg.svdvals(empirical[:, :k].float().T @ predicted[:, :k].float())
    return float((s.pow(2).sum() / k).item())


def circulant_approximation(matrix: Tensor) -> tuple[Tensor, float]:
    """Best cyclic-diagonal average approximation and relative Frobenius error."""
    K = matrix.detach().float()
    n = K.shape[0]
    vals = []
    for shift in range(n):
        vals.append(torch.stack([K[i, (i + shift) % n] for i in range(n)]).mean())
    first_row = torch.stack(vals)
    rows = [torch.roll(first_row, shifts=i) for i in range(n)]
    C = torch.stack(rows)
    err = torch.linalg.norm(K - C) / torch.linalg.norm(K).clamp(min=1e-12)
    return C, float(err.item())


def intrinsic_dimension(eigenvalues: Tensor, threshold: float = 0.9) -> int:
    """Smallest number of eigenvalues explaining `threshold` of positive mass."""
    vals = eigenvalues.detach().float().clamp(min=0)
    total = vals.sum()
    if total <= 0:
        return 0
    csum = torch.cumsum(vals, dim=0) / total
    return int(torch.searchsorted(csum, torch.tensor(threshold)).item()) + 1


def kernel_pca_embedding(gram: Tensor, *, n_components: int | None = None) -> Tensor:
    """Embed items as sqrt(lambda_i) u_i from a positive semidefinite Gram."""
    evals, evecs = eigendecompose_symmetric(gram)
    pos = evals > 1e-8
    if n_components is not None:
        pos = pos & (torch.arange(evals.numel()) < n_components)
    return evecs[:, pos] * evals[pos].sqrt().unsqueeze(0)


def pairwise_euclidean(points: Tensor) -> Tensor:
    """Pairwise Euclidean distance matrix."""
    return torch.cdist(points.detach().float(), points.detach().float())


def probe_distance_matrices(weight: Tensor) -> dict[str, Tensor]:
    """Common distance matrices over probe domain vectors."""
    W = weight.detach().float()
    K = gram_matrix(W)
    return {
        "probe_euclidean": pairwise_euclidean(W),
        "probe_kernel_pca": pairwise_euclidean(kernel_pca_embedding(K)),
        "probe_cosine_distance": 1.0 - gram_matrix(W, normalize=True),
    }


def summarize_geometry(
    weight: Tensor,
    *,
    task_name: str,
    values: list[Any],
    task_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Compute Gram/eigenmode summary and tensor artifacts."""
    K = gram_matrix(weight)
    K_cos = gram_matrix(weight, normalize=True)
    evals, evecs = eigendecompose_symmetric(K)
    basis = predicted_basis(task_name, values, task_config)
    summary: dict[str, Any] = {
        "top2_overlap": subspace_overlap(evecs, basis, 2),
        "top4_overlap": subspace_overlap(evecs, basis, 4),
        "intrinsic_dim_90": intrinsic_dimension(evals, 0.9),
    }
    if task_config.get("domain_type") in {"weekdays", "months", "hours"}:
        _, summary["circulant_error"] = circulant_approximation(K_cos)
    tensors = {
        "gram": K,
        "cosine_gram": K_cos,
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "predicted_basis": basis,
        **probe_distance_matrices(weight),
    }
    return summary, tensors


def save_geometry_artifacts(
    output_dir: str,
    tensors: dict[str, Tensor],
    summary: dict[str, Any],
    *,
    values: list[Any],
    figure_format: str = "pdf",
) -> None:
    """Persist tensors, metadata, heatmaps, spectra, and MDS embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    save_file(
        {k: v.detach().cpu().float().contiguous() for k, v in tensors.items()},
        os.path.join(output_dir, "geometry.safetensors"),
    )
    with open(os.path.join(output_dir, "geometry.json"), "w") as f:
        json.dump({**summary, "values": [str(v) for v in values]}, f, indent=2)
    _plot_heatmap(
        tensors["cosine_gram"],
        values,
        os.path.join(output_dir, f"cosine_gram.{figure_format}"),
    )
    _plot_spectrum(tensors["eigenvalues"], os.path.join(output_dir, f"eigen_spectrum.{figure_format}"))
    for name in ("probe_euclidean", "probe_kernel_pca", "probe_cosine_distance"):
        emb = torch.from_numpy(
            mds_embed(tensors[name].detach().cpu().numpy(), n_components=2)
        ).float()
        save_file({"embedding": emb}, os.path.join(output_dir, f"{name}_mds.safetensors"))
        _plot_mds(emb, values, os.path.join(output_dir, f"{name}_mds.{figure_format}"))


def _plot_heatmap(matrix: Tensor, values: list[Any], path: str) -> None:
    labels = [str(v) for v in values]
    fig, ax = plt.subplots(
        figsize=(max(4, len(labels) * 0.35), max(3.5, len(labels) * 0.35))
    )
    im = ax.imshow(matrix.detach().cpu().numpy(), cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_spectrum(eigenvalues: Tensor, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(eigenvalues.detach().cpu().numpy(), marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_mds(embedding: Tensor, values: list[Any], path: str) -> None:
    xy = embedding.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(xy[:, 0], xy[:, 1])
    for i, label in enumerate(values):
        ax.annotate(str(label), (xy[i, 0], xy[i, 1]), fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
