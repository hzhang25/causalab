"""Probe-softmax dual steering primitives."""

from __future__ import annotations

import torch
from torch import Tensor


def additive_probe_path(
    h_start: Tensor,
    h_end: Tensor,
    w_start: Tensor,
    w_end: Tensor,
    n_steps: int,
    *,
    scale_to_endpoint: bool = True,
) -> Tensor:
    """Build a raw-activation path along a probe-vector difference."""
    beta = (w_end - w_start).to(device=h_start.device, dtype=h_start.dtype)
    if scale_to_endpoint:
        target_delta = h_end - h_start
        scale = target_delta.dot(beta) / beta.dot(beta).clamp(min=1e-12)
        beta = beta * scale
    alphas = torch.linspace(0, 1, n_steps, device=h_start.device, dtype=h_start.dtype)
    return h_start.unsqueeze(0) + alphas.unsqueeze(1) * beta.unsqueeze(0)


def dual_steer_path(
    h0: Tensor,
    target_class: int,
    beta: Tensor,
    probe_W: Tensor,
    *,
    n_steps: int = 50,
    eta: float = 0.01,
    alpha: float = 1e-3,
    target_prob: float | None = None,
    normalize_step: bool = True,
) -> Tensor:
    """Park-style Fisher-preconditioned steering under a probe softmax."""
    device = h0.device
    dtype = h0.dtype
    W = probe_W.to(device=device, dtype=dtype)
    beta = beta.to(device=device, dtype=dtype)
    eye = torch.eye(W.shape[1], device=device, dtype=dtype)
    h = h0.to(device=device, dtype=dtype).clone()
    path = [h.clone()]

    for _ in range(max(0, n_steps - 1)):
        probs = torch.softmax((W @ h).float(), dim=0).to(dtype)
        mu = probs @ W
        centered = W - mu.unsqueeze(0)
        sigma = centered.T @ (centered * probs.unsqueeze(1))
        rhs = beta.unsqueeze(1)
        try:
            v = torch.linalg.solve(sigma + float(alpha) * eye, rhs).squeeze(1)
        except RuntimeError:
            v = torch.linalg.lstsq(sigma + float(alpha) * eye, rhs).solution.squeeze(1)
        if normalize_step:
            v = v / v.norm().clamp(min=1e-12)
        h = h + float(eta) * v
        path.append(h.clone())
        if target_prob is not None:
            p_t = torch.softmax((W @ h).float(), dim=0)[int(target_class)]
            if float(p_t.item()) >= float(target_prob):
                break

    if len(path) < n_steps:
        path.extend([path[-1].clone() for _ in range(n_steps - len(path))])
    return torch.stack(path[:n_steps], dim=0)
