"""Path modes: how a steered path is built between two centroid endpoints.

A :class:`PathMode` carries the centroid space (``intrinsic`` / ``raw`` /
``pca``), the path kind (``geodesic`` / ``linear``), and any featurizer
override applied during steering. Callers dispatch through ``pm.build_path``
and ``pm.select_centroids`` rather than branching on ``pm.label``.

The low-level path primitives live here too so each mode owns the geometry
it prescribes. Import them directly only when you genuinely need both kinds
independent of any mode (e.g. visualization overlays, pullback comparison
passes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from causalab.neural.featurizer import ComposedFeaturizer, Featurizer

logger = logging.getLogger(__name__)


CentroidSpace = Literal["intrinsic", "raw", "pca"]
PathKind = Literal["geodesic", "linear", "additive_probe", "dual_probe"]


# ---------------------------------------------------------------------------
# Path-construction primitives
# ---------------------------------------------------------------------------


def _get_periodic_periods(manifold_obj: Any) -> dict[int, float]:
    """Per-dim periods from a manifold object (empty dict if none)."""
    periodic_dims = getattr(manifold_obj, "periodic_dims", None)
    periods = manifold_obj.periods if hasattr(manifold_obj, "periods") else None
    if not periodic_dims or not periods:
        return {}
    return dict(zip(periodic_dims, periods))


def _build_path_alphas(
    n_steps: int,
    oversteer_frac: float,
    oversteer_steps: int,
    device: torch.device,
) -> Tensor:
    """Alpha values: n_steps from [0,1], then oversteer_steps from (1, 1+frac]."""
    normal = torch.linspace(0, 1, n_steps, device=device)
    if oversteer_frac > 0 and oversteer_steps > 0:
        step_size = oversteer_frac / oversteer_steps
        overshoot = torch.linspace(
            1.0 + step_size,
            1.0 + oversteer_frac,
            oversteer_steps,
            device=device,
        )
        return torch.cat([normal, overshoot])
    return normal


def _build_geodesic_path(
    start_intrinsic: Tensor,
    end_intrinsic: Tensor,
    n_steps: int,
    manifold_obj: Any,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> Tensor:
    """Geodesic path in intrinsic space (linear interp, periodic shortest-arc per dim).

    When oversteer_frac > 0, oversteer_steps coarse points continue past the target,
    wrapping around periodic dimensions naturally.
    """
    alphas = _build_path_alphas(
        n_steps,
        oversteer_frac,
        oversteer_steps,
        start_intrinsic.device,
    )
    d = start_intrinsic.shape[0]
    periodic_periods = _get_periodic_periods(manifold_obj)

    # Clone before in-place wrap so autograd stays safe if inputs ever require grad.
    delta = (end_intrinsic - start_intrinsic).clone()
    for dim, period in periodic_periods.items():
        if dim < d:
            delta[dim] = ((delta[dim] + period / 2.0) % period) - period / 2.0

    return start_intrinsic.unsqueeze(0) + alphas.unsqueeze(1) * delta.unsqueeze(0)


def _build_linear_path_kd(
    start_features: Tensor,
    end_features: Tensor,
    n_steps: int,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> Tensor:
    """Straight-line path in feature space (raw activations or PCA k-dim).

    When oversteer_frac > 0, oversteer_steps coarse points extrapolate beyond
    the target, moving off-manifold and surfacing out-of-distribution mass.
    """
    alphas = _build_path_alphas(
        n_steps,
        oversteer_frac,
        oversteer_steps,
        start_features.device,
    )
    return start_features.unsqueeze(0) + alphas.unsqueeze(1) * (
        end_features - start_features
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# PathMode
# ---------------------------------------------------------------------------


@dataclass
class PathMode:
    """Resolved path mode: centroid space, path kind, and optional featurizer override."""

    label: str
    centroid_space: CentroidSpace
    path_kind: PathKind
    featurizer_override: Featurizer | None = None
    probe_weight: Tensor | None = None
    dual_eta: float = 0.01
    dual_alpha: float = 1e-3
    dual_target_prob: float | None = None
    additive_scale_to_endpoint: bool = True

    def select_centroids(
        self,
        *,
        intrinsic: Tensor,
        raw: Tensor,
        pca: Tensor,
    ) -> Tensor:
        """Return the centroid set this mode operates on."""
        return {"intrinsic": intrinsic, "raw": raw, "pca": pca}[self.centroid_space]

    def build_path(
        self,
        start: Tensor,
        end: Tensor,
        n_steps: int,
        *,
        manifold_obj: Any | None = None,
        oversteer_frac: float = 0.0,
        oversteer_steps: int = 0,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> Tensor:
        """Build the (n_steps [+ oversteer_steps], d) path between two endpoints.

        When ``path_kind="geodesic"`` but no manifold is available, falls back
        to a straight line — preserves the historical fallback behaviour.
        """
        if self.path_kind == "additive_probe":
            if self.probe_weight is None or start_index is None or end_index is None:
                logger.warning("%s missing probe/class indices; falling back to linear path", self.label)
            else:
                from causalab.methods.dual_steering import additive_probe_path

                return additive_probe_path(
                    start,
                    end,
                    self.probe_weight[start_index],
                    self.probe_weight[end_index],
                    n_steps + oversteer_steps,
                    scale_to_endpoint=self.additive_scale_to_endpoint,
                )
        if self.path_kind == "dual_probe":
            if self.probe_weight is None or start_index is None or end_index is None:
                logger.warning("%s missing probe/class indices; falling back to linear path", self.label)
            else:
                from causalab.methods.dual_steering import dual_steer_path

                beta = self.probe_weight[end_index] - self.probe_weight[start_index]
                return dual_steer_path(
                    start,
                    target_class=end_index,
                    beta=beta,
                    probe_W=self.probe_weight,
                    n_steps=n_steps + oversteer_steps,
                    eta=self.dual_eta,
                    alpha=self.dual_alpha,
                    target_prob=self.dual_target_prob,
                )
        if self.path_kind == "geodesic" and manifold_obj is not None:
            return _build_geodesic_path(
                start,
                end,
                n_steps,
                manifold_obj,
                oversteer_frac=oversteer_frac,
                oversteer_steps=oversteer_steps,
            )
        return _build_linear_path_kd(
            start,
            end,
            n_steps,
            oversteer_frac=oversteer_frac,
            oversteer_steps=oversteer_steps,
        )


def resolve_path_modes(
    path_modes_cfg: list,
    composed_featurizer: ComposedFeaturizer | None = None,
    probe_weight: Tensor | None = None,
    probe_cfg: Any | None = None,
) -> list[PathMode]:
    """Resolve each path_mode config entry into a :class:`PathMode`.

    Supported modes:
    - ``"geometric"`` — geodesic path in intrinsic (u-space) coordinates.
    - ``"linear"`` — straight-line path in raw activation space (identity featurizer).
    - ``"linear_subspace"`` — straight-line path in PCA-reduced subspace.
    - ``"additive_probe"`` — raw-space path along probe-vector difference.
    - ``"dual_probe"`` — raw-space Fisher-preconditioned probe-softmax path.
    """
    results: list[PathMode] = []

    for mode in path_modes_cfg:
        if not isinstance(mode, str):
            raise ValueError(f"Invalid path_mode entry: {mode!r}")

        if mode == "geometric":
            results.append(
                PathMode(
                    label="geometric",
                    centroid_space="intrinsic",
                    path_kind="geodesic",
                )
            )
        elif mode == "linear":
            results.append(
                PathMode(
                    label="linear",
                    centroid_space="raw",
                    path_kind="linear",
                    featurizer_override=Featurizer(id="identity"),
                )
            )
        elif mode == "linear_subspace":
            pca_override = None
            if isinstance(composed_featurizer, ComposedFeaturizer):
                pca_override = composed_featurizer.stages[0]
            results.append(
                PathMode(
                    label="linear_subspace",
                    centroid_space="pca",
                    path_kind="linear",
                    featurizer_override=pca_override,
                )
            )
        elif mode == "additive_probe":
            results.append(
                PathMode(
                    label="additive_probe",
                    centroid_space="raw",
                    path_kind="additive_probe",
                    featurizer_override=Featurizer(id="identity"),
                    probe_weight=probe_weight,
                    additive_scale_to_endpoint=bool(getattr(probe_cfg, "scale_to_endpoint", True)),
                )
            )
        elif mode == "dual_probe":
            results.append(
                PathMode(
                    label="dual_probe",
                    centroid_space="raw",
                    path_kind="dual_probe",
                    featurizer_override=Featurizer(id="identity"),
                    probe_weight=probe_weight,
                    dual_eta=float(getattr(probe_cfg, "eta", 0.01)),
                    dual_alpha=float(getattr(probe_cfg, "alpha", 1e-3)),
                    dual_target_prob=getattr(probe_cfg, "target_prob", None),
                )
            )
        else:
            raise ValueError(f"Unknown path_mode: {mode!r}")

    return results
