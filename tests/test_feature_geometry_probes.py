from __future__ import annotations

import torch

from causalab.analyses.feature_geometry.main import _feature_dirs
from causalab.analyses.path_steering.path_mode import resolve_path_modes
from causalab.analyses.subspace.main import _save_pca_per_cell_artifacts
from causalab.methods.dual_steering import dual_steer_path
from causalab.methods.feature_geometry import (
    circulant_approximation,
    dct_basis,
    dft_real_basis,
    grid_laplacian_basis,
    subspace_overlap,
)
from causalab.methods.probes import ProbeResult, find_probe_dir, load_probe, save_probe


def test_probe_artifact_roundtrip(tmp_path):
    result = ProbeResult(
        weight=torch.eye(3),
        train_indices=torch.tensor([0, 1, 2]),
        test_indices=torch.tensor([3, 4, 5]),
        metrics={"test_accuracy": 1.0},
    )

    assert result.weight.shape == (3, 3)
    assert result.metrics["test_accuracy"] > 0.9
    save_probe(str(tmp_path), result, metadata={"layer": 28})
    weight, meta = load_probe(str(tmp_path))
    assert torch.allclose(weight, result.weight)
    assert meta["layer"] == 28


def test_find_probe_dir_prefers_requested_feature_space(tmp_path):
    result = ProbeResult(
        weight=torch.eye(3),
        train_indices=torch.tensor([0]),
        test_indices=torch.tensor([1]),
        metrics={},
    )
    base = tmp_path / "feature_geometry" / "probes" / "pca_k64" / "result"
    save_probe(
        str(base / "L28_last_token" / "activation"),
        result,
        metadata={
            "layer": 28,
            "token_position": "last_token",
            "feature_space": "activation",
        },
    )
    save_probe(
        str(base / "L28_last_token" / "pca"),
        result,
        metadata={"layer": 28, "token_position": "last_token", "feature_space": "pca"},
    )
    found = find_probe_dir(
        str(tmp_path),
        "pca_k64",
        target_variable="result",
        layer=28,
        token_position="last_token",
        feature_space="activation",
        feature_geometry_subdir="probes",
    )
    assert found is not None
    assert found.endswith("activation")


def test_pca_grid_artifacts_save_raw_and_projected_features(tmp_path):
    out_dir = tmp_path / "subspace"
    _save_pca_per_cell_artifacts(
        {
            (28, "last_token"): {
                "rotation": torch.eye(4, 2),
                "explained_variance_ratio": [0.6, 0.2],
            }
        },
        {(28, "last_token"): torch.ones(3, 2)},
        {(28, "last_token"): torch.ones(3, 4)},
        str(out_dir),
    )
    feat_dir = out_dir / "layer_x_pos" / "L28_last_token" / "features"
    assert (feat_dir / "training_features.safetensors").exists()
    assert (feat_dir / "raw_features.safetensors").exists()
    assert _feature_dirs(str(out_dir)) == [(str(feat_dir), 28, "last_token")]


def test_topology_bases_are_orthonormal():
    for basis in (dft_real_basis(7), dct_basis(9), grid_laplacian_basis(25)):
        ident = basis.T @ basis
        assert torch.allclose(ident, torch.eye(basis.shape[1]), atol=1e-5)


def test_subspace_overlap_and_circulant_error():
    basis = dft_real_basis(7)
    K = basis[:, :3] @ basis[:, :3].T
    assert subspace_overlap(basis, basis, 4) > 1.0 - 1e-5
    _circ, err = circulant_approximation(K)
    assert err < 1e-5


def test_dual_steering_increases_target_logit():
    W = torch.eye(3)
    h0 = torch.tensor([2.0, 0.0, 0.0])
    beta = W[1] - W[0]
    path = dual_steer_path(
        h0,
        target_class=1,
        beta=beta,
        probe_W=W,
        n_steps=8,
        eta=0.2,
        alpha=1e-2,
    )
    probs = torch.softmax(path @ W.T, dim=-1)
    assert probs[-1, 1] > probs[0, 1]


def test_resolve_probe_path_modes():
    W = torch.eye(3)
    modes = resolve_path_modes(["additive_probe", "dual_probe"], probe_weight=W)
    assert [m.label for m in modes] == ["additive_probe", "dual_probe"]
    assert all(m.centroid_space == "raw" for m in modes)
    path = modes[0].build_path(
        torch.zeros(3),
        torch.ones(3),
        5,
        start_index=0,
        end_index=1,
    )
    assert path.shape == (5, 3)
