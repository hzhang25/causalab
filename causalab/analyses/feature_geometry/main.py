"""Feature geometry: train linear probes and analyze their domain embedding."""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.io.pipelines import find_subspace_dirs, load_subspace_metadata
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.methods.feature_geometry import save_geometry_artifacts, summarize_geometry
from causalab.methods.probes import labels_from_examples, save_probe, train_multiclass_probe
from causalab.runner.helpers import resolve_task

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "feature_geometry"


def _parse_cell_from_path(feature_dir: str) -> tuple[int | None, str | None]:
    """Parse `layer_x_pos/L<layer>_<pos>/features` style paths."""
    cell = os.path.basename(os.path.dirname(feature_dir))
    if not cell.startswith("L") or "_" not in cell:
        return None, None
    layer_s, pos = cell[1:].split("_", 1)
    try:
        return int(layer_s), pos
    except ValueError:
        return None, pos


def _feature_dirs(subspace_dir: str) -> list[tuple[str, int | None, str | None]]:
    """Return feature directories that contain activation/PCA tensors."""
    direct = os.path.join(subspace_dir, "features")
    if os.path.exists(os.path.join(direct, "raw_features.safetensors")) or os.path.exists(
        os.path.join(direct, "training_features.safetensors")
    ):
        return [(direct, None, None)]

    found: list[tuple[str, int | None, str | None]] = []
    grid_root = os.path.join(subspace_dir, "layer_x_pos")
    if not os.path.isdir(grid_root):
        return found
    for cell in sorted(os.listdir(grid_root)):
        feat_dir = os.path.join(grid_root, cell, "features")
        if os.path.exists(os.path.join(feat_dir, "raw_features.safetensors")) or os.path.exists(
            os.path.join(feat_dir, "training_features.safetensors")
        ):
            layer, pos = _parse_cell_from_path(feat_dir)
            found.append((feat_dir, layer, pos))
    return found


def _subspace_dir(root: str, subspace_sub: str, target_variable: str | None) -> str:
    path = os.path.join(root, "subspace", subspace_sub)
    if target_variable:
        tv_path = os.path.join(path, target_variable)
        if os.path.isdir(tv_path):
            path = tv_path
    return path


def _output_dir(
    analysis: DictConfig,
    subspace_sub: str,
    target_variable: str | None,
    layer: int | None,
    token_position: str | None,
    feature_space: str,
    include_space: bool,
) -> str:
    out = os.path.join(str(analysis._output_dir), subspace_sub)
    if target_variable:
        out = os.path.join(out, target_variable)
    if layer is not None or token_position is not None:
        out = os.path.join(out, f"L{layer}_{token_position}")
    if include_space:
        out = os.path.join(out, feature_space)
    return out


def _configured_feature_spaces(analysis: DictConfig) -> list[str]:
    spaces = list(OmegaConf.to_container(analysis.get("feature_spaces"), resolve=True))
    valid = {"activation", "raw", "pca"}
    unknown = [space for space in spaces if space not in valid]
    if unknown:
        raise ValueError(f"Unknown feature_geometry.feature_spaces: {unknown}")
    return ["activation" if space == "raw" else str(space) for space in spaces]


def _load_feature_space(feat_dir: str, feature_space: str):
    """Load one feature-space tensor from a subspace feature directory."""
    filename = (
        "raw_features.safetensors"
        if feature_space == "activation"
        else "training_features.safetensors"
    )
    path = os.path.join(feat_dir, filename)
    if not os.path.exists(path):
        return None, path
    return load_file(path)["features"].float(), path


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run feature geometry from cached subspace activations/PCA coordinates."""
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)
    root = cfg.experiment_root
    target_variable = cfg.task.get("target_variable")

    task, _task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=target_variable,
        seed=cfg.seed,
    )
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)

    subspaces = [str(analysis.subspace)] if analysis.subspace is not None else find_subspace_dirs(root)
    if not subspaces:
        raise ValueError(f"No subspace artifacts found under {root}/subspace")

    feature_spaces = _configured_feature_spaces(analysis)
    include_space_dir = len(feature_spaces) > 1
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for subspace_sub in subspaces:
        ss_dir = _subspace_dir(root, subspace_sub, target_variable)
        ds_path = os.path.join(ss_dir, "train_dataset.json")
        if not os.path.exists(ds_path):
            logger.warning("Skipping %s: missing train_dataset.json", subspace_sub)
            continue
        examples = load_counterfactual_examples(ds_path, task.causal_model)
        ss_meta = load_subspace_metadata(root, subspace_sub, target_variable)

        for feat_dir, layer_from_path, pos_from_path in _feature_dirs(ss_dir):
            layer = layer_from_path if layer_from_path is not None else ss_meta.get("layer")
            token_position = (
                pos_from_path if pos_from_path is not None else ss_meta.get("token_position")
            )
            for feature_space in feature_spaces:
                features, features_path = _load_feature_space(feat_dir, feature_space)
                if features is None:
                    logger.warning(
                        "Skipping %s/%s/%s %s: missing features at %s",
                        subspace_sub,
                        layer,
                        token_position,
                        feature_space,
                        features_path,
                    )
                    continue
                labels = labels_from_examples(examples, task, n=features.shape[0])
                out_dir = _output_dir(
                    analysis,
                    subspace_sub,
                    target_variable,
                    layer,
                    token_position,
                    feature_space,
                    include_space_dir,
                )
                logger.info(
                    "Training %s-space probe for %s (%s, %s): X=%s",
                    feature_space,
                    subspace_sub,
                    layer,
                    token_position,
                    tuple(features.shape),
                )
                result = train_multiclass_probe(
                    features,
                    labels,
                    n_classes=len(task.intervention_values),
                    train_frac=float(analysis.train_frac),
                    seed=int(cfg.seed),
                    lr=float(analysis.lr),
                    weight_decay=float(analysis.weight_decay),
                    epochs=int(analysis.epochs),
                    batch_size=int(analysis.batch_size),
                )
                meta = {
                    "analysis": ANALYSIS_NAME,
                    "subspace": subspace_sub,
                    "target_variable": target_variable,
                    "layer": layer,
                    "token_position": token_position,
                    "feature_space": feature_space,
                    "task": cfg.task.name,
                    "task_config": task_cfg,
                    "values": [str(v) for v in task.intervention_values],
                    "features_path": features_path,
                    "no_bias": True,
                }
                save_probe(out_dir, result, metadata=meta)
                summary, tensors = summarize_geometry(
                    result.weight,
                    task_name=cfg.task.name,
                    values=list(task.intervention_values),
                    task_config=task_cfg,
                )
                save_geometry_artifacts(
                    out_dir,
                    tensors,
                    summary,
                    values=list(task.intervention_values),
                    figure_format=figure_fmt,
                )

                row = {
                    "subspace": subspace_sub,
                    "layer": layer,
                    "token_position": token_position,
                    "feature_space": feature_space,
                    **result.metrics,
                    **summary,
                }
                rows.append(row)
                results[f"{subspace_sub}/L{layer}_{token_position}/{feature_space}"] = row

                if bool(analysis.get("random_control", False)):
                    shuffled = train_multiclass_probe(
                        features,
                        labels,
                        n_classes=len(task.intervention_values),
                        train_frac=float(analysis.train_frac),
                        seed=int(cfg.seed),
                        lr=float(analysis.lr),
                        weight_decay=float(analysis.weight_decay),
                        epochs=int(analysis.epochs),
                        batch_size=int(analysis.batch_size),
                        shuffle_labels=True,
                    )
                    ctrl_dir = os.path.join(out_dir, "random_control")
                    save_probe(ctrl_dir, shuffled, metadata={**meta, "random_control": True})
                    ctrl_summary, ctrl_tensors = summarize_geometry(
                        shuffled.weight,
                        task_name=cfg.task.name,
                        values=list(task.intervention_values),
                        task_config=task_cfg,
                    )
                    save_geometry_artifacts(
                        ctrl_dir,
                        ctrl_tensors,
                        ctrl_summary,
                        values=list(task.intervention_values),
                        figure_format=figure_fmt,
                    )

    summary_path = os.path.join(str(analysis._output_dir), "summary.csv")
    _write_summary_csv(summary_path, rows)
    os.makedirs(str(analysis._output_dir), exist_ok=True)
    with open(os.path.join(str(analysis._output_dir), "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
