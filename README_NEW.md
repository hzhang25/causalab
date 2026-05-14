# Feature Geometry And Dual Steering

This branch adds a code-first implementation of feature geometry and dual steering on top of the existing manifold steering pipeline.

## What Is Implemented

### Feature Geometry Analysis

New analysis: `feature_geometry`

Files:

- `causalab/analyses/feature_geometry/main.py`
- `causalab/configs/analysis/feature_geometry.yaml`
- `causalab/methods/probes.py`
- `causalab/methods/feature_geometry.py`

The analysis consumes existing `subspace` artifacts:

- `train_dataset.json`
- `features/raw_features.safetensors`
- `features/training_features.safetensors`

It trains no-bias multiclass linear probes in both configured feature spaces:

- `activation`: raw residual activations from `raw_features.safetensors`
- `pca`: PCA coordinates from `training_features.safetensors`

For each space, it saves:

- `probe.safetensors`
- `probe.meta.json`
- `geometry.safetensors`
- `geometry.json`
- cosine Gram heatmaps
- eigenvalue spectrum plots
- probe-distance MDS plots

Implemented metrics include:

- train/test accuracy
- per-class accuracy
- probe Gram matrix `W @ W.T`
- cosine Gram matrix
- eigenvalues/eigenvectors
- DFT basis comparison for cyclic domains
- DCT basis comparison for interval domains
- graph Laplacian basis comparison for 5x5 grid
- top-2 and top-4 subspace overlap
- circulant error for cyclic domains
- intrinsic dimension at 90% variance
- probe Euclidean, cosine, and kernel-PCA distance matrices

### Probe Steering Modes

New path modes for `path_steering`:

- `additive_probe`
- `dual_probe`

Files:

- `causalab/methods/dual_steering.py`
- `causalab/analyses/path_steering/path_mode.py`
- `causalab/analyses/path_steering/main.py`
- `causalab/configs/analysis/path_steering.yaml`

`additive_probe` builds a raw-activation path along:

```python
beta = w_target - w_source
```

`dual_probe` implements probe-softmax Fisher/Newton steering:

```python
v = solve(Sigma + alpha * I, beta)
h = h + eta * normalize(v)
```

Both modes reuse the existing `path_steering` intervention and evaluation machinery, so they produce the same `pair_distributions.safetensors` format as `geometric` and `linear`.

### Runner Configs

Added opt-in Llama 3.1 8B runner configs:

- `causalab/configs/runners/weekdays/weekdays_8b_feature_geometry.yaml`
- `causalab/configs/runners/months/months_8b_feature_geometry.yaml`
- `causalab/configs/runners/alphabet/alphabet_8b_feature_geometry.yaml`
- `causalab/configs/runners/age/age_8b_feature_geometry.yaml`
- `causalab/configs/runners/graph_walk/grid_5x5_8b_feature_geometry.yaml`

Existing default path modes remain unchanged:

```yaml
path_modes:
  - geometric
  - linear
```

Probe modes are enabled only in the new feature-geometry runner configs:

```yaml
path_steering:
  path_modes: [geometric, linear, additive_probe, dual_probe]
```

`path_steering` uses the `activation` probe by default because additive and
dual probe paths operate in raw activation space.

## Sample Runs

### Validate Config Composition

These commands do not load model weights. Replace the config name with any of
the feature-geometry runners listed above.

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_feature_geometry \
  --cfg job
```

### Template Controls

The natural-domain feature-geometry runners can use either a template sweep or
the task preset's single template.

Run the configured sweep:

```bash
./scripts/run_exp.sh alphabet_8b_feature_geometry task.variant=alphabet_templates
```

Disable the sweep by clearing `task.templates` and using a fresh variant:

```bash
./scripts/run_exp.sh alphabet_8b_feature_geometry \
  task.variant=alphabet_single_template \
  task.templates=null
```

Supply a custom sweep with templates containing both `{entity}` and `{number}`:

```bash
./scripts/run_exp.sh weekdays_8b_feature_geometry \
  task.variant=weekdays_custom_templates \
  'task.templates=[
    "Q: What day is {number} days after {entity}?\nA:",
    "If today is {entity}, what day will it be in {number} days?\nA:",
    "Starting on {entity}, advance {number} days. What day do you reach?\nA:"
  ]'
```

When prompt templates are present, probe train/test splits are stratified by
both target class and template. This keeps prompt wording from becoming an
accidental train/test imbalance.

Use a fresh `task.variant` whenever changing templates, `number_range`, or
layer lists. Otherwise you may read stale artifacts from an older run.

### Weekdays Feature Geometry Example

The weekdays runner uses `number_range: 28`, five prompt templates, and a
subspace layer sweep over `[5, 10, 15, 20, 25, 28, 31]`. Each layer enumerates:

```text
7 weekdays x 28 offsets x 5 templates = 980 activation examples
```

Run inline:

```bash
./scripts/run_exp.sh weekdays_8b_feature_geometry task.variant=weekdays_templates_sweep
```

Run with Slurm:

```bash
./scripts/run_exp.sh --slurm weekdays_8b_feature_geometry task.variant=weekdays_templates_sweep
```

Run the single-template control:

```bash
./scripts/run_exp.sh weekdays_8b_feature_geometry \
  task.variant=weekdays_single_template \
  task.templates=null
```

### Natural-Domain Batch Runs

Possible sweep sizes:

```text
weekdays:  7 days x 28 offsets x 5 templates = 980 examples
months:   12 months x 28 offsets x 5 templates = 1680 examples
alphabet: 22 result letters x 4 offsets x 5 templates = 440 examples
age:      909 valid age pairs x 5 templates = 4545 possible examples
```

`weekdays`, `months`, and `alphabet` are small enough for exhaustive
enumeration with the default `n_train: 1000`. `age` is larger than `n_train`,
so the default run samples and deduplicates a subset.

Template-sweep runs:

```bash
./scripts/run_exp.sh weekdays_8b_feature_geometry task.variant=weekdays_templates_sweep
./scripts/run_exp.sh months_8b_feature_geometry task.variant=months_templates
./scripts/run_exp.sh alphabet_8b_feature_geometry task.variant=alphabet_templates
./scripts/run_exp.sh age_8b_feature_geometry task.variant=age_templates
./scripts/run_exp.sh grid_5x5_8b_feature_geometry task.variant=grid_5x5_feature_geometry
```

Single-template controls:

```bash
./scripts/run_exp.sh alphabet_8b_feature_geometry \
  task.variant=alphabet_single_template \
  task.templates=null

./scripts/run_exp.sh age_8b_feature_geometry \
  task.variant=age_single_template \
  task.templates=null

./scripts/run_exp.sh weekdays_8b_feature_geometry \
  task.variant=weekdays_single_template \
  task.templates=null
```

Add `--slurm` after `run_exp.sh` for any command above.

### Run Only Feature Geometry From Existing Subspace Artifacts

If `subspace` artifacts already exist, rerun only the probe/eigenmode analysis.
This is useful after changing probe splitting or template handling because it
refreshes `feature_geometry/probes` without recollecting activations.

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_feature_geometry \
  'defaults=[/base,/task:natural_domains_arithmetic_weekdays,/model:llama31_8b,/analysis/feature_geometry,_self_]' \
  task.target_variable=result \
  task.variant=weekdays_templates_sweep \
  feature_geometry.subspace=pca_k64
```

Change `--config-name`, `/task:...`, and `task.variant` to target another
domain or artifact variant.

### Enable Probe Modes In An Existing Path Steering Run

After running `feature_geometry`, add probe modes to any compatible path-steering run:

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_pipeline \
  task.variant=weekdays_templates_sweep \
  path_steering.path_modes='[geometric,linear,additive_probe,dual_probe]'
```

Tune dual steering:

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_feature_geometry \
  task.variant=weekdays_templates_sweep \
  path_steering.probe.alpha=0.01 \
  path_steering.probe.eta=0.02 \
  path_steering.probe.target_prob=0.99
```

To steer with a probe from another feature-geometry subdirectory or feature
space, override:

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_feature_geometry \
  task.variant=weekdays_templates_sweep \
  path_steering.probe.subdir=probes \
  path_steering.probe.feature_space=activation
```

### Layer Sweep Example

Run feature geometry at multiple layers by overriding the subspace and activation manifold layer lists:

```bash
uv run python -m causalab.runner.run_exp \
  --config-name runners/weekdays/weekdays_8b_feature_geometry \
  task.variant=weekdays_templates_sweep \
  subspace.layers='[5,10,15,20,25,28,31]' \
  activation_manifold.layers='[5,10,15,20,25,28,31]'
```

## Expected Artifact Layout

Feature geometry artifacts:

```text
artifacts/<task>/<model>/<variant>/feature_geometry/probes/<subspace>/<target_variable>/
└── L<layer>_<token_position>/
    ├── activation/
    │   ├── probe.safetensors
    │   ├── probe.meta.json
    │   ├── geometry.safetensors
    │   └── ...
    └── pca/
        ├── probe.safetensors
        ├── probe.meta.json
        ├── geometry.safetensors
        └── ...
```

The augmented-dataset PCA itself is produced by the existing `subspace` step:

```text
artifacts/<task>/<model>/<variant>/subspace/pca_k64/<target_variable>/layer_x_pos/L<layer>_<token_position>/
├── rotation.safetensors
└── features/
    ├── training_features.safetensors
    └── raw_features.safetensors
```

Path steering artifacts with probe modes:

```text
artifacts/<task>/<model>/<variant>/path_steering/<subspace>/<manifold>/<target_variable>/paths/
├── geometric/
├── linear/
├── additive_probe/
└── dual_probe/
```

Each path-mode directory follows the existing contract:

```text
pair_distributions.safetensors
pairs.json
```

## Validation

Focused validation commands:

```bash
uv run ruff check \
  causalab/analyses/subspace/grid.py \
  causalab/analyses/subspace/main.py \
  causalab/methods/probes.py \
  causalab/methods/feature_geometry.py \
  causalab/methods/dual_steering.py \
  causalab/analyses/feature_geometry/main.py \
  causalab/analyses/path_steering/path_mode.py \
  causalab/analyses/path_steering/main.py \
  tests/test_feature_geometry_probes.py \
  tests/test_natural_domains_arithmetic/test_causal_models.py
```

```bash
uv run pytest tests/test_feature_geometry_probes.py
uv run pytest \
  tests/test_natural_domains_arithmetic/test_causal_models.py::test_weekdays_template_variations_expand_inputs \
  tests/test_natural_domains_arithmetic/test_causal_models.py::test_weekdays_templates_sweep_enumerated_baseline_is_balanced \
  tests/test_natural_domains_arithmetic/test_causal_models.py::test_resolve_task_accepts_template_overrides
```

Current focused tests cover:

- probe artifact save/load
- topology basis orthonormality
- subspace overlap
- circulant approximation
- dual steering target-probability improvement
- path-mode resolution for probe modes
- raw and PCA per-cell feature artifacts
- natural-domain template expansion through `resolve_task`
- template-aware probe split stratification
- exhaustive weekdays template-sweep balance across `(result, template)` cells

