# Myo Exoskeleton CleanRL Experiments

This repository is the compact code snapshot for the 2D MyoAssist/MJWarp muscle-control experiments. It keeps the runnable training code, staged configs, reference builders, small reference datasets, and analysis scripts. Large experiment outputs, checkpoints, TensorBoard logs, and rendered rollout videos are intentionally excluded.

## Repository Layout

- `cleanrl/`: self-contained PPO/SAC-style MJWarp training code for 22-muscle 2D MyoAssist control.
- `configs/`: current flat, uphill, and flat-up-flat staged curricula.
- `scripts/`: reference construction, staged-config generation, video rendering, and synergy dataset/analysis helpers.
- `data/camargo_transition_references/`: compact retargeted Camargo transition references used by the staged configs.
- `reference_exports/`: small selected reference `.npz` files, review metadata, and synergy datasets.
- `docs/`: local data-source notes.

## External Dependencies

The MuJoCo XML model is not vendored in this repository. Existing configs point to:

```text
/home/lzn/myoassist/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

On a different machine, either put the model at the same path or update `model.source_xml` in the config JSON files.

The working local environment has been `myoassist-mjwarp`. Avoid upgrading core packages in-place. If dependency changes are needed, clone the environment first:

```bash
conda create --name myoassist-mjwarp-next --clone myoassist-mjwarp
```

## Main Entrypoints

Train from a config:

```bash
python cleanrl/sac_muscle_mjwarp.py --config configs/stageG_flat_up_flat9_staged/muscle_2d_mjwarp_stageG_fuf9_A_h48_islands_imit_sac.json
```

Run a staged curriculum manifest:

```bash
python scripts/run_stageG_staged_pipeline.py --manifest configs/stageG_flat_up_flat9_staged/manifest.json
```

Render a checkpoint:

```bash
python scripts/render_sac_checkpoint_videos.py \
  --config configs/stageG_flat_up_flat9_staged/muscle_2d_mjwarp_stageG_fuf9_F_h192_full_imit_sac.json \
  --checkpoint /path/to/agent_step.pt
```

## Current Useful Config Families

- `configs/stageG_uphill9_footlocked035_clip_staged/`: 9-degree uphill-only specialist curriculum.
- `configs/stageG_flat_up_flat9_staged/`: flat -> 9-degree uphill -> high-flat curriculum.
- `configs/muscle_2d_mjwarp_stageA...stageF...json`: older flat and Camargo transition baselines that still use included data or external MyoAssist paths.

Older configs that referenced excluded `results/` or `results_old/` generated references were left out of this compact upload.

## What Is Excluded

- `results/`
- `results_old/`
- `video_exports/`
- `reference_videos/`
- MuJoCo logs
- model checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- TensorBoard/event logs
- local secrets and machine-specific credentials

