# Co-SDF 2-DoF Dataset Specification

Build the datasets with `python scripts/02_build_dataset.py` (all splits) or with `--eval` (only `eval.pt`).
The settings come from `configs/dataset.yaml`, a `DatasetConfig` (`src/metacosdf/dataset.py`).
`python scripts/write_default_configs.py` rewrites that file with the default values.

Each saved `.pt` file (`train.pt`, `val.pt`, `eval.pt`) is a dictionary with the dataset configuration and a list of scenes:

```python
dataset = {
    # -------------------------------------------------------------------------
    # 1. Dataset configuration: DatasetConfig.to_dict(), plus the split
    # -------------------------------------------------------------------------
    "config": {
        "split": "train",  # 'train', 'val', or 'eval'
        "n_scenes": 100,  # Number of scenes in this file
        "generator": {  # SceneGeneratorConfig: random scenes
            "robot": {  # PlanarArm, common to all scenes
                "link_lengths": [1.0, 1.0],
                "base_position_x": 0.0,
                "base_position_y": 0.0,
                "link_radius": 0.1,
            },
            "bounds": {"xmin": -3.0, "xmax": 3.0, "ymin": -3.0, "ymax": 3.0},
            "max_obstacles": 4,  # 1 to max_obstacles obstacles per scene
            "min_base_clearance": 0.2,
        },
        "sampling": {  # SamplingConfig: labeled configurations
            "n_samples": 5000,  # Number of samples per scene (N)
            "boundary_sampling": True,  # Whether boundary-densified sampling was used
            "boundary_ratio": 0.5,  # Ratio of boundary samples (if enabled)
            "sigma": 0.15,  # Gaussian perturbation std (radians)
            "bisection_steps": 3,  # Midpoint bisection steps for boundary samples
        },
        "n_train": 100,
        "n_val": 20,
        "n_eval": 20,
        "seed": 42,  # Base seed; each split adds its own offset
        "out_dir": "data/scenes",
    },
    # -------------------------------------------------------------------------
    # 2. List of Scenes
    # -------------------------------------------------------------------------
    "scenes": [
        {
            # 1. Training samples in C-space (Joint angles in radians within [0, 2*pi))
            "train_q": torch.Tensor,  # shape: (N, 2), dtype: float32
            # 2. Binary collision labels: +1 = collision-free (C_free), -1 = in collision (C_obs)
            "train_y": torch.Tensor,  # shape: (N,), dtype: float32, values in {-1.0, +1.0}
            # 3. The scene itself, as Scene.to_dict(): rebuild it with Scene.from_dict(metadata).
            # [NOTE] Included ONLY in the evaluation dataset (eval.pt), where it is needed to compute
            # the ground-truth C-space SDF, plot the workspace and run the escape / planning benchmarks.
            # (Omitted in train.pt and val.pt to keep them lightweight)
            "metadata": {
                "obstacle": {
                    "elements": [
                        {
                            "center_x": 1.0,
                            "center_y": 1.0,
                            "radius": 0.5,
                            "obstacle_type": "circle",
                        },
                        {
                            "center_x": -1.5,
                            "center_y": 0.8,
                            "half_size_x": 0.3,
                            "half_size_y": 0.5,
                            "angle": 0.785,  # counter-clockwise rotation in radians
                            "obstacle_type": "box",
                        },
                    ],
                    "obstacle_type": "composition",
                },
                "robot": {...},  # as in config["generator"]["robot"]
                "bounds": {...},  # as in config["generator"]["bounds"]
            },
        },
        # ... remaining scenes
    ],
}
```

**Seeds.** A split uses the seed `seed + offset`, with offsets train 0, val 10000, eval 20000.
That seed drives a numpy generator for the obstacles and, through `torch.manual_seed`, the sampling of configurations, so rebuilding a split with the same config gives the same file.
