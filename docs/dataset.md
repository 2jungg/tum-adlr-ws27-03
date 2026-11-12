# Co-SDF 2-DoF Dataset Specification

Each saved `.pt` file (`train.pt`, `val.pt`, `eval.pt`) is structured as a dictionary containing dataset-level configuration and a list of scenes:

```python
dataset = {
    # -------------------------------------------------------------------------
    # 1. Dataset Generation Config (Global settings & robot environment)
    # -------------------------------------------------------------------------
    "config": {
        "split": "train",              # 'train', 'val', or 'eval'
        "n_scenes": 100,               # Number of scenes in this file
        "n_samples": 5000,             # Number of samples per scene (N)
        "boundary_sampling": True,     # Whether boundary-densified sampling was used
        "boundary_ratio": 0.5,         # Ratio of boundary samples (if enabled)
        "sigma": 0.15,                 # Gaussian perturbation std (radians)
        "bisection_steps": 3,          # Number of midpoint bisection steps for boundary samples
        "max_obs": 4,                  # Maximum number of obstacles per scene
        "seed": 42,                    # Base RNG seed for this split
        
        # Robot kinematic specification (Common to all scenes in this dataset)
        "arm": {
            "link_lengths": [1.0, 1.0],
            "base": [0.0, 0.0],
            "link_radius": 0.1,
        },
        # Workspace bounds [xmin, xmax, ymin, ymax]
        "bounds": [-3.0, 3.0, -3.0, 3.0],
    },

    # -------------------------------------------------------------------------
    # 2. List of Scenes
    # -------------------------------------------------------------------------
    "scenes": [
        {
            # 1. Training samples in C-space (Joint angles in radians within [0, 2*pi))
            "train_q": torch.Tensor,   # shape: (N, 2), dtype: float32
            
            # 2. Binary collision labels: +1 = collision-free (C_free), -1 = in collision (C_obs)
            "train_y": torch.Tensor,   # shape: (N,), dtype: float32, values in {-1.0, +1.0}

            # 3. Scene-specific Geometry Metadata
            # [NOTE] This field is strictly required and included ONLY in the evaluation dataset (eval.pt).
            # It stores the obstacle list needed to reconstruct the Scene object, compute GT C-space SDFs on-the-fly,
            # plot workspace layouts, and run local escape/trajectory planning benchmarks.
            # (Omitted in train.pt and val.pt to keep them lightweight)
            "metadata": {
                "obstacles": [
                    {
                        "type": "circle",
                        "center": [1.0, 1.0],
                        "radius": 0.5,
                    },
                    {
                        "type": "box",
                        "center": [-1.5, 0.8],
                        "half_size": [0.3, 0.5],
                        "angle": 0.785,  # counter-clockwise rotation in radians
                    },
                ],
            },  # Included only in eval.pt; omitted in train.pt and val.pt
        },
        # ... remaining scenes
    ],
}
```
