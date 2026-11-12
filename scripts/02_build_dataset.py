"""Build 2-DoF Co-SDF meta-learning datasets (train.pt, val.pt, eval.pt)."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from metacosdf.scene_generator.arm import PlanarArm
from metacosdf.scene_generator.generator import create_random_scene


def load_yaml_config(path: Path) -> dict:
    """Load YAML config if PyYAML is installed and file exists."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def parse_args():
    default_config_path = Path("src/metacosdf/scene_generator/config.yaml")

    # Pre-parse --config so CLI defaults can be populated from YAML
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=default_config_path)
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_yaml_config(pre_args.config)
    arm_cfg = cfg.get("arm", {})
    scene_cfg = cfg.get("scene", {})
    ds_cfg = cfg.get("dataset", {})

    parser = argparse.ArgumentParser(description="Build 2-DoF Co-SDF meta-learning datasets.")
    parser.add_argument("--config", type=Path, default=default_config_path, help="Path to config.yaml.")
    parser.add_argument("--eval", action="store_true", help="Generate only the evaluation dataset (eval.pt).")

    # Split scene counts
    parser.add_argument("--n-train", type=int, default=ds_cfg.get("n_train", 100), help="Number of train scenes.")
    parser.add_argument("--n-val", type=int, default=ds_cfg.get("n_val", 20), help="Number of val scenes.")
    parser.add_argument("--n-eval", type=int, default=ds_cfg.get("n_eval", 20), help="Number of eval scenes.")

    # Sampling configuration
    parser.add_argument(
        "--n-samples",
        type=int,
        default=ds_cfg.get("n_samples", 5000),
        help="Configurations per scene (default: 5000).",
    )
    parser.add_argument(
        "--boundary-sampling",
        action=argparse.BooleanOptionalAction,
        default=ds_cfg.get("boundary_sampling", True),
        help="Use boundary-densified sampling (default: True).",
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=ds_cfg.get("boundary_ratio", 0.5),
        help="Boundary sample ratio (default: 0.5).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=ds_cfg.get("sigma", 0.15),
        help="Gaussian perturbation std in radians (default: 0.15).",
    )
    parser.add_argument(
        "--bisection-steps",
        type=int,
        default=ds_cfg.get("bisection_steps", 3),
        help="Midpoint bisection iterations (default: 3).",
    )

    # Environment and reproducibility
    parser.add_argument(
        "--max-obs",
        type=int,
        default=scene_cfg.get("max_obs", 4),
        help="Max obstacles per scene (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ds_cfg.get("seed", 42),
        help="Base random seed.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(ds_cfg.get("out_dir", "data/scenes")),
        help="Output directory.",
    )

    args = parser.parse_args()
    return args, arm_cfg, scene_cfg


def build_split(
    split: str,
    n_scenes: int,
    seed: int,
    arm: PlanarArm,
    bounds: tuple[float, float, float, float],
    max_obs: int,
    n_samples: int,
    boundary_sampling: bool,
    boundary_ratio: float,
    sigma: float,
    bisection_steps: int,
    out_dir: Path,
) -> None:
    rng = np.random.default_rng(seed)
    scenes_data = []

    print(f"\n[{split.upper()}] Generating {n_scenes} scenes (seed={seed})...")
    t0 = time.time()
    for idx in range(n_scenes):
        scene = create_random_scene(arm, bounds=bounds, max_obs=max_obs, rng=rng)
        q, y = scene.sample_configurations(
            arm=arm,
            n_samples=n_samples,
            use_boundary_sampling=boundary_sampling,
            boundary_ratio=boundary_ratio,
            sigma=sigma,
            bisection_steps=bisection_steps,
        )

        entry = {
            "train_q": q,
            "train_y": y,
        }
        if split == "eval":
            entry["metadata"] = {
                "obstacles": [obs.to_dict() for obs in scene.obstacles],
            }

        scenes_data.append(entry)
        if (idx + 1) % max(1, n_scenes // 5) == 0 or (idx + 1) == n_scenes:
            print(f"  [{split}] {idx + 1}/{n_scenes} scenes completed...")

    dataset = {
        "config": {
            "split": split,
            "n_scenes": n_scenes,
            "n_samples": n_samples,
            "boundary_sampling": boundary_sampling,
            "boundary_ratio": boundary_ratio if boundary_sampling else 0.0,
            "sigma": sigma,
            "bisection_steps": bisection_steps,
            "max_obs": max_obs,
            "seed": seed,
            "arm": {
                "link_lengths": list(arm.link_lengths),
                "base": list(arm.base),
                "link_radius": arm.link_radius,
            },
            "bounds": list(bounds),
        },
        "scenes": scenes_data,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.pt"
    torch.save(dataset, out_path)
    elapsed = time.time() - t0
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[{split.upper()}] Saved to {out_path} ({file_size_mb:.2f} MB, {elapsed:.1f}s)")


def main() -> None:
    args, arm_cfg, scene_cfg = parse_args()

    link_lengths = tuple(arm_cfg.get("link_lengths", [1.0, 1.0]))
    base = tuple(arm_cfg.get("base", [0.0, 0.0]))
    link_radius = float(arm_cfg.get("link_radius", 0.1))
    arm = PlanarArm(link_lengths=link_lengths, base=base, link_radius=link_radius)

    bounds = tuple(scene_cfg.get("bounds", [-3.0, 3.0, -3.0, 3.0]))

    split_offsets = {
        "train": 0,
        "val": 10000,
        "eval": 20000,
    }

    if args.eval:
        targets = [("eval", args.n_eval)]
    else:
        targets = [
            ("train", args.n_train),
            ("val", args.n_val),
            ("eval", args.n_eval),
        ]

    for split, n_scenes in targets:
        if n_scenes <= 0:
            continue
        split_seed = args.seed + split_offsets[split]
        build_split(
            split=split,
            n_scenes=n_scenes,
            seed=split_seed,
            arm=arm,
            bounds=bounds,
            max_obs=args.max_obs,
            n_samples=args.n_samples,
            boundary_sampling=args.boundary_sampling,
            boundary_ratio=args.boundary_ratio,
            sigma=args.sigma,
            bisection_steps=args.bisection_steps,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
