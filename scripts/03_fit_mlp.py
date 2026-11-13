"""Fit a plain MLP field to the collision labels of one scene, compare it with the ground truth.

Without --dataset, the scene comes from configs/default_scene.yaml and its labeled
configurations are sampled here. With --dataset, scene and labels come from an eval split:
only eval.pt stores the scenes, which the ground truth needs.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from metacosdf.config_utils import load_config
from metacosdf.fields import NeuralField, embedding_size, evaluate_on_grid
from metacosdf.metrics import field_metrics
from metacosdf.models import MLP, MLPConfig
from metacosdf.scene_generator.scene import SamplingConfig, Scene
from metacosdf.scene_generator.visualizer import SceneVisualizer
from metacosdf.training import TrainingConfig, fit_field

ROOT = Path(__file__).parent.parent


def load_scene_and_labels(
    dataset_path: Path | None, scene_index: int
) -> tuple[Scene, Tensor, Tensor]:
    """A scene with its labeled configurations q (N, n) and labels (N,)."""
    if dataset_path is None:
        config = load_config(ROOT / "configs" / "default_scene.yaml")
        scene = SceneVisualizer.from_dict(config["scene_visualizer"]).scene
        q, labels = scene.sample_configurations(SamplingConfig())
        return scene, q, labels
    entry = torch.load(dataset_path)["scenes"][scene_index]
    if "metadata" not in entry:
        raise SystemExit(f"{dataset_path} stores no scenes; only eval.pt does")
    return Scene.from_dict(entry["metadata"]), entry["train_q"], entry["train_y"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an MLP field to one scene.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fit_mlp.yaml")
    parser.add_argument("--dataset", type=Path, default=None, help="e.g. data/scenes/eval.pt")
    parser.add_argument("--scene-index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = MLPConfig.from_dict(config["model"])
    training_config = TrainingConfig.from_dict(config["training"])

    torch.manual_seed(training_config.seed)  # the sampling and the network initialization
    scene, q, labels = load_scene_and_labels(args.dataset, args.scene_index)
    print(f"scene: {scene.obstacle}")
    print(f"{len(q)} labeled configurations, {(labels < 0).float().mean():.1%} in collision")

    field = NeuralField(scene.robot, MLP(embedding_size(scene.robot), model_config))
    start = time.perf_counter()
    history = fit_field(field, q, labels, training_config)
    elapsed = time.perf_counter() - start
    epochs, device = training_config.epochs, training_config.device
    print(f"trained {epochs} epochs on {device} in {elapsed:.1f} s")

    ground_truth = scene.cspace_sdf(256)
    prediction = evaluate_on_grid(field, 256, device=training_config.device)
    metrics = field_metrics(prediction, ground_truth)
    print("on the 256 x 256 grid:", ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    visualizer = SceneVisualizer(scene=scene)
    visualizer.plot_overview(ground_truth, fields={"MLP field d(q)": prediction})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    visualizer.plot_samples(
        q, labels, axes[0], ground_truth=ground_truth, title="training data"
    )
    axes[1].plot(range(1, len(history) + 1), history)
    axes[1].set(xlabel="epoch", ylabel="loss", yscale="log", title="training loss")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
