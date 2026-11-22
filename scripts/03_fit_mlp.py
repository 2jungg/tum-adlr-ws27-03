"""Fit an MLP field to one scene: the default scene, or an eval scene with --dataset."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from metacosdf.fields import NeuralField, embedding_size, evaluate_on_grid
from metacosdf.metrics import field_metrics
from metacosdf.models import MLP, MLPConfig
from metacosdf.scene_generator.scene import SamplingConfig, Scene
from metacosdf.training import TrainingConfig, fit_field
from metacosdf.visualization import plot_overview, plot_samples

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fit_mlp.yaml")
    parser.add_argument("--dataset", type=Path, help="eval.pt, the only split that has scenes")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--device", help="overrides training.device of the config, e.g. cuda")
    args = parser.parse_args()

    model_config = MLPConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)
    if args.device is not None:
        training_config.device = args.device
    torch.manual_seed(training_config.seed)

    if args.dataset is None:
        scene = Scene.from_yaml(ROOT / "configs" / "default_scene.yaml")
        q, labels = scene.sample_configurations(SamplingConfig())
    else:
        entry = torch.load(args.dataset)["scenes"][args.scene_index]
        scene = Scene.from_dict(entry["metadata"])
        q, labels = entry["train_q"], entry["train_y"]
    print(f"{len(q)} samples, {(labels < 0).float().mean():.1%} in collision")

    field = NeuralField(scene.robot, MLP(embedding_size(scene.robot), model_config))
    start = time.perf_counter()
    history = fit_field(field, q, labels, training_config)
    print(f"trained in {time.perf_counter() - start:.1f} s")

    ground_truth = scene.cspace_sdf()
    prediction = evaluate_on_grid(field, device=training_config.device)
    metrics = field_metrics(prediction, ground_truth)
    print(", ".join(f"{name} {value:.3f}" for name, value in metrics.items()))

    plot_overview(scene, ground_truth, fields={"MLP": prediction})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_samples(q, labels, ax=axes[0], ground_truth=ground_truth, title="training data")
    axes[1].plot(history)
    axes[1].set(xlabel="epoch", ylabel="loss", yscale="log")
    fig.tight_layout()
    plt.show()
