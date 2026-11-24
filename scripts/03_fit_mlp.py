"""Fit one field (module, loss and training from --config) to one scene: the default scene, or
an eval scene with --dataset."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from metacosdf.fields import NeuralField, embedding_size, evaluate_on_grid
from metacosdf.losses import LossConfig
from metacosdf.metrics import field_metrics
from metacosdf.modules import ModuleConfig
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

    module_config = ModuleConfig.from_yaml(args.config)
    loss_config = LossConfig.from_yaml(args.config)
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

    field = NeuralField(scene.robot, module_config.create_module(embedding_size(scene.robot)))
    start = time.perf_counter()
    history = fit_field(field, q, labels, loss_config, training_config)
    print(f"trained in {time.perf_counter() - start:.1f} s")

    ground_truth = scene.cspace_sdf()
    prediction = evaluate_on_grid(field, device=training_config.device)
    metrics = field_metrics(prediction, ground_truth)
    print(", ".join(f"{name} {value:.3f}" for name, value in metrics.items()))

    plot_overview(scene, ground_truth, fields={args.config.stem: prediction})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_samples(q, labels, ax=axes[0], ground_truth=ground_truth, title="training data")
    for term in history[0]:
        axes[1].plot([epoch_terms[term] for epoch_terms in history], label=term)
    axes[1].set(xlabel="epoch", ylabel="mean over the epoch")
    axes[1].legend()
    fig.tight_layout()
    plt.show()
