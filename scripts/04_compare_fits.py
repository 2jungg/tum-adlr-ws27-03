"""Fit one field per config to the same scene and training data, and compare them. By default:
the MLP with MSE, the MLP with MSE + Eikonal, and Co-SDF (SLL with hKR + Eikonal)."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from metacosdf.fields import NeuralField, embedding_size, evaluate_on_grid
from metacosdf.losses import LossConfig
from metacosdf.metrics import field_metrics
from metacosdf.modules import ModuleConfig
from metacosdf.scene_generator.scene import SamplingConfig, Scene
from metacosdf.training import TrainingConfig, fit_field
from metacosdf.visualization import plot_overview, plot_samples

ROOT = Path(__file__).parent.parent
DEFAULT_CONFIGS = [
    ROOT / "configs" / name
    for name in ("fit_mlp_no_eikonal.yaml", "fit_mlp.yaml", "fit_cosdf.yaml")
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--dataset", type=Path, help="eval.pt, the only split that has scenes")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="for sampling the training data")
    parser.add_argument("--device", help="overrides training.device of every config, e.g. cuda")
    parser.add_argument("--save-dir", type=Path, help="save figures instead of showing them")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.dataset is None:
        scene = Scene.from_yaml(ROOT / "configs" / "default_scene.yaml")
        q, labels = scene.sample_configurations(SamplingConfig())
    else:
        entry = torch.load(args.dataset)["scenes"][args.scene_index]
        scene = Scene.from_dict(entry["metadata"])
        q, labels = entry["train_q"], entry["train_y"]
    print(f"{len(q)} samples, {(labels < 0).float().mean():.1%} in collision")
    ground_truth = scene.cspace_sdf()

    predictions: dict[str, Tensor] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for path in args.configs:
        name = path.stem
        module_config = ModuleConfig.from_yaml(path)
        loss_config = LossConfig.from_yaml(path)
        training_config = TrainingConfig.from_yaml(path)
        if args.device is not None:
            training_config.device = args.device
        print(f"\n{name}: {module_config}, {loss_config}")
        # the same initialization whatever the order of the configs
        torch.manual_seed(training_config.seed)
        network = module_config.create_module(embedding_size(scene.robot))
        field = NeuralField(scene.robot, network)
        start = time.perf_counter()
        histories[name] = fit_field(field, q, labels, loss_config, training_config)
        seconds = time.perf_counter() - start
        predictions[name] = evaluate_on_grid(field, device=training_config.device)
        metrics[name] = {**field_metrics(predictions[name], ground_truth), "seconds": seconds}

    columns = list(next(iter(metrics.values())))
    print("\n" + " " * 20 + "".join(f"{column:>10s}" for column in columns))
    for name, values in metrics.items():
        print(f"{name:20s}" + "".join(f"{values[column]:10.3f}" for column in columns))

    overview = plot_overview(
        scene,
        ground_truth,
        fields={
            f"{name}, Spearman {metrics[name]['spearman']:.3f}": prediction
            for name, prediction in predictions.items()
        },
    )
    training, axes = plt.subplots(
        1, len(histories) + 1, figsize=(5.2 * (len(histories) + 1), 4.6), squeeze=False
    )
    plot_samples(q, labels, ax=axes[0, 0], ground_truth=ground_truth, title="training data")
    for ax, (name, history) in zip(axes[0, 1:], histories.items(), strict=True):
        epochs = range(1, len(history) + 1)
        for term in history[0]:
            ax.plot(epochs, [epoch_terms[term] for epoch_terms in history], label=term)
        ax.set(title=name, xlabel="epoch", ylabel="mean over the epoch")
        ax.legend()
    training.tight_layout()

    if args.save_dir is None:
        plt.show()
    else:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        overview.savefig(args.save_dir / "overview.png", dpi=150, bbox_inches="tight")
        training.savefig(args.save_dir / "training.png", dpi=150, bbox_inches="tight")
        print(f"figures saved in {args.save_dir}")
