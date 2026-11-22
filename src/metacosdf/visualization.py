"""Workspace and C-space plots. C-space fields are passed as values on cspace_grid(res)."""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patheffects import Normal, Stroke
from torch import Tensor

from metacosdf.config_utils import Config
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.scene import Scene, cspace_grid
from metacosdf.tensor_utils import to_numpy


@dataclass
class SceneVisualizationConfig(Config):
    """Shared by all panels, so that a pose keeps its color and number everywhere."""

    field_color_map: str = "RdBu_r"
    obstacle_color: str = "#2166ac"
    free_color: str = "#008300"
    collision_color: str = "#e34948"
    pose_colors: tuple[str, ...] = ("#eb6834", "#1baf7a", "#6250d6", "#eda100")
    # behind the text labels, readable on any background
    label_background_color: str = "white"
    label_background_alpha: float = 0.75

    def pose_color(self, index: int) -> str:
        return self.pose_colors[index % len(self.pose_colors)]

    @property
    def label_box(self) -> dict[str, str | float]:
        """matplotlib bbox of the text labels"""
        return {
            "boxstyle": "round,pad=0.15",
            "facecolor": self.label_background_color,
            "edgecolor": "none",
            "alpha": self.label_background_alpha,
        }


def _diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """centered at 0; padded because TwoSlopeNorm needs vmin < 0 < vmax"""
    return TwoSlopeNorm(0.0, min(float(values.min()), -1e-6), max(float(values.max()), 1e-6))


def _add_colorbar(ax: plt.Axes, mappable, label: str | None = None):
    """ticks on both sides of 0, the default ticks can miss the shorter side"""
    vmin, vmax = mappable.norm.vmin, mappable.norm.vmax
    span = vmax - vmin
    ticks = [0.0]
    if -vmin > 0.05 * span:
        ticks += np.linspace(vmin, 0.0, 3)[:-1].tolist()
    if vmax > 0.05 * span:
        ticks += np.linspace(0.0, vmax, 3)[1:].tolist()
    return ax.figure.colorbar(
        mappable, ax=ax, shrink=0.8, ticks=sorted(ticks), format="%.2f", label=label
    )


def _draw_zero_level(
    ax: plt.Axes, horizontal: np.ndarray, vertical: np.ndarray, values: np.ndarray, **kwargs
) -> None:
    """only if the values change sign, matplotlib warns otherwise"""
    if values.min() < 0.0 < values.max():
        ax.contour(horizontal, vertical, values, levels=[0.0], **kwargs)


def _format_cspace_axes(ax: plt.Axes, title: str) -> None:
    """[0, 2pi)^2, q1 horizontal"""
    ticks = [0.0, math.pi, math.tau]
    ax.set(
        xlim=(0.0, math.tau),
        ylim=(0.0, math.tau),
        xticks=ticks,
        yticks=ticks,
        xticklabels=["0", "π", "2π"],
        yticklabels=["0", "π", "2π"],
        xlabel=r"$q_1$ [rad]",
        ylabel=r"$q_2$ [rad]",
        title=title,
        aspect="equal",
    )


def _draw_arm(
    ax: plt.Axes,
    robot: PlanarArm,
    q: Tensor,
    color: str,
    text: str,
    text_color: str,
    config: SceneVisualizationConfig,
) -> None:
    """q (n,), text at the tip"""
    p = to_numpy(robot.joint_positions(q))  # (n+1, 2): base, joints, tip
    outline = [Stroke(linewidth=7, foreground="black"), Normal()]
    ax.plot(
        p[:, 0],
        p[:, 1],
        "-",
        color=color,
        lw=5,
        solid_capstyle="round",
        path_effects=outline,
        zorder=3,
    )
    ax.plot(p[:-1, 0], p[:-1, 1], "o", color="black", ms=4, zorder=4)
    ax.annotate(
        text,
        p[-1],
        xytext=(5, 4),
        textcoords="offset points",
        color=text_color,
        fontsize=9,
        weight="bold",
        bbox=config.label_box,
        zorder=5,
    )


def plot_workspace(
    scene: Scene,
    *,
    ax: plt.Axes | None = None,
    qs: Tensor | None = None,
    show_sdf: bool = False,
    resolution: int = 300,
    title: str = "",
    config: SceneVisualizationConfig | None = None,
) -> plt.Axes:
    """obstacles or, with show_sdf, the workspace SDF; reach dashed; qs (n,) or (k, n)"""
    if config is None:
        config = SceneVisualizationConfig()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.2))

    grid = scene.workspace_grid(resolution)  # (res, res, 2)
    sdf_values = to_numpy(scene.obstacle.sdf(grid))  # (res, res)
    grid_x, grid_y = to_numpy(grid[..., 0]), to_numpy(grid[..., 1])

    if show_sdf:
        mesh = ax.pcolormesh(
            grid_x,
            grid_y,
            sdf_values,
            cmap=config.field_color_map,
            norm=_diverging_norm(sdf_values),
            shading="nearest",
        )
        iso_levels = np.arange(0.5, sdf_values.max(), 0.5)
        if iso_levels.size > 0:
            ax.contour(
                grid_x,
                grid_y,
                sdf_values,
                levels=iso_levels,
                colors="black",
                linewidths=0.4,
                alpha=0.35,
            )
        _add_colorbar(ax, mesh, label="workspace SDF")
    elif sdf_values.min() < 0.0:
        ax.contourf(
            grid_x,
            grid_y,
            sdf_values,
            levels=[sdf_values.min() - 1.0, 0.0],
            colors=[config.obstacle_color],
        )
    _draw_zero_level(ax, grid_x, grid_y, sdf_values, colors="black", linewidths=1.0)

    robot = scene.robot
    ax.add_patch(
        plt.Circle(
            robot.base_position,
            robot.reach + robot.link_radius,
            fill=False,
            ls="--",
            color="0.5",
            lw=1.0,
        )
    )

    if qs is not None:
        qs = torch.atleast_2d(qs)  # (k, n)
        for index, (q, label) in enumerate(zip(qs, scene.collision_labels(qs), strict=False)):
            free = bool(label > 0)
            _draw_arm(
                ax,
                robot,
                q,
                config.pose_color(index),
                text=f"{index + 1}: {'free' if free else 'collision'}",
                text_color=config.free_color if free else config.collision_color,
                config=config,
            )

    bounds = scene.bounds
    ax.set(
        xlim=(bounds.xmin, bounds.xmax),
        ylim=(bounds.ymin, bounds.ymax),
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title=title,
    )
    return ax


def plot_cspace(
    values: Tensor,
    *,
    ax: plt.Axes | None = None,
    boundary: Tensor | None = None,
    qs: Tensor | None = None,
    title: str = "",
    colorbar_label: str | None = None,
    config: SceneVisualizationConfig | None = None,
) -> plt.Axes:
    """values (res, res); the zero level of boundary (e.g. ground truth) is dashed"""
    if config is None:
        config = SceneVisualizationConfig()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.2))

    grid = cspace_grid(values.shape[0])  # (res, res, 2)
    grid_q1, grid_q2 = to_numpy(grid[..., 0]), to_numpy(grid[..., 1])
    field_values = to_numpy(values)
    mesh = ax.pcolormesh(
        grid_q1,
        grid_q2,
        field_values,
        cmap=config.field_color_map,
        norm=_diverging_norm(field_values),
        shading="nearest",
    )
    _draw_zero_level(ax, grid_q1, grid_q2, field_values, colors="black", linewidths=1.2)
    if boundary is not None:
        _draw_zero_level(
            ax,
            grid_q1,
            grid_q2,
            to_numpy(boundary),
            colors=config.free_color,
            linewidths=1.5,
            linestyles="--",
        )

    if qs is not None:
        for index, q in enumerate(torch.remainder(torch.atleast_2d(qs), math.tau)):
            q1, q2 = q.tolist()
            ax.plot(q1, q2, "o", color=config.pose_color(index), mec="black", mew=1.2, ms=9)
            ax.annotate(
                str(index + 1),
                (q1, q2),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=10,
                weight="bold",
                bbox=config.label_box,
            )

    _format_cspace_axes(ax, title)
    _add_colorbar(ax, mesh, label=colorbar_label)
    return ax


def plot_samples(
    q_samples: Tensor,
    y_samples: Tensor,
    *,
    ax: plt.Axes | None = None,
    ground_truth: Tensor | None = None,
    title: str | None = None,
    config: SceneVisualizationConfig | None = None,
) -> plt.Axes:
    """q_samples (N, n), labels y_samples (N,); ground_truth shades the C-obstacles"""
    if config is None:
        config = SceneVisualizationConfig()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.2))

    if ground_truth is not None:
        grid = cspace_grid(ground_truth.shape[0])  # (res, res, 2)
        grid_q1, grid_q2 = to_numpy(grid[..., 0]), to_numpy(grid[..., 1])
        values = to_numpy(ground_truth)
        if values.min() < 0.0:
            ax.contourf(
                grid_q1,
                grid_q2,
                values,
                levels=[values.min() - 1.0, 0.0],
                colors=[config.obstacle_color],
                alpha=0.2,
            )
        _draw_zero_level(ax, grid_q1, grid_q2, values, colors="black", linewidths=1.2)

    q_wrapped = to_numpy(torch.remainder(q_samples, math.tau))  # (N, n)
    free = to_numpy(y_samples) > 0  # (N,)
    for mask, color, name in [
        (free, config.free_color, "free"),
        (~free, config.collision_color, "collision"),
    ]:
        ax.scatter(
            q_wrapped[mask, 0],
            q_wrapped[mask, 1],
            s=4,
            color=color,
            alpha=0.5,
            label=f"{name} ({mask.sum()})",
        )
    ax.legend(loc="upper right", fontsize=8, markerscale=2.5, framealpha=0.85)

    if title is None:
        title = f"sampled configurations (N={len(q_samples)})"
    _format_cspace_axes(ax, title)
    return ax


def plot_overview(
    scene: Scene,
    ground_truth: Tensor,
    *,
    qs: Tensor | None = None,
    fields: dict[str, Tensor] | None = None,
    ncols: int = 3,
    save_path: str | Path | None = None,
    config: SceneVisualizationConfig | None = None,
) -> plt.Figure:
    """workspace | ground truth | one panel per field, ground-truth boundary dashed"""
    if config is None:
        config = SceneVisualizationConfig()
    fields = fields or {}
    n_panels = 2 + len(fields)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.ravel()

    plot_workspace(
        scene,
        ax=axes[0],
        qs=qs,
        title="workspace, poses and the arm's reach (dashed)",
        config=config,
    )
    plot_cspace(
        ground_truth, ax=axes[1], qs=qs, title="ground-truth C-space SDF [rad]", config=config
    )
    for ax, (name, values) in zip(axes[2:], fields.items(), strict=False):
        plot_cspace(values, ax=ax, boundary=ground_truth, qs=qs, title=name, config=config)
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
