"""Plots of a 2D workspace scene and of scalar fields on its C-space.

Every plot method takes an `ax` (or creates one) and returns it,
so panels can be combined freely.
A C-space field is passed as its values on `cspace_grid(res)`: a (res, res) tensor with
values[i, j] = field(q1_i, q2_j).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patheffects import Normal, Stroke
from torch import Tensor

from metacosdf.scene_generator.scene import Scene, cspace_grid

TWO_PI = 2.0 * math.pi
# readable on any background
LABEL_BOX = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)


def _to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """
    Color normalization centered at 0. TwoSlopeNorm needs vmin < 0 < vmax,
    so the limits are padded.
    """
    return TwoSlopeNorm(0.0, min(float(values.min()), -1e-6), max(float(values.max()), 1e-6))


def _add_colorbar(ax: plt.Axes, mappable, label: str | None = None):
    """
    Colorbar with ticks on both sides of 0. TwoSlopeNorm stretches each side separately,
    so matplotlib's default ticks can leave the shorter side unlabeled.
    """
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
    """
    Zero contour of values (res, res) on a grid, drawn only if they change sign
    (matplotlib warns otherwise).
    """
    if values.min() < 0.0 < values.max():
        ax.contour(horizontal, vertical, values, levels=[0.0], **kwargs)


@dataclass
class SceneVisualizer:
    """
    Plots for one scene:
    its workspace (obstacles, arm, workspace SDF) and fields on its C-space.

    Poses keep the same color and number in every panel:
    an arm in the workspace, a marker in C-space.
    """

    scene: Scene
    field_color_map: str = "RdBu_r"
    obstacle_color: str = "#2166ac"
    free_color: str = "#008300"
    collision_color: str = "#e34948"
    pose_colors: tuple[str, ...] = ("#eb6834", "#1baf7a", "#6250d6", "#eda100")

    def _pose_color(self, index: int) -> str:
        return self.pose_colors[index % len(self.pose_colors)]

    def _draw_arm(
        self, ax: plt.Axes, q: Tensor, color: str, text: str, text_color: str
    ) -> None:
        """
        One pose q (n,): the links with a black outline, base and joints as dots,
        the text at the tip.
        """
        p = _to_numpy(self.scene.robot.joint_positions(q))  # (n+1, 2): base, joints, tip
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
            bbox=LABEL_BOX,
            zorder=5,
        )

    def plot_workspace(
        self,
        ax: plt.Axes | None = None,
        qs: Tensor | None = None,
        show_sdf: bool = False,
        resolution: int = 300,
        title: str = "",
    ) -> plt.Axes:
        """
        Obstacles (sdf < 0), or the workspace SDF as a heatmap with iso-lines every 0.5,
        plus the arm's reach (dashed) and the poses qs (n,) or (k, n),
        labeled free or collision.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        grid = self.scene.workspace_grid(resolution)  # (res, res, 2)
        sdf_values = _to_numpy(self.scene.obstacle.sdf(grid))  # (res, res)
        grid_x, grid_y = _to_numpy(grid[..., 0]), _to_numpy(grid[..., 1])

        if show_sdf:
            mesh = ax.pcolormesh(
                grid_x,
                grid_y,
                sdf_values,
                cmap=self.field_color_map,
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
                colors=[self.obstacle_color],
            )
        _draw_zero_level(ax, grid_x, grid_y, sdf_values, colors="black", linewidths=1.0)

        robot = self.scene.robot
        base = (robot.base_position_x, robot.base_position_y)
        ax.add_patch(
            plt.Circle(
                base, robot.reach + robot.link_radius, fill=False, ls="--", color="0.5", lw=1.0
            )
        )

        if qs is not None:
            qs = torch.atleast_2d(qs)  # (k, n)
            for index, (q, label) in enumerate(
                zip(qs, self.scene.collision_labels(qs), strict=False)
            ):
                free = bool(label > 0)
                self._draw_arm(
                    ax,
                    q,
                    self._pose_color(index),
                    text=f"{index + 1}: {'free' if free else 'collision'}",
                    text_color=self.free_color if free else self.collision_color,
                )

        bounds = self.scene.bounds
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
        self,
        values: Tensor,
        ax: plt.Axes | None = None,
        boundary: Tensor | None = None,
        qs: Tensor | None = None,
        title: str = "",
        colorbar_label: str | None = None,
    ) -> plt.Axes:
        """
        A C-space field from its values (res, res) on cspace_grid(res);
        q1 is horizontal, q2 vertical.

        Solid black: the field's own zero level set.
        Dashed: the zero level set of `boundary` (res, res),
        e.g. the ground truth, for comparison.
        qs (n,) or (k, n): numbered markers in the pose colors.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        grid = cspace_grid(values.shape[0])  # (res, res, 2)
        grid_q1, grid_q2 = _to_numpy(grid[..., 0]), _to_numpy(grid[..., 1])
        field_values = _to_numpy(values)
        # explicit coordinates: no transpose and no origin flip, unlike imshow
        mesh = ax.pcolormesh(
            grid_q1,
            grid_q2,
            field_values,
            cmap=self.field_color_map,
            norm=_diverging_norm(field_values),
            shading="nearest",
        )
        _draw_zero_level(ax, grid_q1, grid_q2, field_values, colors="black", linewidths=1.2)
        if boundary is not None:
            _draw_zero_level(
                ax,
                grid_q1,
                grid_q2,
                _to_numpy(boundary),
                colors=self.free_color,
                linewidths=1.5,
                linestyles="--",
            )

        if qs is not None:
            # wrap into [0, 2pi)
            for index, q in enumerate(torch.remainder(torch.atleast_2d(qs), TWO_PI)):
                q1, q2 = q.tolist()
                ax.plot(q1, q2, "o", color=self._pose_color(index), mec="black", mew=1.2, ms=9)
                ax.annotate(
                    str(index + 1),
                    (q1, q2),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=10,
                    weight="bold",
                    bbox=LABEL_BOX,
                )

        ticks = [0.0, math.pi, TWO_PI]
        ax.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=["0", "π", "2π"],
            yticklabels=["0", "π", "2π"],
            xlabel=r"$q_1$ [rad]",
            ylabel=r"$q_2$ [rad]",
            title=title,
            aspect="equal",
        )
        _add_colorbar(ax, mesh, label=colorbar_label)
        return ax

    def plot_overview(
        self,
        ground_truth: Tensor,
        qs: Tensor | None = None,
        fields: dict[str, Tensor] | None = None,
        ncols: int = 3,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """
        Workspace | ground-truth C-space SDF | one panel per field

        ground_truth: (res, res) from `scene.cspace_sdf(res)`.
        fields: name -> (res, res) values on cspace_grid(res),
        e.g. a model evaluated on the grid;
        each field panel shows the ground-truth boundary dashed.
        The numbered poses appear in every panel.
        The figure is also saved if `save_path` is given.
        """
        fields = fields or {}
        n_panels = 2 + len(fields)
        ncols = min(ncols, n_panels)
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False
        )
        axes = axes.ravel()

        self.plot_workspace(
            axes[0], qs=qs, title="workspace, poses and the arm's reach (dashed)"
        )
        self.plot_cspace(ground_truth, axes[1], qs=qs, title="ground-truth C-space SDF [rad]")
        for ax, (name, values) in zip(axes[2:], fields.items(), strict=False):
            self.plot_cspace(values, ax, boundary=ground_truth, qs=qs, title=name)
        for ax in axes[n_panels:]:
            ax.axis("off")
        fig.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches="tight")
        return fig
