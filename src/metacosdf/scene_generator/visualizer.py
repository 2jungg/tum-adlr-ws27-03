"""Plots of a 2D workspace scene and of scalar fields on its C-space.

Every plot method takes an `ax` (or creates one) and returns it, so panels can be combined freely.
A C-space field is passed as its values on `cspace_grid(n)`: an (n, n) tensor with V[i, j] = field(q1_i, q2_j).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from torch import Tensor

from metacosdf.scene_generator.scene import Scene, cspace_grid

TWO_PI = 2.0 * math.pi
LABEL_BOX = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)  # readable on any background


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Color normalization centered at 0. TwoSlopeNorm needs vmin < 0 < vmax, so the limits are padded."""
    return TwoSlopeNorm(0.0, min(float(values.min()), -1e-6), max(float(values.max()), 1e-6))


def _add_colorbar(ax: plt.Axes, im, label: str | None = None):
    """Colorbar with ticks on both sides of 0. TwoSlopeNorm stretches each side separately,
    so matplotlib's default ticks can leave the shorter side unlabeled."""
    vmin, vmax = im.norm.vmin, im.norm.vmax
    span = vmax - vmin
    ticks = [0.0]
    if -vmin > 0.05 * span:
        ticks += np.linspace(vmin, 0.0, 3)[:-1].tolist()
    if vmax > 0.05 * span:
        ticks += np.linspace(0.0, vmax, 3)[1:].tolist()
    return ax.figure.colorbar(im, ax=ax, shrink=0.8, ticks=sorted(ticks), format="%.2f", label=label)


def _draw_zero_level(ax: plt.Axes, X: np.ndarray, Y: np.ndarray, V: np.ndarray, **kwargs) -> None:
    """Zero contour of V, drawn only if V changes sign (matplotlib warns otherwise)."""
    if V.min() < 0.0 < V.max():
        ax.contour(X, Y, V, levels=[0.0], **kwargs)


@dataclass(frozen=True)
class SceneVisualizer:
    """Plots for one scene: its workspace (obstacles, arm, workspace SDF) and fields on its C-space.

    Poses keep the same color and number in every panel: an arm in the workspace, a marker in C-space.
    """

    scene: Scene
    field_color_map: str = "RdBu_r"
    obstacle_color: str = "#2166ac"
    free_color: str = "#008300"
    collision_color: str = "#e34948"
    pose_colors: tuple[str, ...] = ("#eb6834", "#1baf7a", "#6250d6", "#eda100")

    def _pose_color(self, k: int) -> str:
        return self.pose_colors[k % len(self.pose_colors)]

    def _draw_arm(self, ax: plt.Axes, q: Tensor, color: str, label: str, label_color: str) -> None:
        """One pose q (n,): the links with a black outline, base and joints as dots, the label at the tip."""
        p = _to_numpy(self.scene.robot.joint_positions(q))  # (n+1, 2): base, joints, tip
        outline = [pe.Stroke(linewidth=7, foreground="black"), pe.Normal()]
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=5, solid_capstyle="round", path_effects=outline, zorder=3)
        ax.plot(p[:-1, 0], p[:-1, 1], "o", color="black", ms=4, zorder=4)
        ax.annotate(label, p[-1], xytext=(5, 4), textcoords="offset points", color=label_color,
                    fontsize=9, weight="bold", bbox=LABEL_BOX, zorder=5)

    def plot_workspace(
        self,
        ax: plt.Axes | None = None,
        qs: Tensor | None = None,
        show_sdf: bool = False,
        resolution: int = 300,
        title: str = "",
    ) -> plt.Axes:
        """Obstacles (sdf < 0), or the workspace SDF as a heatmap with iso-lines every 0.5,
        plus the arm's reach (dashed) and the poses qs (n,) or (k, n), labeled free or collision."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        X, Y = self.scene.workspace_grid(resolution)  # each (res, res)
        D = _to_numpy(self.scene.obstacle.sdf(torch.stack([X, Y], dim=-1)))  # (res, res, 2) -> (res, res)
        X, Y = _to_numpy(X), _to_numpy(Y)

        if show_sdf:
            im = ax.pcolormesh(X, Y, D, cmap=self.field_color_map, norm=_diverging_norm(D), shading="nearest")
            iso_levels = np.arange(0.5, D.max(), 0.5)
            if iso_levels.size > 0:
                ax.contour(X, Y, D, levels=iso_levels, colors="black", linewidths=0.4, alpha=0.35)
            _add_colorbar(ax, im, label="workspace SDF")
        elif D.min() < 0.0:
            ax.contourf(X, Y, D, levels=[D.min() - 1.0, 0.0], colors=[self.obstacle_color])
        _draw_zero_level(ax, X, Y, D, colors="black", linewidths=1.0)

        robot = self.scene.robot
        base = (robot.base_position_x, robot.base_position_y)
        ax.add_patch(plt.Circle(base, robot.reach + robot.link_radius, fill=False, ls="--", color="0.5", lw=1.0))

        if qs is not None:
            qs = torch.atleast_2d(qs)  # (k, n)
            for k, (q, y) in enumerate(zip(qs, self.scene.collision_labels(qs))):
                free = bool(y > 0)
                self._draw_arm(
                    ax, q, self._pose_color(k),
                    label=f"{k + 1}: {'free' if free else 'collision'}",
                    label_color=self.free_color if free else self.collision_color,
                )

        bounds = self.scene.bounds
        ax.set(xlim=(bounds.xmin, bounds.xmax), ylim=(bounds.ymin, bounds.ymax), aspect="equal",
               xlabel="x", ylabel="y", title=title)
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
        """A C-space field from its values (n, n) on cspace_grid(n); q1 is horizontal, q2 vertical.

        Solid black: the field's own zero level set. Dashed: the zero level set of `boundary` (n, n),
        e.g. the ground truth, for comparison. qs (n,) or (k, n): numbered markers in the pose colors.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        Q = _to_numpy(cspace_grid(values.shape[0]))  # (n, n, 2), "ij" indexing: Q[i, j] = (q1_i, q2_j)
        Q1, Q2 = Q[..., 0], Q[..., 1]
        V = _to_numpy(values)
        # explicit coordinates: no transpose and no origin flip, unlike imshow
        im = ax.pcolormesh(Q1, Q2, V, cmap=self.field_color_map, norm=_diverging_norm(V), shading="nearest")
        _draw_zero_level(ax, Q1, Q2, V, colors="black", linewidths=1.2)
        if boundary is not None:
            _draw_zero_level(ax, Q1, Q2, _to_numpy(boundary), colors=self.free_color, linewidths=1.5,
                             linestyles="--")

        if qs is not None:
            for k, q in enumerate(torch.remainder(torch.atleast_2d(qs), TWO_PI)):  # wrap into [0, 2pi)
                q1, q2 = q.tolist()
                ax.plot(q1, q2, "o", color=self._pose_color(k), mec="black", mew=1.2, ms=9)
                ax.annotate(str(k + 1), (q1, q2), xytext=(6, 4), textcoords="offset points",
                            fontsize=10, weight="bold", bbox=LABEL_BOX)

        ticks = [0.0, math.pi, TWO_PI]
        ax.set(xticks=ticks, yticks=ticks, xticklabels=["0", "π", "2π"], yticklabels=["0", "π", "2π"],
               xlabel=r"$q_1$ [rad]", ylabel=r"$q_2$ [rad]", title=title, aspect="equal")
        _add_colorbar(ax, im, label=colorbar_label)
        return ax

    def plot_overview(
        self,
        gt: Tensor,
        qs: Tensor | None = None,
        fields: dict[str, Tensor] | None = None,
        ncols: int = 3,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Workspace | ground-truth C-space SDF | one panel per field, in the style of Co-SDF Fig. 1.

        gt: (n, n) from `scene.cspace_sdf(n)`. fields: name -> (n, n) values on cspace_grid(n),
        e.g. a model evaluated on the grid; each field panel shows the ground-truth boundary dashed.
        The numbered poses appear in every panel. The figure is also saved if `save_path` is given.
        """
        fields = fields or {}
        n_panels = 2 + len(fields)
        ncols = min(ncols, n_panels)
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
        axes = axes.ravel()

        self.plot_workspace(axes[0], qs=qs, title="workspace, poses and the arm's reach (dashed)")
        self.plot_cspace(gt, axes[1], qs=qs, title="ground-truth C-space SDF [rad]")
        for ax, (name, values) in zip(axes[2:], fields.items()):
            self.plot_cspace(values, ax, boundary=gt, qs=qs, title=name)
        for ax in axes[n_panels:]:
            ax.axis("off")
        fig.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches="tight")
        return fig
