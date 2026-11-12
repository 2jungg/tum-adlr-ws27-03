"""Visualization utilities for 2D workspace and C-space distance fields."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from torch import Tensor

from metacosdf.scene_generator.scene import Scene, cspace_grid


TWO_PI = 2.0 * math.pi


def _add_colorbar(ax: plt.Axes, im, vmin: float, vmax: float, label: str | None = None):
    """Attach a diverging colorbar with ticks on both sides of zero."""
    span = vmax - vmin
    ticks = [0.0]
    if -vmin > 0.05 * span:
        ticks += np.linspace(vmin, 0.0, 3)[:-1].tolist()
    if vmax > 0.05 * span:
        ticks += np.linspace(0.0, vmax, 3)[1:].tolist()
    return plt.colorbar(im, ax=ax, shrink=0.82, ticks=sorted(ticks), format="%.2f", label=label)


@dataclass(frozen=True)
class SceneVisualizer:
    """Visualizer for a 2D workspace scene and its C-space fields."""

    scene: Scene
    field_color_map: str = "RdBu_r"
    obstacle_color: str = "#2166ac"
    free_color: str = "#008300"
    collision_color: str = "#e34948"
    pose_colors: list[str] = field(
        default_factory= lambda: ["#eb6834", "#1baf7a", "#6250d6", "#eda100"]
    )

    def _draw_arm(
        self,
        ax: plt.Axes,
        q: Tensor,
        color: str,
        label: str | None = None,
    ) -> None:
        p = self.scene.robot.joint_positions(q).detach().cpu().numpy()
        ax.plot(
            p[:, 0],
            p[:, 1],
            "-",
            color=color,
            lw=5,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )
        ax.plot(p[:-1, 0], p[:-1, 1], "o", color="black", ms=4, zorder=4)
        if label is not None:
            ax.annotate(
                label,
                p[-1],
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                zorder=5,
            )


    def plot_workspace(
        self,
        ax: plt.Axes | None = None,
        qs: Tensor | None = None,
        show_sdf: bool = False,
        resolution: int = 300,
    ) -> plt.Axes:
        """Plot workspace obstacles, optional workspace SDF, arm reach, and sample poses."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        X, Y = self.scene.workspace_grid(resolution=resolution)
        D = self.scene.obstacle.sdf(torch.stack([X, Y], dim=-1)).detach().cpu().numpy()
        X_np, Y_np = X.numpy(), Y.numpy()

        if show_sdf:
            vmin, vmax = min(float(D.min()), -1e-6), max(float(D.max()), 1e-6)
            im = ax.pcolormesh(
                X_np,
                Y_np,
                D,
                cmap=self.field_color_map,
                norm=TwoSlopeNorm(0.0, vmin, vmax),
                shading="nearest",
            )
            if D.max() > 0.5:
                ax.contour(
                    X_np,
                    Y_np,
                    D,
                    levels=np.arange(0.5, D.max(), 0.5),
                    colors="black",
                    linewidths=0.4,
                    alpha=0.35,
                )
            _add_colorbar(ax, im, vmin, vmax, label="workspace SDF")
        elif D.min() < 0.0:
            ax.contourf(
                X_np,
                Y_np,
                D,
                levels=[float(D.min()) - 1.0, 0.0],
                colors=[self.obstacle_color],
                alpha=0.85,
            )

        if D.min() < 0.0 < D.max():
            ax.contour(X_np, Y_np, D, levels=[0.0], colors="black", linewidths=1.2)

        if not show_sdf:
            reach_patch = plt.Circle(
                (self.scene.robot.base_position_x, self.scene.robot.base_position_y),
                self.scene.robot.reach + self.scene.robot.link_radius,
                fill=False,
                ls="--",
                color="0.5",
                lw=1.0,
            )
            ax.add_patch(reach_patch)

            if qs is not None:
                qs_batch = qs.unsqueeze(0) if qs.ndim == 1 else qs
                labels = self.scene.collision_labels(qs_batch)
                for idx, (q, y) in enumerate(zip(qs_batch, labels)):
                    color = self.free_color if y > 0 else self.collision_color
                    self._draw_arm(ax, q, color=color, label=str(idx + 1))

        ax.set(
            xlim=(self.scene.bounds.xmin, self.scene.bounds.xmax),
            ylim=(self.scene.bounds.ymin, self.scene.bounds.ymax),
            aspect="equal",
            xlabel="x",
            ylabel="y",
        )
        return ax

    def plot_cspace(
        self,
        ax: plt.Axes | None = None,
        values: Tensor | None = None,
        boundary: Tensor | None = None,
        qs: Tensor | None = None,
        n: int = 256,
        title: str = "C-Space Ground-Truth SDF [rad]",
    ) -> plt.Axes:
        """Plot a 2D C-space scalar field (defaults to ground-truth C-space SDF)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        if values is None:
            values = self.scene.cspace_sdf(n=n)

        grid_n = values.shape[0]
        Q = cspace_grid(grid_n).numpy()
        V = values.detach().cpu().numpy()

        vmin, vmax = min(float(V.min()), -1e-6), max(float(V.max()), 1e-6)
        im = ax.pcolormesh(
            Q[..., 0],
            Q[..., 1],
            V,
            cmap=self.field_color_map,
            norm=TwoSlopeNorm(0.0, vmin, vmax),
            shading="nearest",
        )

        if V.min() < 0.0 < V.max():
            ax.contour(Q[..., 0], Q[..., 1], V, levels=[0.0], colors="black", linewidths=1.3)

        if boundary is not None:
            B = boundary.detach().cpu().numpy()
            if B.min() < 0.0 < B.max():
                ax.contour(
                    Q[..., 0],
                    Q[..., 1],
                    B,
                    levels=[0.0],
                    colors=self.free_color,
                    linewidths=1.5,
                    linestyles="--",
                )

        if qs is not None:
            qs_batch = qs.unsqueeze(0) if qs.ndim == 1 else qs
            for idx, q in enumerate(qs_batch):
                q1, q2 = torch.remainder(q, TWO_PI).tolist()
                ax.plot(
                    q1,
                    q2,
                    "o",
                    color=self.pose_colors[idx % len(self.pose_colors)],
                    mec="black",
                    mew=1.2,
                    ms=8,
                )
                ax.annotate(
                    str(idx + 1),
                    (q1, q2),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=9,
                    weight="bold",
                )

        ticks = [0.0, math.pi, TWO_PI]
        ax.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=["0", "π", "2π"],
            yticklabels=["0", "π", "2π"],
            xlabel=r"$\theta_1$ [rad]",
            ylabel=r"$\theta_2$ [rad]",
            title=title,
            aspect="equal",
        )
        _add_colorbar(ax, im, vmin, vmax)
        return ax

    def show(
        self,
        qs: Tensor | None = None,
        n: int = 256,
        show_ws_sdf: bool = False,
        save_path: str | Path | None = None,
    ) -> None:
        """Display workspace and C-space GT SDF side by side, or save to file if save_path is given."""

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
        self.plot_workspace(ax=axes[0], qs=qs, show_sdf=show_ws_sdf)
        axes[0].set_title("Workspace (green=free, red=collision)")
        self.plot_cspace(ax=axes[1], qs=qs, n=n)

        fig.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
