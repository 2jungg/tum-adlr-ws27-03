"""Visualization utilities for 2D workspace and C-space distance fields.
This part is written with AI to make clear plot image"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from torch import Tensor

from metacosdf.scene_generator.arm import PlanarArm
from metacosdf.scene_generator.scene import Scene, cspace_grid

TWO_PI = 2.0 * math.pi
FIELD_CMAP = "RdBu_r"
OBSTACLE_COLOR = "#2166ac"
FREE_COLOR = "#008300"
COLLISION_COLOR = "#e34948"
POSE_COLORS = ["#eb6834", "#1baf7a", "#6250d6", "#eda100"]


def _add_colorbar(ax: plt.Axes, im, vmin: float, vmax: float, label: str | None = None):
    """Attach a diverging colorbar with ticks on both sides of zero."""
    span = vmax - vmin
    ticks = [0.0]
    if -vmin > 0.05 * span:
        ticks += np.linspace(vmin, 0.0, 3)[:-1].tolist()
    if vmax > 0.05 * span:
        ticks += np.linspace(0.0, vmax, 3)[1:].tolist()
    return plt.colorbar(im, ax=ax, shrink=0.82, ticks=sorted(ticks), format="%.2f", label=label)


@dataclass
class SceneVisualizer:
    """Visualizer for a 2D workspace scene and its C-space fields."""

    scene: Scene
    arm: PlanarArm | None = None

    def _workspace_grid(self, res: int = 300) -> tuple[Tensor, Tensor]:
        xmin, xmax, ymin, ymax = self.scene.bounds
        return torch.meshgrid(
            torch.linspace(xmin, xmax, res),
            torch.linspace(ymin, ymax, res),
            indexing="ij",
        )

    def _draw_arm(
        self,
        ax: plt.Axes,
        q: Tensor,
        color: str,
        label: str | None = None,
    ) -> None:
        assert self.arm is not None
        p = self.arm.joint_positions(q).detach().cpu().numpy()
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=5, solid_capstyle="round", zorder=3)
        ax.plot(p[:-1, 0], p[:-1, 1], "o", color="black", ms=4, zorder=4)
        if label is not None:
            ax.annotate(label, p[-1], xytext=(5, 4), textcoords="offset points", fontsize=9, weight="bold", zorder=5)

    def plot_workspace(
        self,
        ax: plt.Axes | None = None,
        qs: Tensor | None = None,
        show_sdf: bool = False,
        res: int = 300,
        title: str = "Workspace",
    ) -> plt.Axes:
        """Plot workspace obstacles, optional workspace SDF, arm reach, and sample poses."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.2))

        X, Y = self._workspace_grid(res=res)
        D = self.scene.sdf(torch.stack([X, Y], dim=-1)).detach().cpu().numpy()
        X_np, Y_np = X.numpy(), Y.numpy()

        if show_sdf:
            vmin, vmax = min(float(D.min()), -1e-6), max(float(D.max()), 1e-6)
            im = ax.pcolormesh(X_np, Y_np, D, cmap=FIELD_CMAP, norm=TwoSlopeNorm(0.0, vmin, vmax), shading="nearest")
            if D.max() > 0.5:
                ax.contour(X_np, Y_np, D, levels=np.arange(0.5, D.max(), 0.5), colors="black", linewidths=0.4, alpha=0.35)
            _add_colorbar(ax, im, vmin, vmax, label="workspace SDF")
        elif D.min() < 0.0:
            ax.contourf(X_np, Y_np, D, levels=[float(D.min()) - 1.0, 0.0], colors=[OBSTACLE_COLOR], alpha=0.85)

        if D.min() < 0.0 < D.max():
            ax.contour(X_np, Y_np, D, levels=[0.0], colors="black", linewidths=1.2)

        if self.arm is not None:
            reach_patch = plt.Circle(
                self.arm.base,
                self.arm.reach + self.arm.link_radius,
                fill=False,
                ls="--",
                color="0.5",
                lw=1.0,
            )
            ax.add_patch(reach_patch)

            if qs is not None:
                qs_batch = qs.unsqueeze(0) if qs.ndim == 1 else qs
                labels = self.scene.collision_labels(self.arm, qs_batch)
                for idx, (q, y) in enumerate(zip(qs_batch, labels)):
                    color = FREE_COLOR if y > 0 else COLLISION_COLOR
                    self._draw_arm(ax, q, color=color, label=str(idx + 1))

        xmin, xmax, ymin, ymax = self.scene.bounds
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xlabel="x", ylabel="y", title=title)
        ax.grid(True, linestyle=":", alpha=0.4)
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
            if self.arm is None:
                raise ValueError("arm must be provided to compute ground-truth C-space SDF.")
            values = self.scene.cspace_sdf(self.arm, n=n)

        grid_n = values.shape[0]
        Q = cspace_grid(grid_n).numpy()
        V = values.detach().cpu().numpy()

        vmin, vmax = min(float(V.min()), -1e-6), max(float(V.max()), 1e-6)
        im = ax.pcolormesh(Q[..., 0], Q[..., 1], V, cmap=FIELD_CMAP, norm=TwoSlopeNorm(0.0, vmin, vmax), shading="nearest")

        if V.min() < 0.0 < V.max():
            ax.contour(Q[..., 0], Q[..., 1], V, levels=[0.0], colors="black", linewidths=1.3)

        if boundary is not None:
            B = boundary.detach().cpu().numpy()
            if B.min() < 0.0 < B.max():
                ax.contour(Q[..., 0], Q[..., 1], B, levels=[0.0], colors=FREE_COLOR, linewidths=1.5, linestyles="--")

        if qs is not None:
            qs_batch = qs.unsqueeze(0) if qs.ndim == 1 else qs
            for idx, q in enumerate(qs_batch):
                q1, q2 = torch.remainder(q, TWO_PI).tolist()
                ax.plot(q1, q2, "o", color=POSE_COLORS[idx % len(POSE_COLORS)], mec="black", mew=1.2, ms=8)
                ax.annotate(str(idx + 1), (q1, q2), xytext=(6, 4), textcoords="offset points", fontsize=9, weight="bold")

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
        if self.arm is None:
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            self.plot_workspace(ax=ax, show_sdf=show_ws_sdf)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
            self.plot_workspace(ax=axes[0], qs=qs, show_sdf=show_ws_sdf, title="Workspace (green=free, red=collision)")
            self.plot_cspace(ax=axes[1], qs=qs, n=n)

        fig.tight_layout()
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    from metacosdf.scene_generator.obstacles import Box, Circle

    arm = PlanarArm()
    scene = Scene(
        obstacles=(
            Circle(center=(1.2, 0.5), radius=0.35),
            Box(center=(-0.8, 0.9), half_size=(0.3, 0.35), angle=0.5),
        )
    )
    qs = torch.tensor([
        [0.2, 0.2],
        [2.2, 0.4],
        [4.2, 0.8],
    ])
    SceneVisualizer(scene=scene, arm=arm).show(qs=qs, show_ws_sdf=False)
