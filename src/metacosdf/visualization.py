import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from metacosdf.robot import PlanarArm
from metacosdf.scene import Scene


def add_colorbar(ax, im, vmin: float, vmax: float, label: str | None = None):
    """Colorbar with ticks on both sides of 0."""
    span = vmax - vmin
    ticks = [0.0]
    if -vmin > 0.05 * span:
        ticks += np.linspace(vmin, 0, 3)[:-1].tolist()
    if vmax > 0.05 * span:
        ticks += np.linspace(0, vmax, 3)[1:].tolist()
    return plt.colorbar(im, ax=ax, shrink=0.8, ticks=sorted(ticks), format="%.2f", label=label)


def draw_arm(ax, arm: PlanarArm, q: Tensor, color: str, label: str | None = None):
    p = arm.joint_positions(q).detach().cpu().numpy()
    ax.plot(
        p[:, 0], p[:, 1], "-", color=color, lw=5, solid_capstyle="round", label=label, zorder=3
    )
    ax.plot(p[:-1, 0], p[:-1, 1], "o", color="k", ms=4, zorder=4)  # base and joints
    if label is not None:
        ax.annotate(
            label, p[-1], xytext=(4, 4), textcoords="offset points", fontsize=9, zorder=5
        )


def plot_workspace(
    ax,
    scene: Scene,
    arm: PlanarArm | None = None,
    qs=None,
    colors=None,
    labels=None,
    show_sdf: bool = False,
    resolution: int = 300
):
    """
    Obstacles (sdf < 0).
    Optionally the workspace SDF as a heatmap, the arm's reach and some poses.
    """
    X, Y = scene.workspace_grid(resolution)
    D = scene.obstacle.sdf(torch.stack([X, Y], dim=-1))
    X, Y, D = X.numpy(), Y.numpy(), D.numpy()
    if show_sdf:
        im = ax.pcolormesh(
            X,
            Y,
            D,
            cmap=scene.field_color_map,
            norm=TwoSlopeNorm(0.0, D.min(), D.max()),
            shading="nearest",
        )
        ax.contour(
            X,
            Y,
            D,
            levels=np.arange(0.5, D.max(), 0.5),
            colors="k",
            linewidths=0.4,
            alpha=0.35,
        )
        add_colorbar(ax, im, D.min(), D.max(), label="workspace SDF")
    else:
        ax.contourf(X, Y, D, levels=[D.min() - 1, 0.0], colors=[scene.obstacle_color])
    ax.contour(X, Y, D, levels=[0.0], colors="k", linewidths=1)
    if arm is not None:
        ax.add_patch(
            plt.Circle(
                (arm.base_position_x, arm.base_position_y),
                arm.reach + arm.link_radius,
                fill=False,
                ls="--",
                color="0.5",
                lw=1,
            )
        )
        for k, q in enumerate(qs if qs is not None else []):
            draw_arm(
                ax,
                arm,
                q,
                colors[k] if colors else "orange",
                labels[k] if labels else None,
            )
    ax.set(
        xlim=(scene.bounds.xmin, scene.bounds.xmax),
        ylim=(scene.bounds.ymin, scene.bounds.ymax),
        aspect="equal",
        xlabel="x",
        ylabel="y",
    )
    return ax
