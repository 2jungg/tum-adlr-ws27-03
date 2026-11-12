import torch
import numpy as np
from metacosdf.scene_generator.arm import PlanarArm
from metacosdf.scene_generator.scene import Scene
from metacosdf.scene_generator.obstacles import Obstacle, Circle, Box

import math

def create_random_obs(
    arm: PlanarArm,
    min_base_clearance: float = 0.2,
    rng: np.random.Generator | None = None,
) -> Obstacle:
    rng = rng or np.random.default_rng()
    theta = rng.uniform(0, 2 * math.pi)
    r_min = min_base_clearance + arm.link_radius
    r_max = arm.reach
    r = np.sqrt(rng.uniform(r_min**2, r_max**2))    # uniform randomness regarding polar coordinate system
    cx = arm.base[0] + math.cos(theta) * r
    cy = arm.base[1] + math.sin(theta) * r

    if rng.uniform(0.0, 1.0) > 0.5:   # Circle obstacle
        radius = float(rng.uniform(0.1, 0.5))
        return Circle((float(cx), float(cy)), radius)
    else:
        hx, hy = float(rng.uniform(0.1, 0.5)), float(rng.uniform(0.1, 0.5))
        angle = float(rng.uniform(0, math.pi))
        return Box((float(cx), float(cy)), (hx, hy), angle)

def create_random_scene(
    arm: PlanarArm,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    max_obs: int = 4,
    min_base_clearance: float = 0.2,
    rng: np.random.Generator | None = None,
) -> Scene:
    rng = rng or np.random.default_rng()
    num_obs = rng.integers(1, max_obs + 1)
    cnt = 0
    obstacles = []
    base_tensor = torch.tensor(arm.base, dtype=torch.float32)
    for _ in range(100):
        obs = create_random_obs(arm, min_base_clearance=min_base_clearance, rng=rng)
        if obs.sdf(base_tensor) > arm.link_radius + min_base_clearance:
            cnt += 1
            obstacles.append(obs)
            if cnt == num_obs:
                break

    # We can also consider to check collision label ratio (not too low and high)

    return Scene(tuple(obstacles), bounds)