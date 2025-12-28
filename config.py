from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    alpha: float = 2.0
    R_endurance: float = 20.0
    ring_resolution: int = 10
    shrink_factor: float = 0.9  # Initial candidate ring radius as fraction of max feasible
    shrink_freedom: float = 1.0  # SOCP freedom region as fraction of distance to nearest obstacle
    obs_margin: float = 0.1  # distance subtracted from freedom radii to prevent strange behavior
    max_iters: int = 50
    socp_stop_threshold: float = 0.0001  # Stop SOCP when improvement is below this

    # Ship weight multipliers for objective function
    obj_weights_ship: List[float] = field(default_factory=lambda: [1.1, 2.0, 5.0])

    num_targets_list: List[int] = field(default_factory=lambda: [10])
    num_obstacles_list: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    allow_overlap: bool = False  # controls whether obstacles are allowed to overlap
    seeds: List[int] = field(default_factory=lambda: list(range(25)))

    # Custom instances: if non-empty, run these instead of random grid
    custom_instances: List[str] = field(default_factory=list)
    # custom_instances: List[str] = field(default_factory=lambda: ["pentagon_pyramid"])
    custom_instances_dir: str = "custom_instances"

    orig: tuple = (-10.0, -10.0)
    dest: tuple = (-10.0, -10.0)

    plots_dir: str = "R20_Alpha2/plots"
    instance_info_dir: str = "R20_Alpha2/instance_info"
    route_info_dir: str = "R20_Alpha2/route_info"
    results_dir: str = "R20_Alpha2"

    make_plots: bool = True
    debug: bool = True