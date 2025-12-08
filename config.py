from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    alpha: float = 2.0
    R_endurance: float = 20.0
    ring_resolution: int = 10
    shrink_factor: float = 0.9  # Initial candidate ring radius as fraction of max feasible
    shrink_freedom: float = 1.0  # SOCP freedom region as fraction of distance to nearest obstacle
    max_iters: int = 25

    num_targets_list: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    num_obstacles_list: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    seeds: List[int] = field(default_factory=lambda: list(range(25)))

    orig: tuple = (-10.0, -10.0)
    dest: tuple = (-10.0, -10.0)

    plots_dir: str = "R20_Alpha2/plots"
    instance_info_dir: str = "R20_Alpha2/instance_info"
    route_info_dir: str = "R20_Alpha2/route_info"
    results_dir: str = "R20_Alpha2"

    make_plots: bool = True
    debug: bool = True
