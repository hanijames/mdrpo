import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from geometry import Point, point_in_poly


@dataclass
class Obstacle:
    verts: List[Point]


@dataclass
class Instance:
    orig: Point
    dest: Point
    alpha: float
    R: float
    obstacles: List[Obstacle]
    targets: List[Point]
    name: str = ""


def union_contains(p: Point, obstacles: List[Obstacle]) -> bool:
    return any(point_in_poly(p, o.verts) for o in obstacles)


def generate_obstacles(num_obst: int, seed: int) -> List[Obstacle]:
    rng = random.Random(seed)
    obstacles: List[Obstacle] = []
    for _ in range(num_obst):
        k = rng.randint(3, 8)
        cx, cy = rng.uniform(0, 100), rng.uniform(0, 100)
        R = rng.uniform(3, 5)
        theta0 = rng.uniform(0, 2 * math.pi)
        verts = []
        for i in range(k):
            ang = theta0 + 2 * math.pi * i / k
            verts.append((cx + R * math.cos(ang), cy + R * math.sin(ang)))
        obstacles.append(Obstacle(verts))
    return obstacles


def generate_targets_on_land(num_targets: int, obstacles: List[Obstacle], seed: int) -> List[Point]:
    # Targets must be on obstacles (land) because mothership cannot reach them directly
    rng = random.Random(seed)
    targets: List[Point] = []
    while len(targets) < num_targets:
        x = rng.uniform(0.0, 100.0)
        y = rng.uniform(0.0, 100.0)
        if union_contains((x, y), obstacles):
            targets.append((x, y))
    return targets


def auto_instance_name(n_obstacles: int, n_targets: int, seed: int, R: float) -> str:
    return f"T{n_targets}_O{n_obstacles}_S{seed}"


def make_instance(
    num_obst: int,
    num_targets: int,
    seed: int,
    alpha: float = 2.0,
    R: float = 20.0,
    orig: Point = (-10.0, -10.0),
    dest: Point = (-10.0, -10.0)
) -> Instance:
    obstacles = generate_obstacles(num_obst, seed)
    targets = generate_targets_on_land(num_targets, obstacles, seed + 2)
    name = auto_instance_name(n_obstacles=num_obst, n_targets=num_targets, seed=seed, R=R)
    return Instance(orig=orig, dest=dest, alpha=alpha, R=R, obstacles=obstacles, targets=targets, name=name)


def save_instance_json(instance: Instance, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {"depot": list(instance.orig)}
    for i, obs in enumerate(instance.obstacles, 1):
        payload[f"obstacle_{i}"] = [list(v) for v in obs.verts]
    for i, target in enumerate(instance.targets, 1):
        payload[f"target_{i}"] = list(target)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
