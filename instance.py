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


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Compute minimum distance from point p to line segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p

    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def polygon_min_distance(verts1: List[Point], verts2: List[Point]) -> float:
    """Compute minimum distance between two convex polygons."""
    min_dist = float('inf')

    # Check all vertices of poly1 against all edges of poly2
    n2 = len(verts2)
    for v in verts1:
        for j in range(n2):
            d = point_to_segment_distance(v, verts2[j], verts2[(j + 1) % n2])
            min_dist = min(min_dist, d)

    # Check all vertices of poly2 against all edges of poly1
    n1 = len(verts1)
    for v in verts2:
        for i in range(n1):
            d = point_to_segment_distance(v, verts1[i], verts1[(i + 1) % n1])
            min_dist = min(min_dist, d)

    return min_dist


OBSTACLE_BUFFER = 0.5


def generate_obstacles(num_obst: int, seed: int) -> List[Obstacle]:
    rng = random.Random(seed)
    obstacles: List[Obstacle] = []
    max_attempts = 1000
    attempts = 0
    while len(obstacles) < num_obst and attempts < max_attempts:
        attempts += 1
        k = rng.randint(3, 8)
        cx, cy = rng.uniform(0, 100), rng.uniform(0, 100)
        R = rng.uniform(3, 5)
        theta0 = rng.uniform(0, 2 * math.pi)
        verts = []
        for i in range(k):
            ang = theta0 + 2 * math.pi * i / k
            verts.append((cx + R * math.cos(ang), cy + R * math.sin(ang)))

        too_close = any(
            polygon_min_distance(verts, obs.verts) < OBSTACLE_BUFFER
            for obs in obstacles
        )
        if not too_close:
            obstacles.append(Obstacle(verts))

    if len(obstacles) < num_obst:
        print(f"Warning: only generated {len(obstacles)} non-overlapping obstacles (requested {num_obst})")

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


def load_instance_json(path: str, alpha: float = 2.0, R: float = 20.0,
                       orig: Point = (-10.0, -10.0), dest: Point = (-10.0, -10.0)) -> Instance:
    with open(path, "r") as f:
        data = json.load(f)

    depot = tuple(data.get("depot", orig))

    obstacles = []
    i = 1
    while f"obstacle_{i}" in data:
        verts = [tuple(v) for v in data[f"obstacle_{i}"]]
        obstacles.append(Obstacle(verts))
        i += 1

    targets = []
    i = 1
    while f"target_{i}" in data:
        targets.append(tuple(data[f"target_{i}"]))
        i += 1

    name = os.path.splitext(os.path.basename(path))[0]

    return Instance(orig=depot, dest=dest, alpha=alpha, R=R,
                    obstacles=obstacles, targets=targets, name=name)