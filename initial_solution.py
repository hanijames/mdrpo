import math
from typing import List, Dict, Tuple, Optional

import numpy as np

import gtsp_lk
from geometry import Point
from instance import Instance, Obstacle, union_contains
from visibility import build_los_graph, apsp_vertices_with_next, wet_route_cost_via_Dvv


def ring_radius(alpha: float, R: float, shrink_factor: float = 0.9) -> float:
    return shrink_factor * 0.5 * alpha * R


def ring_candidates_around_target(
    t: Point, obstacles: List[Obstacle], alpha: float, R: float,
    resolution: int, shrink_factor: float = 0.9
) -> List[Point]:
    r = ring_radius(alpha, R, shrink_factor)
    pts: List[Point] = []
    for k in range(resolution):
        ang = 2 * math.pi * k / resolution
        p = (t[0] + r * math.cos(ang), t[1] + r * math.sin(ang))
        if not union_contains(p, obstacles):
            pts.append(p)
    return pts


def build_candidate_map(instance: Instance, resolution: int = 10, shrink_factor: float = 0.9) -> Dict[int, List[Point]]:
    candidate_map: Dict[int, List[Point]] = {}
    for i, t in enumerate(instance.targets):
        C = ring_candidates_around_target(t, instance.obstacles, instance.alpha, instance.R, resolution=resolution, shrink_factor=shrink_factor)
        if not C:
            C = ring_candidates_around_target(t, instance.obstacles, instance.alpha, instance.R, resolution=360, shrink_factor=shrink_factor)
        candidate_map[i] = C
    return candidate_map


def solve_gtsp_lkh(
    instance: Instance, candidate_map: Dict[int, List[Point]], base_verts, Dvv: np.ndarray,
    time_limit: Optional[float] = None, seed: int = 0, method: str = "lk-exact",
    accept_rule: int = 4, restarts: int = 8
) -> Tuple[List[int], List[Point], List[Point], float]:
    depot = instance.orig
    nodes: List[Point] = [depot]
    node_to_target: List[int] = [-1]
    cluster_indices: Dict[int, List[int]] = {}
    for t_idx, pts in candidate_map.items():
        idxs = []
        for p in pts:
            idx = len(nodes)
            nodes.append(p)
            node_to_target.append(t_idx)
            idxs.append(idx)
        cluster_indices[t_idx] = idxs
    N = len(nodes)
    if N == 1:
        return [], [], [], 0.0
    C = [[math.inf] * N for _ in range(N)]
    for i in range(N):
        p = nodes[i]
        for j in range(N):
            if i == j:
                continue
            q = nodes[j]
            C[i][j] = wet_route_cost_via_Dvv(p, q, base_verts, instance.obstacles, Dvv)
    target_ids = list(cluster_indices.keys())
    solver_clusters: List[List[int]] = [[0]]
    solver_to_target: List[int] = [-1]
    for t in target_ids:
        solver_clusters.append(cluster_indices[t])
        solver_to_target.append(t)
    if method == "lk-exact":
        res = gtsp_lk.solve_gtsp_lk_exact(
            C, solver_clusters, seed=seed, time_limit_s=time_limit,
            restarts=max(1, restarts // 2), accept_rule=accept_rule, verbose=False
        )
    else:
        res = gtsp_lk.solve_gtsp_order2opt(
            C, solver_clusters, seed=seed, time_limit_s=time_limit, restarts=restarts, verbose=False
        )
    best_order = res.best_order
    best_reps = res.best_reps
    if 0 in best_order:
        k = best_order.index(0)
        best_order = best_order[k:] + best_order[:k]
        best_reps = best_reps[k:] + best_reps[:k]
    order_solver_clusters = best_order[1:]
    reps_nodes = best_reps[1:]
    order_targets: List[int] = [solver_to_target[cid] for cid in order_solver_clusters]
    chosen_points: List[Point] = [nodes[u] for u in reps_nodes]
    return order_targets, chosen_points[:], chosen_points[:], float(res.best_cost)


def initial_solution(
    instance: Instance, ring_resolution: int = 10, shrink_factor: float = 0.9,
    time_limit: Optional[float] = 12000
) -> Dict:
    base_verts, base_adj = build_los_graph(instance.obstacles)
    Dvv, Next = apsp_vertices_with_next(base_adj)
    candidate_map = build_candidate_map(instance, resolution=ring_resolution, shrink_factor=shrink_factor)
    order, chosen_return, chosen_launch, ship_cost = solve_gtsp_lkh(
        instance, candidate_map, base_verts, Dvv, time_limit=time_limit
    )
    wait = 0.0
    for k, idx in enumerate(order):
        launch = chosen_launch[k]
        target = instance.targets[idx]
        wait += (2.0 * math.hypot(launch[0] - target[0], launch[1] - target[1]) / instance.alpha)
    init_obj = ship_cost + wait
    return {
        "Init Obj": init_obj, "Ship Cost": ship_cost,
        "order": order, "chosen_return": chosen_return, "chosen_launch": chosen_launch,
        "candidate_map": candidate_map, "base_verts": base_verts, "base_adj": base_adj, "Dvv": Dvv, "Next": Next
    }
