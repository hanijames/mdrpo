import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import cvxpy as cp

from geometry import Point, dist
from instance import Instance
from visibility import freedom_radius, first_last_turn, wet_distance


def refine_by_socp(
    instance: Instance,
    order: List[int],
    L0_list: List[Point],
    R0_list: List[Point],
    max_iters: int = 50,
    shrink_freedom: float = 1.0,
    base_verts=None,
    base_adj=None,
    Dvv: Optional[np.ndarray] = None,
    init_obj: Optional[float] = None,
    return_points: bool = False,
    on_iteration=None,
    stop_threshold: float = 0.0001,
    ship_weight: float = 1.0
) -> Tuple[List[Dict], float, int]:
    target = [instance.targets[i] for i in order]
    n = len(target)
    launch_cur = [list(L0_list[i]) for i in range(n)]
    return_cur = [list(R0_list[i]) for i in range(n)]
    recs = []
    best_val = float("inf")
    if init_obj is not None and init_obj < float("inf"):
        best_val = init_obj
    best_iter = 0
    prev_obj = best_val

    for it in range(1, max_iters + 1):
        launch_prev = [(float(launch_cur[i][0]), float(launch_cur[i][1])) for i in range(n)]
        return_prev = [(float(return_cur[i][0]), float(return_cur[i][1])) for i in range(n)]
        # Compute freedom radius: maximum distance to move without hitting obstacles
        launch_freedom = [freedom_radius(launch_prev[i], instance.obstacles, shrink_freedom) for i in range(n)]
        return_freedom = [freedom_radius(return_prev[i], instance.obstacles, shrink_freedom) for i in range(n)]
        launch_var = [cp.Variable(2) for _ in range(n)]
        return_var = [cp.Variable(2) for _ in range(n)]
        constraints = []
        objective = cp.Constant(0)

        # Build objective: ship path segments + max(drone_time, ship_time) per target
        # Ship segments between waypoints use wet distance (path around obstacles)
        first_obst, last_obst, wrd = first_last_turn(instance.orig, launch_prev[0], instance.obstacles, base_verts, Dvv)
        if first_obst is None:
            objective += ship_weight * cp.norm(cp.Constant(instance.orig) - launch_var[0])
        else:
            objective += ship_weight * cp.norm(cp.Constant(instance.orig) - cp.Constant(first_obst))
            objective += ship_weight * cp.Constant(wrd)
            objective += ship_weight * cp.norm(cp.Constant(last_obst) - launch_var[0])

        first_obst, last_obst, wrd = first_last_turn(return_prev[-1], instance.dest, instance.obstacles, base_verts, Dvv)
        if first_obst is None:
            objective += ship_weight * cp.norm(cp.Constant(instance.dest) - return_var[-1])
        else:
            objective += ship_weight * cp.norm(return_var[-1] - cp.Constant(first_obst))
            objective += ship_weight * cp.Constant(wrd)
            objective += ship_weight * cp.norm(cp.Constant(last_obst) - cp.Constant(instance.dest))

        for i in range(n):
            target_i = cp.Constant(target[i])
            Li_prev = cp.Constant([launch_prev[i][0], launch_prev[i][1]])
            constraints += [cp.norm(launch_var[i] - Li_prev) <= launch_freedom[i]]
            Ri_prev = cp.Constant([return_prev[i][0], return_prev[i][1]])
            constraints += [cp.norm(return_var[i] - Ri_prev) <= return_freedom[i]]
            constraints += [cp.norm(launch_var[i] - target_i) + cp.norm(return_var[i] - target_i) <= cp.Constant(instance.alpha * instance.R)]
            drone_time = (cp.norm(launch_var[i] - target_i) + cp.norm(return_var[i] - target_i)) / instance.alpha
            first_obst_s, last_obst_s, wrd_s = first_last_turn(launch_prev[i], return_prev[i], instance.obstacles, base_verts, Dvv)
            if first_obst_s is None:
                constraints += [cp.norm(launch_var[i] - return_var[i]) <= cp.Constant(instance.R)]
                ship_time = cp.norm(launch_var[i] - return_var[i])
            else:
                constraints += [cp.norm(launch_var[i] - cp.Constant(first_obst_s)) + cp.Constant(wrd_s) + cp.norm(return_var[i] - cp.Constant(last_obst_s)) <= cp.Constant(instance.R)]
                ship_time = cp.norm(launch_var[i] - cp.Constant(first_obst_s)) + cp.Constant(wrd_s) + cp.norm(return_var[i] - cp.Constant(last_obst_s))
            objective += cp.maximum(drone_time, ship_weight * ship_time)
            if i + 1 < n:
                first_obst_c, last_obst_c, wrd_c = first_last_turn(return_prev[i], launch_prev[i + 1], instance.obstacles, base_verts, Dvv)
                if first_obst_c is None:
                    objective += ship_weight * cp.norm(return_var[i] - launch_var[i + 1])
                else:
                    objective += ship_weight * cp.norm(return_var[i] - cp.Constant(first_obst_c))
                    objective += ship_weight * cp.norm(launch_var[i + 1] - cp.Constant(last_obst_c))
                    objective += ship_weight * cp.Constant(wrd_c)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return recs, best_val, best_iter

        obj_value = objective.value
        Ls_now = [(float(launch_var[i].value[0]), float(launch_var[i].value[1])) for i in range(n)]
        Rs_now = [(float(return_var[i].value[0]), float(return_var[i].value[1])) for i in range(n)]

        for i in range(n):
            launch_cur[i] = [Ls_now[i][0], Ls_now[i][1]]
            return_cur[i] = [Rs_now[i][0], Rs_now[i][1]]

        if on_iteration is not None:
            on_iteration(it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev)

        rec = {"iter": it, "obj": obj_value}
        if return_points:
            rec["Ls"] = Ls_now
            rec["Rs"] = Rs_now
            rec["launch_freedom"] = launch_freedom
            rec["return_freedom"] = return_freedom
        recs.append(rec)

        if math.isfinite(obj_value):
            if obj_value < best_val:
                best_val = obj_value
                best_iter = it
            
            improvement = prev_obj - obj_value
            if improvement < stop_threshold:
                break
            prev_obj = obj_value

    return recs, best_val, best_iter


def ship_true_length(instance: Instance, order: List[int], Ls: List[Point], Rs: List[Point], base_verts, base_adj) -> float:
    prev = instance.orig
    s = 0.0
    for k in range(len(order)):
        s += wet_distance(prev, Ls[k], base_verts, base_adj, instance.obstacles)
        s += wet_distance(Ls[k], Rs[k], base_verts, base_adj, instance.obstacles)
        prev = Rs[k]
    s += wet_distance(prev, instance.dest, base_verts, base_adj, instance.obstacles)
    return s


def compute_total_time(instance: Instance, order: List[int], Ls: List[Point], Rs: List[Point],
                       base_verts, base_adj) -> float:
    """Compute the original total time objective (ship_weight=1)."""
    total = 0.0
    
    total += wet_distance(instance.orig, Ls[0], base_verts, base_adj, instance.obstacles)
    
    for i, idx in enumerate(order):
        target = instance.targets[idx]
        drone_time = (dist(Ls[i], target) + dist(Rs[i], target)) / instance.alpha
        ship_time = wet_distance(Ls[i], Rs[i], base_verts, base_adj, instance.obstacles)
        total += max(drone_time, ship_time)
        
        if i + 1 < len(order):
            total += wet_distance(Rs[i], Ls[i + 1], base_verts, base_adj, instance.obstacles)
    
    total += wet_distance(Rs[-1], instance.dest, base_verts, base_adj, instance.obstacles)
    
    return total
