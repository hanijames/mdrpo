import math
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import cvxpy as cp

from geometry import Point, dist
from instance import Instance
from visibility import freedom_radius, first_last_turn, wet_distance, wet_route_cost_via_Dvv
import gtsp_lk


def solve_tsp_on_launches(
    instance: Instance,
    Ls: List[Point],
    base_verts,
    Dvv: np.ndarray,
    seed: int = 0
) -> List[int]:
    """
    Solve TSP on current launch locations to find best visiting order.
    Returns a permutation of [0, 1, ..., n-1] indicating new order of positions.
    """
    n = len(Ls)
    if n <= 1:
        return list(range(n))

    depot = instance.orig
    nodes = [depot] + list(Ls)
    N = len(nodes)

    D = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            D[i][j] = wet_route_cost_via_Dvv(nodes[i], nodes[j], base_verts, instance.obstacles, Dvv)

    clusters = [[i] for i in range(N)]

    res = gtsp_lk.solve_gtsp_order2opt(
        D, clusters, seed=seed, time_limit_s=30.0, restarts=8, verbose=False
    )

    tour = res.best_order
    if 0 in tour:
        k = tour.index(0)
        tour = tour[k:] + tour[:k]

    new_pos_sequence = [c - 1 for c in tour[1:]]
    return new_pos_sequence


def _run_socp_iterations(
    instance: Instance,
    order: List[int],
    launch_cur: List[List[float]],
    return_cur: List[List[float]],
    num_iters: int,
    iter_offset: int,
    shrink_freedom: float,
    base_verts,
    Dvv: np.ndarray,
    best_val: float,
    best_iter: int,
    on_iteration: Optional[Callable],
    recs: List[Dict],
    return_points: bool
) -> Tuple[float, int, bool]:
    """
    Run a fixed number of SOCP iterations. Returns (best_val, best_iter, success).
    Modifies launch_cur, return_cur, recs in place.
    """
    target = [instance.targets[i] for i in order]
    n = len(target)

    for local_it in range(1, num_iters + 1):
        it = iter_offset + local_it

        launch_prev = [(float(launch_cur[i][0]), float(launch_cur[i][1])) for i in range(n)]
        return_prev = [(float(return_cur[i][0]), float(return_cur[i][1])) for i in range(n)]

        launch_freedom = [freedom_radius(launch_prev[i], instance.obstacles, shrink_freedom) for i in range(n)]
        return_freedom = [freedom_radius(return_prev[i], instance.obstacles, shrink_freedom) for i in range(n)]

        launch_var = [cp.Variable(2) for _ in range(n)]
        return_var = [cp.Variable(2) for _ in range(n)]
        constraints = []
        objective = cp.Constant(0)

        first_obst, last_obst, wrd = first_last_turn(instance.orig, launch_prev[0], instance.obstacles, base_verts, Dvv)
        if first_obst is None:
            objective += cp.norm(cp.Constant(instance.orig) - launch_var[0])
        else:
            objective += cp.norm(cp.Constant(instance.orig) - cp.Constant(first_obst))
            objective += cp.Constant(wrd)
            objective += cp.norm(cp.Constant(last_obst) - launch_var[0])

        first_obst, last_obst, wrd = first_last_turn(return_prev[-1], instance.dest, instance.obstacles, base_verts, Dvv)
        if first_obst is None:
            objective += cp.norm(cp.Constant(instance.dest) - return_var[-1])
        else:
            objective += cp.norm(return_var[-1] - cp.Constant(first_obst))
            objective += cp.Constant(wrd)
            objective += cp.norm(cp.Constant(last_obst) - cp.Constant(instance.dest))

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
            objective += cp.maximum(drone_time, ship_time)
            if i + 1 < n:
                first_obst_c, last_obst_c, wrd_c = first_last_turn(return_prev[i], launch_prev[i + 1], instance.obstacles, base_verts, Dvv)
                if first_obst_c is None:
                    objective += cp.norm(return_var[i] - launch_var[i + 1])
                else:
                    objective += cp.norm(return_var[i] - cp.Constant(first_obst_c))
                    objective += cp.norm(launch_var[i + 1] - cp.Constant(last_obst_c))
                    objective += cp.Constant(wrd_c)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return best_val, best_iter, False

        obj_value = objective.value
        Ls_now = [(float(launch_var[i].value[0]), float(launch_var[i].value[1])) for i in range(n)]
        Rs_now = [(float(return_var[i].value[0]), float(return_var[i].value[1])) for i in range(n)]

        for i in range(n):
            launch_cur[i] = [Ls_now[i][0], Ls_now[i][1]]
            return_cur[i] = [Rs_now[i][0], Rs_now[i][1]]

        if on_iteration is not None:
            on_iteration(it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev, order)

        rec = {"iter": it, "obj": obj_value, "order": order[:]}
        if return_points:
            rec["Ls"] = Ls_now
            rec["Rs"] = Rs_now
            rec["launch_freedom"] = launch_freedom
            rec["return_freedom"] = return_freedom
        recs.append(rec)

        if math.isfinite(obj_value) and obj_value < best_val:
            best_val = obj_value
            best_iter = it

    return best_val, best_iter, True


def refine_by_socp(
    instance: Instance,
    order: List[int],
    L0_list: List[Point],
    R0_list: List[Point],
    max_iters: int = 25,
    shrink_freedom: float = 1.0,
    base_verts=None,
    base_adj=None,
    Dvv: Optional[np.ndarray] = None,
    init_obj: Optional[float] = None,
    return_points: bool = False,
    on_iteration=None,
    reorder_iters: int = 10,
    max_reorder_attempts: int = 2
) -> Tuple[List[Dict], float, int, int]:
    """
    Refine launch/return points via successive SOCP iterations.
    
    After reorder_iters iterations, solve TSP on launch locations. If the order
    changes, restart SOCP with the new order (keeping current launch/land points).
    Allow up to max_reorder_attempts restarts; on the final attempt, run full
    max_iters with no further TSP checks.
    
    Returns: (recs, best_obj, best_iter, num_attempts)
        num_attempts is 1, 2, or 3 indicating how many times we started SOCP
    """
    n = len(order)
    if n == 0:
        return [], init_obj if init_obj else 0.0, 0, 1

    current_order = list(order)
    launch_cur = [list(L0_list[i]) for i in range(n)]
    return_cur = [list(R0_list[i]) for i in range(n)]
    
    recs = []
    best_val = float("inf")
    if init_obj is not None and init_obj < float("inf"):
        best_val = init_obj
    best_iter = 0
    
    total_iter_offset = 0
    final_attempt = 1
    
    for attempt in range(max_reorder_attempts + 1):
        final_attempt = attempt + 1  # 1-indexed
        is_final_attempt = (attempt == max_reorder_attempts)
        
        if is_final_attempt:
            # Final attempt: run full max_iters, no TSP check
            best_val, best_iter, success = _run_socp_iterations(
                instance, current_order, launch_cur, return_cur,
                num_iters=max_iters,
                iter_offset=total_iter_offset,
                shrink_freedom=shrink_freedom,
                base_verts=base_verts,
                Dvv=Dvv,
                best_val=best_val,
                best_iter=best_iter,
                on_iteration=on_iteration,
                recs=recs,
                return_points=return_points
            )
            break
        else:
            # Run reorder_iters iterations
            best_val, best_iter, success = _run_socp_iterations(
                instance, current_order, launch_cur, return_cur,
                num_iters=reorder_iters,
                iter_offset=total_iter_offset,
                shrink_freedom=shrink_freedom,
                base_verts=base_verts,
                Dvv=Dvv,
                best_val=best_val,
                best_iter=best_iter,
                on_iteration=on_iteration,
                recs=recs,
                return_points=return_points
            )
            
            if not success:
                break
            
            # Solve TSP on current launch locations
            Ls_current = [(float(launch_cur[i][0]), float(launch_cur[i][1])) for i in range(n)]
            new_pos_sequence = solve_tsp_on_launches(
                instance, Ls_current, base_verts, Dvv, seed=attempt
            )
            
            # Check if order changed
            new_order = [current_order[j] for j in new_pos_sequence]
            
            if new_order == current_order:
                # Order stable, finish remaining iterations
                remaining = max_iters - reorder_iters
                if remaining > 0:
                    total_iter_offset += reorder_iters
                    best_val, best_iter, success = _run_socp_iterations(
                        instance, current_order, launch_cur, return_cur,
                        num_iters=remaining,
                        iter_offset=total_iter_offset,
                        shrink_freedom=shrink_freedom,
                        base_verts=base_verts,
                        Dvv=Dvv,
                        best_val=best_val,
                        best_iter=best_iter,
                        on_iteration=on_iteration,
                        recs=recs,
                        return_points=return_points
                    )
                break
            else:
                # Order changed, reorder and restart
                new_launch = [launch_cur[j] for j in new_pos_sequence]
                new_return = [return_cur[j] for j in new_pos_sequence]
                
                current_order = new_order
                launch_cur = [[p[0], p[1]] for p in new_launch]
                return_cur = [[p[0], p[1]] for p in new_return]
                
                # Reset iteration counter for fresh start
                total_iter_offset = 0
                recs.clear()

    return recs, best_val, best_iter, final_attempt


def ship_true_length(instance: Instance, order: List[int], Ls: List[Point], Rs: List[Point], base_verts, base_adj) -> float:
    prev = instance.orig
    s = 0.0
    for k in range(len(order)):
        s += wet_distance(prev, Ls[k], base_verts, base_adj, instance.obstacles)
        s += wet_distance(Ls[k], Rs[k], base_verts, base_adj, instance.obstacles)
        prev = Rs[k]
    s += wet_distance(prev, instance.dest, base_verts, base_adj, instance.obstacles)
    return s
