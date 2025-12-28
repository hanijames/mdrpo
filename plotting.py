import os
import csv
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import Point, dist
from instance import Instance
from visibility import first_last_turn, reconstruct_vertex_path, wet_polyline
from initial_solution import ring_radius

def plot_iter(
    instance: Instance,
    order: List[int],
    Ls: List[Point],
    Rs: List[Point],
    base_verts,
    base_adj,
    Dvv: np.ndarray,
    Next: np.ndarray,
    filename: Optional[str] = None,
    show_circles: bool = False,
    candidate_map: Optional[Dict[int, List[Point]]] = None,
    title: Optional[str] = None,
    obj_value: Optional[float] = None,
    freedom_range: Optional[Tuple[List[float], List[float]]] = None,
    freedom_centers: Optional[Tuple[List[Point], List[Point]]] = None,
    max_iters: int = 25,
    route_info_dir: str = "route_info"
):
    V = np.asarray(base_verts, dtype=float)
    vert_idx = {(float(V[k, 0]), float(V[k, 1])): k for k in range(len(V))}
    fig, ax = plt.subplots(figsize=(8, 8))
    # Accumulate all waypoints for route CSV export
    ship_points = [instance.orig]

    for obs in instance.obstacles:
        xs = [v[0] for v in obs.verts] + [obs.verts[0][0]]
        ys = [v[1] for v in obs.verts] + [obs.verts[0][1]]
        ax.fill(xs, ys, color="#bbbbbb", alpha=0.8, zorder=2)
        ax.plot(xs, ys, color="#888888", linewidth=1.0, zorder=3)

    ax.scatter([instance.orig[0]], [instance.orig[1]], s=80, marker="s", color="purple", zorder=8, label="Depot")
    tx = [instance.targets[i][0] for i in range(len(instance.targets))]
    ty = [instance.targets[i][1] for i in range(len(instance.targets))]
    ax.scatter(tx, ty, s=40, marker="x", color="purple", zorder=8, label="Target")

    if show_circles:
        rad = ring_radius(instance.alpha, instance.R)
        did_label = False
        for i, t in enumerate(instance.targets):
            circ = plt.Circle(t, rad, edgecolor="orange", facecolor="none", linestyle="--", linewidth=1.5, zorder=4)
            ax.add_patch(circ)
            if candidate_map is not None and i in candidate_map and candidate_map[i]:
                X = [p[0] for p in candidate_map[i]]
                Y = [p[1] for p in candidate_map[i]]
                ax.scatter(X, Y, s=25, marker="o", facecolors="blue", edgecolors="blue", linewidths=1, alpha=0.95, zorder=9, label=None if did_label else "launch/land candidates")
                did_label = True

    if freedom_range is not None:
        rL, rR = freedom_range
        cL = Ls if freedom_centers is None else freedom_centers[0]
        cR = Rs if freedom_centers is None else freedom_centers[1]
        labeled_L = labeled_R = False
        for k in range(len(order)):
            if k < len(cL) and k < len(rL) and rL[k] > 0:
                ax.add_patch(plt.Circle(cL[k], rL[k], edgecolor="blue", facecolor="none", linestyle=(0, (1,3)),
                                        linewidth=1.5, zorder=4, label=None if labeled_L else "Launch Freedom"))
                labeled_L = True
            if k < len(cR) and k < len(rR) and rR[k] > 0:
                # ax.add_patch(plt.Circle(cR[k], rR[k], edgecolor="red", facecolor="none", linestyle=":", linewidth=1.5, zorder=4, label=None if labeled_R else "Landing Freedom"))
                ax.add_patch(plt.Circle(cR[k], rR[k], edgecolor="red", facecolor="none", linestyle=(2, (1, 3)),
                                        linewidth=1.5, zorder=4, label=None if labeled_R else "Landing Freedom"))
                labeled_R = True

    def _plot_leg(p, q, label=None):
        nonlocal ship_points
        first, last, wrd = first_last_turn(p, q, instance.obstacles, base_verts, Dvv)
        if first is None:
            poly = [p, q]
            ship_points.append(q)
        else:
            vi = vert_idx.get((float(first[0]), float(first[1])))
            vj = vert_idx.get((float(last[0]), float(last[1])))
            if vi is not None and vj is not None:
                vpath = reconstruct_vertex_path(vi, vj, Next)
                if vpath:
                    poly = [p] + [(float(V[k, 0]), float(V[k, 1])) for k in vpath] + [q]
                    for k in vpath:
                        ship_points.append((float(V[k, 0]), float(V[k, 1])))
                    ship_points.append(q)
                else:
                    poly = wet_polyline(p, q, base_verts, base_adj, instance.obstacles)
                    ship_points.extend(poly[1:])
            else:
                poly = wet_polyline(p, q, base_verts, base_adj, instance.obstacles)
                ship_points.extend(poly[1:])
        ax.plot([pt[0] for pt in poly], [pt[1] for pt in poly], "-", color="black", linewidth=2.0, zorder=5, label=label)

    if len(order) > 0:
        _plot_leg(instance.orig, Ls[0], label="Mothership")
        for k in range(len(order)):
            _plot_leg(Ls[k], Rs[k], label=None)
            if k + 1 < len(order):
                _plot_leg(Rs[k], Ls[k + 1], label=None)
        _plot_leg(Rs[-1], instance.dest, label=None)
    else:
        _plot_leg(instance.orig, instance.dest, label="Mothership")

    for k, idx in enumerate(order):
        T = instance.targets[idx]
        ax.plot([Ls[k][0], T[0]], [Ls[k][1], T[1]], "--", color="blue", linewidth=1.5, zorder=6, label="Drone Out" if k == 0 else None)
        ax.plot([T[0], Rs[k][0]], [T[1], Rs[k][1]], "--", color="red", linewidth=1.5, zorder=6, label="Drone Return" if k == 0 else None)

    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 120)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if title is not None and obj_value is not None:
        ax.set_title(f"{title} — Obj={obj_value:.3f}")
    elif title is not None:
        ax.set_title(title)
    elif obj_value is not None:
        ax.set_title(f"Objective = {obj_value:.3f}")

    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    if title is not None:
        import re
        match = re.search(r'Iter\s*(\d+)', title)
        if match:
            it = int(match.group(1))
        else:
            raise ValueError(f"Could not parse iteration number from title: {title}")
        # Create subfolder for this instance's iteration CSVs
        iter_csv_dir = os.path.join(route_info_dir, instance.name)
        _write_route_csv(instance, order, Ls, Rs, ship_points, iter_csv_dir, iteration=it, obj_value=obj_value)


def _write_route_csv(instance: Instance, order: List[int], Ls: List[Point], Rs: List[Point], ship_points: List[Point], route_info_dir: str, iteration: Optional[int] = None, obj_value: Optional[float] = None):
    os.makedirs(route_info_dir, exist_ok=True)
    if iteration is not None:
        fname = f"{instance.name}_iter_{iteration:02d}.csv"
    else:
        fname = instance.name + "_route.csv"
    p = os.path.join(route_info_dir, fname)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Mothership Route Points", "Distance from Previous", "Drone Distance Out", "Drone Distance Return", "Target Location", "Drone Trip Distance", "Ship Waiting Time"])
        w.writerow([f"({ship_points[0][0]:.3f},{ship_points[0][1]:.3f})"])
        ship_dist = 0
        ship_wait_time = 0
        ship_launch_to_land = 0
        drone_out = False
        for k in range(1, len(ship_points)):
            dist_from_last = dist(ship_points[k - 1], ship_points[k])
            ship_dist += dist_from_last
            drone_point = False
            if drone_out:
                ship_launch_to_land += dist_from_last
            for m, idx in enumerate(order):
                T = instance.targets[idx]
                launch_point = (Ls[m][0], Ls[m][1])
                return_point = (Rs[m][0], Rs[m][1])
                if ship_points[k] == launch_point:
                    drone_point = True
                    drone_out = True
                    drone_dist_out = dist(ship_points[k], T)
                    if launch_point == return_point:
                        drone_out = False
                        w.writerow([f"({ship_points[k][0]:.3f},{ship_points[k][1]:.3f})", f"{dist_from_last:.3f}", f"{drone_dist_out:.3f}", f"{drone_dist_out:.3f}", f"({T[0]:.3f},{T[1]:.3f})", f"{drone_dist_out*2:.3f}", f"{drone_dist_out:.3f}"])
                    else:
                        w.writerow([f"({ship_points[k][0]:.3f},{ship_points[k][1]:.3f})", f"{dist_from_last:.3f}", f"{drone_dist_out:.3f}", "", f"({T[0]:.3f},{T[1]:.3f})", ""])
                elif ship_points[k] == return_point:
                    drone_point = True
                    drone_out = False
                    drone_dist_out = dist(launch_point, T)
                    drone_dist_back = dist(ship_points[k], T)
                    drone_time = (drone_dist_out + drone_dist_back) / instance.alpha
                    # Ship waits only if drone takes longer than ship travel between launch and return
                    if drone_time > ship_launch_to_land:
                        wait_time = drone_time - ship_launch_to_land
                        ship_wait_time += wait_time
                        w.writerow([f"({ship_points[k][0]:.3f},{ship_points[k][1]:.3f})", f"{dist_from_last:.3f}", "", f"{drone_dist_back:.3f}", "", f"{drone_dist_out + drone_dist_back:.3f}", f"{wait_time:.3f}"])
                    else:
                        w.writerow([f"({ship_points[k][0]:.3f},{ship_points[k][1]:.3f})", f"{dist_from_last:.3f}", "", f"{drone_dist_back:.3f}", "", f"{drone_dist_out + drone_dist_back:.3f}"])
                    ship_launch_to_land = 0
            if not drone_point:
                w.writerow([f"({ship_points[k][0]:.3f},{ship_points[k][1]:.3f})", f"{dist_from_last:.3f}", "", "", "", ""])
        w.writerow([f"Total Time: ", f"{ship_dist + ship_wait_time:.3f}", "", "", "", ""])
        if obj_value is not None:
            w.writerow([f"SOCP Objective: ", f"{obj_value:.3f}", "", "", "", ""])

def append_results_csv(num_targets: int, num_obst: int, seed: int, init_obj: float, best_obj: float, results_dir: str = "."):
    fname = f"results_T{num_targets}_O{num_obst}.csv"
    p = os.path.join(results_dir, fname)
    need_header = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["seed", "initial_objective", "best_objective", "proportion_reduction"])
        prop_reduction = (init_obj - best_obj) / init_obj if init_obj > 0 else 0.0
        w.writerow([seed, f"{init_obj:.6f}", f"{best_obj:.6f}", f"{prop_reduction:.6f}"])
