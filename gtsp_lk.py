from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class GTSPResult:
    best_order: List[int]
    best_reps: List[int]
    best_cost: float
    stats: Dict[str, float]

def build_euclidean_D(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            D[i][j] = math.hypot(xi - xj, yi - yj)
    return D

def tour_cost_from_reps(D: List[List[float]], reps: List[int]) -> float:
    total = 0.0
    m = len(reps)
    for i in range(m):
        u = reps[i]
        v = reps[(i + 1) % m]
        total += D[u][v]
    return total

def cluster_optimize_fixed_order(
    D: List[List[float]],
    clusters: List[List[int]],
    order: List[int],
) -> Tuple[List[int], float]:

    m = len(order)
    if m == 0:
        return [], 0.0
    if m == 1:
        c0 = clusters[order[0]]
        best_node = min(c0, key=lambda u: D[u][u])
        return [best_node], 0.0

    C0 = clusters[order[0]]
    layers = [clusters[order[i]] for i in range(1, m)]

    best_cycle_cost = math.inf
    best_reps = None

    for s in C0:
        preds: List[Dict[int, int]] = []
        cost = {v: D[s][v] for v in layers[0]}
        preds.append({v: s for v in layers[0]})
        for i in range(1, len(layers)):
            prev_nodes = list(cost.keys())
            new_cost: Dict[int, float] = {}
            new_pred: Dict[int, int] = {}
            layer = layers[i]
            for v in layer:
                best_val = math.inf
                best_p = -1
                for p in prev_nodes:
                    cand = cost[p] + D[p][v]
                    if cand < best_val:
                        best_val = cand
                        best_p = p
                new_cost[v] = best_val
                new_pred[v] = best_p
            cost = new_cost
            preds.append(new_pred)

        # close to s
        best_last = None
        best_val = math.inf
        for v, v_cost in cost.items():
            cand = v_cost + D[v][s]
            if cand < best_val:
                best_val = cand
                best_last = v

        reps = [None]*m
        reps[0] = s
        cur = best_last
        for layer_idx in reversed(range(len(layers))):
            reps[layer_idx+1] = cur
            cur = preds[layer_idx][cur]

        if best_val < best_cycle_cost:
            best_cycle_cost = best_val
            best_reps = reps

    if best_reps is None:
        best_reps = [clusters[c][0] for c in order]
        best_cycle_cost = tour_cost_from_reps(D, best_reps)
    return best_reps, best_cycle_cost

def path_optimize_fixed_order(
    D: List[List[float]],
    clusters: List[List[int]],
    path_order: List[int],
) -> Tuple[List[int], float]:
    k = len(path_order)
    if k == 0:
        return [], 0.0
    if k == 1:
        c0 = clusters[path_order[0]]
        best_node = min(c0, key=lambda u: 0.0)
        return [best_node], 0.0

    layers = [clusters[idx] for idx in path_order]
    # init at first layer (no incoming edge cost)
    cost = {u: 0.0 for u in layers[0]}
    preds: List[Dict[int, int]] = []  # pred for layer i>0
    preds.append({})  # dummy for layer 0

    for i in range(1, k):
        prev_nodes = list(cost.keys())
        new_cost: Dict[int, float] = {}
        new_pred: Dict[int, int] = {}
        layer = layers[i]
        for v in layer:
            best_val = math.inf
            best_p = -1
            for p in prev_nodes:
                cand = cost[p] + D[p][v]
                if cand < best_val:
                    best_val = cand
                    best_p = p
            new_cost[v] = best_val
            new_pred[v] = best_p
        cost = new_cost
        preds.append(new_pred)

    # terminal
    last_v = min(cost, key=lambda v: cost[v])
    best_cost = cost[last_v]

    reps = [None]*k
    reps[-1] = last_v
    cur = last_v
    for i in range(k-1, 0, -1):
        cur = preds[i][cur]
        reps[i-1] = cur
    return reps, best_cost

def build_initial_order_minlink(
    D: List[List[float]], clusters: List[List[int]], rng: random.Random
) -> List[int]:
    m = len(clusters)
    # precompute min inter-cluster distances
    minAB = [[0.0]*m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            best = math.inf
            for u in clusters[a]:
                Du = D[u]
                for v in clusters[b]:
                    w = Du[v]
                    if w < best:
                        best = w
            minAB[a][b] = best

    unvisited = list(range(m))
    start = rng.choice(unvisited)
    order = [start]
    unvisited.remove(start)

    while unvisited:
        best_add = math.inf
        best_pos = None
        best_b = None
        k = len(order)
        for b in unvisited:
            for pos in range(k):
                a = order[pos]
                c = order[(pos+1) % k] if k >= 2 else a
                delta = (minAB[a][b] + minAB[b][c]) - minAB[a][c]
                if delta < best_add:
                    best_add = delta
                    best_pos = (pos+1) % (k if k else 1)
                    best_b = b
        order.insert(best_pos, best_b)
        unvisited.remove(best_b)
    return order

def two_opt_reverse_segment_ring(order: List[int], i: int, j: int) -> List[int]:
    m = len(order)
    assert 0 <= i < j < m
    seg = order[i+1:j+1]
    seg.reverse()
    return order[:i+1] + seg + order[j+1:]

def double_bridge_kick(order: List[int], rng: random.Random) -> List[int]:
    m = len(order)
    if m < 8:
        i = rng.randrange(1, m-2)
        j = rng.randrange(i+1, m-1)
        return order[:i] + list(reversed(order[i:j])) + order[j:]
    cuts = sorted(rng.sample(range(1, m), 4))
    a,b,c,d = cuts
    A = order[:a]
    B = order[a:b]
    C = order[b:c]
    D = order[c:d] + order[d:]
    return A + C + B + D

def solve_gtsp_order2opt(
    D: List[List[float]],
    clusters: List[List[int]],
    seed: int = 0,
    time_limit_s: Optional[float] = None,
    restarts: int = 8,
    max_passes_without_gain: int = 3,
    verbose: bool = True,
) -> GTSPResult:
    t0 = time.time()
    rng = random.Random(seed)
    m = len(clusters)

    def time_ok() -> bool:
        return (time_limit_s is None) or ((time.time() - t0) < time_limit_s)

    best_cost = math.inf
    best_order: List[int] = []
    best_reps: List[int] = []
    passes_total = 0
    kicks_total = 0
    evals_total = 0

    for rs in range(restarts):
        if not time_ok():
            break
        order = build_initial_order_minlink(D, clusters, rng)
        reps, cost = cluster_optimize_fixed_order(D, clusters, order)
        evals_total += 1
        if verbose:
            print(f"[order-2opt restart {rs+1}/{restarts}] init cost = {cost:.6f}")

        no_gain_passes = 0
        improved = True
        while improved and time_ok():
            improved = False
            passes_total += 1
            rot = rng.randrange(m)
            order = order[rot:] + order[:rot]
            reps, cost = cluster_optimize_fixed_order(D, clusters, order)
            evals_total += 1

            found_this_pass = False
            for i in range(0, m-2):
                if found_this_pass or not time_ok():
                    break
                for j in range(i+2, m):
                    if (i == 0 and j == m-1):
                        continue
                    new_order = two_opt_reverse_segment_ring(order, i, j)
                    new_reps, new_cost = cluster_optimize_fixed_order(D, clusters, new_order)
                    evals_total += 1
                    if new_cost + 1e-12 < cost:
                        order, reps, cost = new_order, new_reps, new_cost
                        found_this_pass = True
                        improved = True
                        if verbose:
                            print(f"  2-opt improved: {cost:.6f}")
                        break
            if found_this_pass:
                no_gain_passes = 0
                continue

            no_gain_passes += 1
            if no_gain_passes >= max_passes_without_gain and time_ok():
                order = double_bridge_kick(order, rng)
                reps, cost = cluster_optimize_fixed_order(D, clusters, order)
                kicks_total += 1
                evals_total += 1
                no_gain_passes = 0
                if verbose:
                    print(f"  kick -> cost {cost:.6f}")
                improved = True

        if cost < best_cost:
            best_cost = cost
            best_order = order[:]
            best_reps = reps[:]
            if verbose:
                print(f"  >> new global best: {best_cost:.6f}")

    elapsed = time.time() - t0
    return GTSPResult(
        best_order=best_order,
        best_reps=best_reps,
        best_cost=best_cost,
        stats={
            "elapsed_sec": elapsed,
            "passes": passes_total,
            "kicks": kicks_total,
            "evals": evals_total,
            "restarts": restarts,
            "clusters": len(clusters),
            "nodes": sum(len(c) for c in clusters),
            "method": "order-2opt",
        },
    )

def solve_gtsp_lk_exact(
    D: List[List[float]],
    clusters: List[List[int]],
    seed: int = 0,
    time_limit_s: Optional[float] = None,
    restarts: int = 4,
    accept_rule: int = 4,  # 4 or 5 (per K&G)
    verbose: bool = True,
) -> GTSPResult:
    t0 = time.time()
    rng = random.Random(seed)
    m = len(clusters)

    def time_ok() -> bool:
        return (time_limit_s is None) or ((time.time() - t0) < time_limit_s)

    # helper to compute acceptance threshold (rule 4 or 5)
    def accept(new_path_cost: float, T_cost: float, m: int) -> bool:
        if accept_rule == 4:
            # optimistic: path must be < current tour
            return new_path_cost + 1e-12 < T_cost
        elif accept_rule == 5:
            # assume a half-average edge to close
            return new_path_cost + (T_cost / (2.0*m)) + 1e-12 < T_cost
        else:
            return new_path_cost + 1e-12 < T_cost

    best_cost = math.inf
    best_order: List[int] = []
    best_reps: List[int] = []
    evals_total = 0
    passes_total = 0
    improvements = 0

    for rs in range(restarts):
        if not time_ok():
            break
        order = build_initial_order_minlink(D, clusters, rng)
        reps, T_cost = cluster_optimize_fixed_order(D, clusters, order)
        evals_total += 1
        if verbose:
            print(f"[lk-exact restart {rs+1}/{restarts}] init cost = {T_cost:.6f}")

        improved_any = True
        while improved_any and time_ok():
            improved_any = False
            passes_total += 1
            # Cycle through edges (e->b) by rotating starting point b
            for rot in range(m):
                if not time_ok():
                    break
                b_idx = rot % m
                e_idx = (rot - 1) % m
                if b_idx <= e_idx:
                    P = order[b_idx:e_idx+1]
                else:
                    P = order[b_idx:] + order[:e_idx+1]
                # precompute w_co for current path
                _, wco_P = path_optimize_fixed_order(D, clusters, P)
                evals_total += 1
                # Try all internal edges x->y on path with x != b (skip i=0)
                found_local = False
                for i in range(1, len(P)-1):
                    if not time_ok() or found_local:
                        break
                    # break edge (x,y) at i,i+1 and reverse suffix
                    newP = P[:i+1] + list(reversed(P[i+1:]))
                    # evaluate w_co of new path
                    _, wco_newP = path_optimize_fixed_order(D, clusters, newP)
                    evals_total += 1
                    # GainIsAcceptable (rule 4/5)
                    if accept(wco_newP, T_cost, m):
                        new_reps, new_T = cluster_optimize_fixed_order(D, clusters, newP)
                        evals_total += 1
                        if new_T + 1e-12 < T_cost:
                            # accept improvement
                            order = newP
                            reps = new_reps
                            T_cost = new_T
                            improved_any = True
                            improvements += 1
                            found_local = True
                            if verbose:
                                print(f"  LK-E improved: {T_cost:.6f}")
                            break  # restart from new tour
                if improved_any:
                    break  # restart outer rotation loop

        if T_cost < best_cost:
            best_cost = T_cost
            best_order = order[:]
            best_reps = reps[:]
            if verbose:
                print(f"  >> new global best (LK-E): {best_cost:.6f}")

    elapsed = time.time() - t0
    return GTSPResult(
        best_order=best_order,
        best_reps=best_reps,
        best_cost=best_cost,
        stats={
            "elapsed_sec": elapsed,
            "passes": passes_total,
            "evals": evals_total,
            "restarts": restarts,
            "clusters": len(clusters),
            "nodes": sum(len(c) for c in clusters),
            "method": "lk-exact",
            "accept_rule": accept_rule,
            "improvements": improvements,
        },
    )

def make_demo_instance(
    m_clusters: int, nodes_per_cluster: int, seed: int = 0
) -> Tuple[List[List[float]], List[List[int]]]:
    rng = random.Random(seed)
    points: List[Tuple[float, float]] = []
    clusters: List[List[int]] = []
    idx = 0
    import math
    grid_side = math.ceil(math.sqrt(m_clusters))
    spacing = 100.0
    jitter = 25.0
    for k in range(m_clusters):
        gx = k % grid_side
        gy = k // grid_side
        cx = gx * spacing
        cy = gy * spacing
        cluster_nodes = []
        for _ in range(nodes_per_cluster):
            x = cx + rng.uniform(-jitter, jitter)
            y = cy + rng.uniform(-jitter, jitter)
            points.append((x, y))
            cluster_nodes.append(idx)
            idx += 1
        clusters.append(cluster_nodes)
    D = build_euclidean_D(points)
    return D, clusters

def load_dist_csv(path: str) -> List[List[float]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        D = [[float(x) for x in row] for row in reader]
    n = len(D)
    for row in D:
        if len(row) != n:
            raise ValueError("Distance CSV must be square (N x N).")
    return D

def load_clusters_json(path: str) -> List[List[int]]:
    with open(path, "r") as f:
        clusters = json.load(f)
    seen = set()
    for idx, c in enumerate(clusters):
        if not c:
            raise ValueError(f"Cluster {idx} is empty.")
        for v in c:
            if v in seen:
                raise ValueError(f"Node {v} appears in multiple clusters.")
            seen.add(v)
    return clusters

def main():
    ap = argparse.ArgumentParser(description="LK-style heuristics for GTSP")
    ap.add_argument("--dist-csv", type=str, default=None,
                    help="Path to N×N distance matrix CSV")
    ap.add_argument("--clusters-json", type=str, default=None,
                    help="Path to clusters JSON (e.g., [[0,3],[1,2,4],...])")
    ap.add_argument("--demo", type=int, default=0,
                    help="If >0, build a random Euclidean demo instance of this many clusters")
    ap.add_argument("--nodes-per-cluster", type=int, default=6,
                    help="Nodes per cluster when using --demo")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--restarts", type=int, default=8, help="Restarts (multi-start)")
    ap.add_argument("--method", type=str, default="order-2opt",
                    choices=["order-2opt", "lk-exact"],
                    help="Heuristic: baseline 2-opt on cluster order, or LK Exact (E) variant")
    ap.add_argument("--accept", type=int, default=4, choices=[4,5],
                    help="Acceptance rule for LK-E (4=optimistic, 5=half-average)")
    ap.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = ap.parse_args()

    if args.demo and (args.dist_csv or args.clusters_json):
        raise SystemExit("Use either --demo or explicit --dist-csv/--clusters-json, not both.")

    if args.demo > 0:
        D, clusters = make_demo_instance(args.demo, args.nodes_per_cluster, args.seed)
    else:
        if not args.dist_csv or not args.clusters_json:
            raise SystemExit("Provide both --dist-csv and --clusters-json, or use --demo.")
        D = load_dist_csv(args.dist_csv)
        clusters = load_clusters_json(args.clusters_json)

    if args.method == "order-2opt":
        res = solve_gtsp_order2opt(
            D, clusters,
            seed=args.seed,
            time_limit_s=args.time_limit,
            restarts=args.restarts,
            verbose=not args.quiet,
        )
    else:
        # LK-E typically needs fewer restarts; allow override via --restarts
        res = solve_gtsp_lk_exact(
            D, clusters,
            seed=args.seed,
            time_limit_s=args.time_limit,
            restarts=max(1, args.restarts//2),
            accept_rule=args.accept,
            verbose=not args.quiet,
        )

    print("\n=== GTSP Heuristic Result ===")
    print(f"Method: {res.stats.get('method','order-2opt')}")
    print(f"Best cost: {res.best_cost:.6f}")
    print(f"Best order (clusters): {res.best_order}")
    print(f"Best representatives (nodes in that order): {res.best_reps}")
    print("Stats:", res.stats)

if __name__ == "__main__":
    main()
