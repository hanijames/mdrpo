import math
import heapq
from typing import List, Tuple, Dict, Optional
from functools import lru_cache

import numpy as np

from geometry import Point, point_in_poly, point_segment_sqdist
from instance import Obstacle, union_contains


BOUNDARY_TOL = 1e-9
FREEDOM_BUFFER = 0.1  # Prevents numerical issues when points are near obstacle vertices


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_strictly_inside_poly(p: Point, poly_verts: List[Point], tol: float = BOUNDARY_TOL) -> bool:
    x, y = p
    n = len(poly_verts)

    for i in range(n):
        ax, ay = poly_verts[i]
        bx, by = poly_verts[(i + 1) % n]
        if abs(_orient(ax, ay, bx, by, x, y)) < tol:
            if (min(ax, bx) - tol <= x <= max(ax, bx) + tol and
                    min(ay, by) - tol <= y <= max(ay, by) + tol):
                return False  # On boundary

    return point_in_poly(p, poly_verts)


def _segment_crosses_polygon_interior(p1: Point, p2: Point, poly_verts: List[Point]) -> bool:
    n = len(poly_verts)

    if _point_strictly_inside_poly(p1, poly_verts):
        return True
    if _point_strictly_inside_poly(p2, poly_verts):
        return True

    for i in range(n):
        a = poly_verts[i]
        b = poly_verts[(i + 1) % n]

        o1 = _orient(a[0], a[1], b[0], b[1], p1[0], p1[1])
        o2 = _orient(a[0], a[1], b[0], b[1], p2[0], p2[1])
        o3 = _orient(p1[0], p1[1], p2[0], p2[1], a[0], a[1])
        o4 = _orient(p1[0], p1[1], p2[0], p2[1], b[0], b[1])

        if (o1 * o2 < 0) and (o3 * o4 < 0):
            return True

    for t in [0.25, 0.5, 0.75]:
        sample = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        if _point_strictly_inside_poly(sample, poly_verts):
            return True

    return False


def visible(a: Point, b: Point, obstacles: List[Obstacle]) -> bool:
    for obs in obstacles:
        if _segment_crosses_polygon_interior(a, b, obs.verts):
            return False
    return True


@lru_cache(maxsize=None)
def _cached_edges(verts_tuple: Tuple[Tuple[float, float], ...]) -> List[Tuple[float, float, float, float]]:
    edges = []
    n = len(verts_tuple)
    for i in range(n):
        x1, y1 = verts_tuple[i]
        x2, y2 = verts_tuple[(i + 1) % n]
        edges.append((x1, y1, x2, y2))
    return edges


def freedom_radius(p: Point, obstacles: List[Obstacle], shrink_freedom: float = 1.0) -> float:
    if union_contains(p, obstacles):
        return 0.0
    cx, cy = float(p[0]), float(p[1])
    best_d2 = math.inf
    for obs in obstacles:
        edges = _cached_edges(tuple(obs.verts))
        for x1, y1, x2, y2 in edges:
            d2 = point_segment_sqdist(cx, cy, x1, y1, x2, y2)
            if d2 < best_d2:
                best_d2 = d2
                if best_d2 <= 1e-18:
                    break
    if not math.isfinite(best_d2):
        return 0.0
    return max(0.0, shrink_freedom * math.sqrt(best_d2) - FREEDOM_BUFFER)


def build_los_graph(obstacles: List[Obstacle]) -> Tuple[List[Point], Dict[int, List[Tuple[int, float]]]]:
    verts: List[Point] = []
    for obs in obstacles:
        verts.extend(obs.verts)
    m = len(verts)
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(m)}
    for i in range(m):
        for j in range(i + 1, m):
            if visible(verts[i], verts[j], obstacles):
                d = math.hypot(verts[i][0] - verts[j][0], verts[i][1] - verts[j][1])
                adj[i].append((j, d))
                adj[j].append((i, d))
    return verts, adj


def apsp_vertices_with_next(base_adj: Dict[int, List[Tuple[int, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    M = len(base_adj)
    D = [[math.inf] * M for _ in range(M)]
    Next = [[-1] * M for _ in range(M)]
    for s in range(M):
        dist = [math.inf] * M
        parent = [-1] * M
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in base_adj.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        D[s] = dist
        for t in range(M):
            if t == s or not math.isfinite(dist[t]):
                Next[s][t] = -1
                continue
            cur = t
            prev = parent[cur]
            while prev != -1 and prev != s:
                cur, prev = prev, parent[prev]
            Next[s][t] = cur if prev == s else -1
    return np.asarray(D, dtype=float), np.asarray(Next, dtype=int)


def visible_vertex_indices_from_point(p: Point, base_verts, obstacles: List[Obstacle]) -> List[int]:
    vis = []
    for j, vj in enumerate(base_verts):
        if visible(p, vj, obstacles):
            vis.append(j)
    return vis


def wet_route_cost_via_Dvv(p: Point, q: Point, base_verts, obstacles: List[Obstacle], Dvv: np.ndarray) -> float:
    if visible(p, q, obstacles):
        return float(math.hypot(q[0] - p[0], q[1] - p[1]))
    V = np.asarray(base_verts, dtype=float)
    P = np.array(p, dtype=float)
    Q = np.array(q, dtype=float)
    VA = visible_vertex_indices_from_point(p, base_verts, obstacles)
    VB = visible_vertex_indices_from_point(q, base_verts, obstacles)
    if not VA:
        VA = [int(np.argmin(np.linalg.norm(V - P, axis=1)))]
    if not VB:
        VB = [int(np.argmin(np.linalg.norm(V - Q, axis=1)))]
    VA_arr = np.array(VA, dtype=int)
    VB_arr = np.array(VB, dtype=int)
    p_to_VA = np.linalg.norm(V[VA_arr] - P, axis=1)
    VB_to_q = np.linalg.norm(V[VB_arr] - Q, axis=1)
    D_sub = Dvv[np.ix_(VA_arr, VB_arr)]
    vals = p_to_VA[:, None] + D_sub + VB_to_q[None, :]
    return float(np.min(vals))


def first_last_turn(a: Point, b: Point, obstacles: List[Obstacle], base_verts, Dvv: np.ndarray) -> Tuple[
    Optional[Point], Optional[Point], float]:
    a = (float(a[0]), float(a[1]))
    b = (float(b[0]), float(b[1]))
    V = np.asarray(base_verts, dtype=float)
    if visible(a, b, obstacles):
        return None, None, 0.0
    VA = visible_vertex_indices_from_point(a, base_verts, obstacles)
    VB = visible_vertex_indices_from_point(b, base_verts, obstacles)
    if not VA or not VB:
        return None, None, float("inf")
    VA_arr = np.array(VA, dtype=int)
    VB_arr = np.array(VB, dtype=int)
    A_to_VA = np.linalg.norm(V[VA_arr] - np.array(a, float), axis=1)
    VB_to_B = np.linalg.norm(V[VB_arr] - np.array(b, float), axis=1)
    D_sub = Dvv[np.ix_(VA_arr, VB_arr)]
    totals = A_to_VA[:, None] + D_sub + VB_to_B[None, :]
    mask = np.isfinite(totals)
    if not mask.any():
        return None, None, float("inf")
    ii, jj = np.unravel_index(int(np.argmin(np.where(mask, totals, np.inf))), totals.shape)
    vi, vj = int(VA_arr[ii]), int(VB_arr[jj])
    first = (float(V[vi, 0]), float(V[vi, 1]))
    last = (float(V[vj, 0]), float(V[vj, 1]))
    wrd = float(Dvv[vi, vj])
    return first, last, wrd


def wet_polyline(p: Point, q: Point, base_verts, base_adj: Dict[int, List[Tuple[int, float]]],
                 obstacles: List[Obstacle]) -> List[Point]:
    nodes: List[Point] = [p] + list(base_verts) + [q]
    P = 0
    Q = len(nodes) - 1
    M = len(base_verts)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(nodes))]
    for i in range(M):
        u = 1 + i
        for v_idx, w in base_adj.get(i, []):
            v = 1 + v_idx
            adj[u].append((v, w))
    for i, v in enumerate(base_verts):
        if visible(p, v, obstacles):
            d = math.hypot(p[0] - v[0], p[1] - v[1])
            adj[P].append((1 + i, d))
        if visible(q, v, obstacles):
            d = math.hypot(q[0] - v[0], q[1] - v[1])
            adj[1 + i].append((Q, d))
    if visible(p, q, obstacles):
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        adj[P].append((Q, d))
    dist_arr = [math.inf] * len(nodes)
    prev: List[Optional[int]] = [None] * len(nodes)
    dist_arr[P] = 0.0
    pq = [(0.0, P)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == Q:
            break
        if d != dist_arr[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not math.isfinite(dist_arr[Q]):
        return [p, q]
    path_idx = []
    cur = Q
    while cur is not None:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return [nodes[k] for k in path_idx]


def wet_distance(p: Point, q: Point, base_verts, base_adj: Dict[int, List[Tuple[int, float]]],
                 obstacles: List[Obstacle]) -> float:
    poly = wet_polyline(p, q, base_verts, base_adj, obstacles)
    s = 0.0
    for k in range(1, len(poly)):
        s += math.hypot(poly[k][0] - poly[k - 1][0], poly[k][1] - poly[k - 1][1])
    return s


def reconstruct_vertex_path(i: int, j: int, Next: np.ndarray) -> Optional[List[int]]:
    if i == j:
        return [i]
    if Next is None:
        return None
    n = Next.shape[0]
    if not (0 <= i < n and 0 <= j < n):
        return None
    path = [i]
    u = i
    for _ in range(n + 5):
        nh = int(Next[u, j])
        if nh < 0:
            return None
        path.append(nh)
        if nh == j:
            return path
        u = nh
    return None
