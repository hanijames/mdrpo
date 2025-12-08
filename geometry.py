import math
from typing import List, Tuple

Point = Tuple[float, float]


def point_in_poly(p: Point, poly: List[Point]) -> bool:
    # Ray casting algorithm: count intersections with edges extending rightward from p
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-16) + x1
            if x_int > x:
                inside = not inside
    return inside


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_sqdist(cx: float, cy: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return (cx - x1) ** 2 + (cy - y1) ** 2
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    px, py = x1 + t * dx, y1 + t * dy
    return (cx - px) ** 2 + (cy - py) ** 2
