"""Douglas-Peucker polygon simplification algorithm.

Reduces redundant vertices in segmented deforestation vector boundaries
while preserving geographic topological integrity.
"""

import math
from typing import List


def _perpendicular_distance(
    pt: List[float], line_start: List[float], line_end: List[float]
) -> float:
    """Calculate Euclidean / planar perpendicular distance from point to line segment."""
    x0, y0 = pt[0], pt[1]
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_end[1]

    dx = x2 - x1
    dy = y2 - y1

    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0.0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    numerator = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
    return numerator / denom


def douglas_peucker(points: List[List[float]], epsilon: float) -> List[List[float]]:
    """Simplify a line string using the Douglas-Peucker algorithm.

    Args:
        points: Ordered list of [x, y] coordinates.
        epsilon: Distance threshold tolerance.

    Returns:
        Simplified list of [x, y] coordinates.
    """
    if len(points) <= 2:
        return points

    dmax = 0.0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = _perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        # Recursive step
        rec_results1 = douglas_peucker(points[: index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[end]]


def simplify_polygon(
    coordinates: List[List[float]], epsilon: float = 0.0001
) -> List[List[float]]:
    """Simplify a closed polygon boundary ring while ensuring valid closure.

    Splits ring at its two furthest extremal vertices (geometric diameter)
    and applies Douglas-Peucker along both chains to eliminate collinear
    redundant vertices across all boundary edges.

    Args:
        coordinates: List of [lon, lat] points.
        epsilon: Simplification tolerance in degrees (~0.0001 deg is ~11 meters).

    Returns:
        Simplified polygon coordinates maintaining closure and minimum vertex count.
    """
    if len(coordinates) <= 3:
        return coordinates

    ring = [list(p) for p in coordinates]
    is_closed = (
        ring[0][0] == ring[-1][0]
        and ring[0][1] == ring[-1][1]
    )

    open_ring = ring[:-1] if is_closed else ring
    n = len(open_ring)

    if n <= 3:
        res = list(open_ring)
        if is_closed:
            res.append(list(res[0]))
        return res

    # Find the two vertices furthest apart to serve as natural chain anchors
    max_d_sq = -1.0
    idx1, idx2 = 0, n // 2

    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (open_ring[i][0] - open_ring[j][0]) ** 2 + (
                open_ring[i][1] - open_ring[j][1]
            ) ** 2
            if d_sq > max_d_sq:
                max_d_sq = d_sq
                idx1, idx2 = i, j

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    chain1 = open_ring[idx1 : idx2 + 1]
    chain2 = open_ring[idx2:] + open_ring[: idx1 + 1]

    sim1 = douglas_peucker(chain1, epsilon)
    sim2 = douglas_peucker(chain2, epsilon)

    # Combine chains without duplicating anchor endpoints
    simplified = sim1[:-1] + sim2[:-1]

    # Ensure a valid polygon ring has at least 3 distinct vertices
    if len(simplified) < 3:
        simplified = open_ring[:3]

    if is_closed:
        simplified.append(list(simplified[0]))

    return simplified
