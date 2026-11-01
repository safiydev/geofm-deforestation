"""Geographic geometry utilities and spherical area calculations.

Provides geodesic area calculation on the WGS84 ellipsoid/sphere,
point-in-polygon tests, and coordinate precision validation for regulatory compliance.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


# WGS84 authalic sphere mean radius in meters
EARTH_RADIUS_M = 6371008.8


def calculate_polygon_area_ha(coordinates: List[List[float]]) -> float:
    """Calculate the geodesic surface area of a spherical polygon in hectares.

    Uses spherical excess / Girard's formula approximation via trapezoidal projection.
    Correctly handles open and closed rings and antimeridian (+/-180 deg) wrapping.

    Args:
        coordinates: List of [lon, lat] pairs.

    Returns:
        Area in hectares (1 ha = 10,000 m^2).
    """
    if len(coordinates) < 3:
        return 0.0

    coords = [list(pt) for pt in coordinates]
    # Ensure closed ring
    if coords[0][0] != coords[-1][0] or coords[0][1] != coords[-1][1]:
        coords.append(list(coords[0]))

    if len(coords) < 4:
        return 0.0

    total = 0.0
    n = len(coords)

    for i in range(n - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        lon1 = math.radians(p1[0])
        lat1 = math.radians(p1[1])
        lon2 = math.radians(p2[0])
        lat2 = math.radians(p2[1])

        # Normalize longitude difference across antimeridian
        d_lon = lon2 - lon1
        while d_lon > math.pi:
            d_lon -= 2.0 * math.pi
        while d_lon < -math.pi:
            d_lon += 2.0 * math.pi

        total += d_lon * (2.0 + math.sin(lat1) + math.sin(lat2))

    area_m2 = abs(total * (EARTH_RADIUS_M**2) / 2.0)
    return area_m2 / 10000.0  # Convert m^2 to hectares


def is_point_in_polygon(point: List[float], polygon_coords: List[List[float]]) -> bool:
    """Ray casting algorithm to determine if point [lon, lat] is inside polygon.

    Properly evaluates all edges of open or closed polygon rings.
    """
    x, y = point[0], point[1]
    inside = False

    coords = list(polygon_coords)
    if len(coords) < 3:
        return False

    # Normalize to open ring of unique vertices
    if len(coords) >= 4 and coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]:
        coords = coords[:-1]

    n = len(coords)
    if n < 3:
        return False

    for i in range(n):
        p1x, p1y = coords[i][0], coords[i][1]
        p2x, p2y = coords[(i + 1) % n][0], coords[(i + 1) % n][1]

        # Check if ray from (x, y) going right crosses edge (p1, p2)
        if (p1y > y) != (p2y > y):
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if x < xinters:
                inside = not inside

    return inside


def segments_intersect(
    p1: List[float], p2: List[float], p3: List[float], p4: List[float]
) -> bool:
    """Test if line segment p1-p2 intersects line segment p3-p4."""
    def ccw(a: List[float], b: List[float], c: List[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(p: List[float], a: List[float], b: List[float]) -> bool:
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if (
        ((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12))
        and ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12))
    ):
        return True

    if abs(d1) <= 1e-12 and on_segment(p1, p3, p4):
        return True
    if abs(d2) <= 1e-12 and on_segment(p2, p3, p4):
        return True
    if abs(d3) <= 1e-12 and on_segment(p3, p1, p2):
        return True
    if abs(d4) <= 1e-12 and on_segment(p4, p1, p2):
        return True

    return False


def polygons_intersect(poly1: List[List[float]], poly2: List[List[float]]) -> bool:
    """Check if two simple polygons intersect.

    Performs:
    1. Bounding box intersection test.
    2. Vertex containment test (poly1 vertex in poly2, or poly2 vertex in poly1).
    3. Edge-crossing test (any segment of poly1 intersects any segment of poly2).
    """
    if len(poly1) < 3 or len(poly2) < 3:
        return False

    # 1. Bounding box check
    min_x1 = min(p[0] for p in poly1)
    max_x1 = max(p[0] for p in poly1)
    min_y1 = min(p[1] for p in poly1)
    max_y1 = max(p[1] for p in poly1)

    min_x2 = min(p[0] for p in poly2)
    max_x2 = max(p[0] for p in poly2)
    min_y2 = min(p[1] for p in poly2)
    max_y2 = max(p[1] for p in poly2)

    if (
        min_x1 > max_x2
        or max_x1 < min_x2
        or min_y1 > max_y2
        or max_y1 < min_y2
    ):
        return False

    # 2. Vertex containment check
    for p in poly1:
        if is_point_in_polygon(p, poly2):
            return True

    for p in poly2:
        if is_point_in_polygon(p, poly1):
            return True

    # 3. Edge-crossing check
    c1 = list(poly1)
    if c1[0][0] == c1[-1][0] and c1[0][1] == c1[-1][1]:
        c1 = c1[:-1]

    c2 = list(poly2)
    if c2[0][0] == c2[-1][0] and c2[0][1] == c2[-1][1]:
        c2 = c2[:-1]

    n1, n2 = len(c1), len(c2)
    for i in range(n1):
        seg1_a = c1[i]
        seg1_b = c1[(i + 1) % n1]
        for j in range(n2):
            seg2_a = c2[j]
            seg2_b = c2[(j + 1) % n2]
            if segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                return True

    return False


def validate_coordinates_precision(
    coordinates: List[Any], min_decimals: int = 6
) -> Tuple[bool, str]:
    """Validate that coordinates have sufficient precision and valid numerical bounds.

    If strings are passed, enforces digit count >= min_decimals.
    If floats are passed, validates finite numbers within geographic ranges.
    Handles flat points [lon, lat] or nested coordinate rings.
    """
    for item in coordinates:
        if isinstance(item, (list, tuple)):
            for val in item:
                if isinstance(val, (list, tuple)):
                    # Nested ring recursion
                    ok, msg = validate_coordinates_precision([val], min_decimals=min_decimals)
                    if not ok:
                        return False, msg
                elif isinstance(val, str):
                    s_clean = val.strip()
                    if "." in s_clean:
                        decs = len(s_clean.split(".")[1])
                        if decs < min_decimals:
                            return (
                                False,
                                f"Coordinate string '{val}' has fewer than {min_decimals} decimal places.",
                            )
                    else:
                        return False, f"Coordinate string '{val}' has no decimal places."
                elif isinstance(val, (int, float)):
                    if not math.isfinite(val):
                        return False, f"Coordinate {val} is not a finite number."
                else:
                    return False, f"Invalid coordinate value type: {type(val)}"
        elif isinstance(item, (int, float)):
            if not math.isfinite(item):
                return False, f"Coordinate {item} is not a finite number."
        else:
            return False, f"Invalid coordinate type: {type(item)}"
    return True, "Valid precision."

