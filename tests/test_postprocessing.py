"""Tests for geometric utilities, Douglas-Peucker simplification, and polygonization."""

import math
import numpy as np
import pytest

from geofm.postprocessing.geometry_utils import (
    calculate_polygon_area_ha,
    is_point_in_polygon,
    polygons_intersect,
    validate_coordinates_precision,
)
from geofm.postprocessing.douglas_peucker import simplify_polygon, douglas_peucker
from geofm.postprocessing.polygonizer import Polygonizer
from geofm.preprocessing.cog_reader import GeoTransform


def test_polygon_area_and_point_in_polygon():
    # ~1km x 1km square near equator (~100 hectares)
    # 0.009 degrees is approx 1 km at the equator
    coords = [
        [0.0, 0.0],
        [0.009, 0.0],
        [0.009, 0.009],
        [0.0, 0.009],
        [0.0, 0.0],
    ]
    area_ha = calculate_polygon_area_ha(coords)
    assert 90.0 < area_ha < 110.0  # Around 100 ha

    # Point containment
    inside_pt = [0.004, 0.004]
    outside_pt = [0.020, 0.020]
    assert is_point_in_polygon(inside_pt, coords) == True
    assert is_point_in_polygon(outside_pt, coords) == False


def test_polygon_intersection():
    poly1 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    poly2 = [
        [0.5, 0.5],
        [1.5, 0.5],
        [1.5, 1.5],
        [0.5, 1.5],
        [0.5, 0.5],
    ]
    poly3 = [
        [5.0, 5.0],
        [6.0, 5.0],
        [6.0, 6.0],
        [5.0, 6.0],
        [5.0, 5.0],
    ]
    assert polygons_intersect(poly1, poly2) == True
    assert polygons_intersect(poly1, poly3) == False


def test_douglas_peucker_simplification():
    # Collinear points with slight deviations
    line = [
        [0.0, 0.0],
        [1.0, 0.00001],
        [2.0, 0.00002],
        [3.0, 1.0],
        [4.0, 0.0],
    ]
    simplified = douglas_peucker(line, epsilon=0.01)
    # Intermediate collinear points should be eliminated
    assert len(simplified) < len(line)


def test_polygonizer_mmu_filter():
    polygonizer = Polygonizer(mmu_ha=0.5)  # 0.5 ha MMU
    transform = GeoTransform(west=-60.0, north=-3.0, pixel_width=0.0001, pixel_height=-0.0001)

    prob_map = np.zeros((100, 100), dtype=np.float32)

    # Cluster 1: Tiny noise (1 pixel ~ 0.01 ha) -> below MMU (should be discarded)
    prob_map[10, 10] = 0.95

    # Cluster 2: Large deforestation patch (20x20 pixels ~ 4.8 ha) -> exceeds MMU
    prob_map[40:60, 40:60] = 0.95

    geojson = polygonizer.polygonize(prob_map, threshold=0.5, transform=transform)
    features = geojson["features"]

    assert len(features) == 1  # Only the large patch survived MMU
    assert features[0]["properties"]["area_ha"] > 0.5


def test_polygon_intersection_crossing_without_vertices_inside():
    # Two thin perpendicular rectangles that intersect like a '+' cross,
    # where neither has any vertex inside the other.
    poly1 = [
        [-2.0, -0.5],
        [2.0, -0.5],
        [2.0, 0.5],
        [-2.0, 0.5],
        [-2.0, -0.5],
    ]
    poly2 = [
        [-0.5, -2.0],
        [0.5, -2.0],
        [0.5, 2.0],
        [-0.5, 2.0],
        [-0.5, -2.0],
    ]
    assert polygons_intersect(poly1, poly2) is True


def test_is_point_in_polygon_unclosed_triangle():
    # Triangle with vertices (5, 10), (10, 0), (0, 0)
    poly_open = [[5.0, 10.0], [10.0, 0.0], [0.0, 0.0]]
    # Point inside
    assert is_point_in_polygon([5.0, 2.0], poly_open) is True
    assert is_point_in_polygon([1.0, 1.0], poly_open) is True
    # Point outside to the left of the closing edge
    assert is_point_in_polygon([-1.0, 5.0], poly_open) is False


def test_closed_polygon_simplification_all_collinear_edges():
    # Square with redundant collinear midpoints along all 4 edges
    poly = [
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [10.0, 5.0],
        [10.0, 10.0],
        [5.0, 10.0],
        [0.0, 10.0],
        [0.0, 5.0],
        [0.0, 0.0],
    ]
    simplified = simplify_polygon(poly, epsilon=0.1)
    # Must simplify to the 4 corners + closing point = 5 points
    assert len(simplified) == 5
    assert [0.0, 5.0] not in simplified


def test_polygonizer_l_shape_and_multipolygon():
    polygonizer = Polygonizer(mmu_ha=0.01)
    transform = GeoTransform(west=-60.0, north=-3.0, pixel_width=0.0001, pixel_height=-0.0001)

    prob_map = np.zeros((30, 30), dtype=np.float32)
    # L-shaped connected component
    prob_map[5:15, 5:9] = 0.9
    prob_map[11:15, 5:20] = 0.9

    fc = polygonizer.polygonize(prob_map, threshold=0.5, transform=transform, as_multipolygon=False)
    assert len(fc["features"]) == 1

    # MultiPolygon generation
    mp_feat = polygonizer.to_multipolygon_feature(fc)
    assert mp_feat["type"] == "Feature"
    assert mp_feat["geometry"]["type"] == "MultiPolygon"
    assert len(mp_feat["geometry"]["coordinates"]) == 1
