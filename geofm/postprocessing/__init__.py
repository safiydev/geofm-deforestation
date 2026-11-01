"""Post-processing and vector polygonization modules."""

from geofm.postprocessing.geometry_utils import (
    calculate_polygon_area_ha,
    is_point_in_polygon,
    polygons_intersect,
    validate_coordinates_precision,
)
from geofm.postprocessing.douglas_peucker import douglas_peucker, simplify_polygon
from geofm.postprocessing.polygonizer import Polygonizer

__all__ = [
    "calculate_polygon_area_ha",
    "is_point_in_polygon",
    "polygons_intersect",
    "validate_coordinates_precision",
    "douglas_peucker",
    "simplify_polygon",
    "Polygonizer",
]
