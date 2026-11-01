"""Raster-to-Vector Polygonization and Minimum Mapping Unit (MMU) filtering.

Extracts contiguous deforestation alert clusters from raster masks,
filters by Minimum Mapping Unit (MMU e.g. 0.1 - 0.2 ha), simplifies polygon
boundaries, and formats valid GeoJSON FeatureCollections for downstream GIS and DDS audits.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from geofm.preprocessing.cog_reader import GeoTransform
from geofm.postprocessing.douglas_peucker import simplify_polygon
from geofm.postprocessing.geometry_utils import calculate_polygon_area_ha


class Polygonizer:
    """Extracts, filters, and formats deforestation alert vector polygons."""

    def __init__(
        self,
        mmu_ha: float = 0.1,  # Minimum Mapping Unit in hectares
        simplify_tolerance_deg: float = 0.00005,  # ~5m boundary simplification
    ):
        self.mmu_ha = mmu_ha
        self.simplify_tolerance = simplify_tolerance_deg

    def _find_connected_components(
        self, binary_mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Label 8-connected components in binary 2D mask using two-pass algorithm."""
        h, w = binary_mask.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Pass 1: Assign preliminary labels and record equivalences
        for r in range(h):
            for c in range(w):
                if not binary_mask[r, c]:
                    continue

                neighbors = []
                # Check 8-connected predecessors: (r-1, c-1), (r-1, c), (r-1, c+1), (r, c-1)
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and labeled[nr, nc] > 0:
                        neighbors.append(labeled[nr, nc])

                if not neighbors:
                    current_label += 1
                    labeled[r, c] = current_label
                else:
                    min_l = min(neighbors)
                    labeled[r, c] = min_l
                    for l_val in neighbors:
                        union(l_val, min_l)

        # Pass 2: Resolve equivalence classes
        label_mapping = {}
        unique_id = 0
        for r in range(h):
            for c in range(w):
                if labeled[r, c] > 0:
                    root = find(labeled[r, c])
                    if root not in label_mapping:
                        unique_id += 1
                        label_mapping[root] = unique_id
                    labeled[r, c] = label_mapping[root]

        return labeled, unique_id

    def _trace_component_boundary(
        self, comp_mask: np.ndarray, transform: GeoTransform
    ) -> List[List[float]]:
        """Trace pixel-accurate outer boundary coordinates of a connected component.

        Extracts directed boundary edges separating foreground from background pixels,
        chains them into a closed topological contour ring, and maps pixel corners
        to (lon, lat) coordinates via the geotransform.
        """
        h, w = comp_mask.shape
        edges: Dict[Tuple[int, int], Tuple[int, int]] = {}
        rows, cols = np.where(comp_mask)
        if len(rows) == 0:
            return []

        # Find directed perimeter edges around pixel boxes
        for r, c in zip(rows, cols):
            # Top edge: (c, r) -> (c + 1, r)
            if r == 0 or not comp_mask[r - 1, c]:
                edges[(int(c), int(r))] = (int(c + 1), int(r))
            # Right edge: (c + 1, r) -> (c + 1, r + 1)
            if c == w - 1 or not comp_mask[r, c + 1]:
                edges[(int(c + 1), int(r))] = (int(c + 1), int(r + 1))
            # Bottom edge: (c + 1, r + 1) -> (c, r + 1)
            if r == h - 1 or not comp_mask[r + 1, c]:
                edges[(int(c + 1), int(r + 1))] = (int(c), int(r + 1))
            # Left edge: (c, r + 1) -> (c, r)
            if c == 0 or not comp_mask[r, c - 1]:
                edges[(int(c), int(r + 1))] = (int(c), int(r))

        if not edges:
            return []

        # Traverse edges to form closed outer ring
        start_node = min(edges.keys())
        ring = [start_node]
        curr = start_node
        visited = set()
        max_steps = len(edges) + 10

        while len(ring) < max_steps:
            visited.add(curr)
            next_node = edges.get(curr)
            if next_node is None:
                break
            ring.append(next_node)
            if next_node == start_node:
                break
            curr = next_node

        if len(ring) < 4 or ring[0] != ring[-1]:
            # Fallback to bounding box corners if ring did not close
            min_r, max_r = int(np.min(rows)), int(np.max(rows))
            min_c, max_c = int(np.min(cols)), int(np.max(cols))
            ring = [
                (min_c, min_r),
                (max_c + 1, min_r),
                (max_c + 1, max_r + 1),
                (min_c, max_r + 1),
                (min_c, min_r),
            ]

        # Convert pixel corners to geographic coordinates
        geo_coords = [list(transform.pixel_to_geo(c, r)) for c, r in ring]
        return geo_coords

    def to_multipolygon_feature(self, geojson_fc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a FeatureCollection of Polygons into a unified MultiPolygon Feature."""
        features = geojson_fc.get("features", [])
        multipoly_coords = []
        for feat in features:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                multipoly_coords.append(geom["coordinates"])
            elif geom.get("type") == "MultiPolygon":
                multipoly_coords.extend(geom["coordinates"])

        return {
            "type": "Feature",
            "id": "deforestation-multipolygon-layer",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": multipoly_coords,
            },
            "properties": geojson_fc.get("properties", {}),
        }

    def polygonize(
        self,
        probability_map: np.ndarray,
        threshold: float = 0.5,
        transform: Optional[GeoTransform] = None,
        as_multipolygon: bool = False,
    ) -> Dict[str, Any]:
        """Convert probability raster into compliant GeoJSON FeatureCollection or MultiPolygon.

        Args:
            probability_map: 2D array of deforestation probabilities [0, 1].
            threshold: Binarization threshold.
            transform: Geotransform mapping pixel coordinates to (lon, lat).
            as_multipolygon: If True, returns a GeoJSON MultiPolygon Feature.

        Returns:
            GeoJSON FeatureCollection or MultiPolygon dict.
        """
        if transform is None:
            transform = GeoTransform(
                west=-60.0, north=-3.0, pixel_width=0.0001, pixel_height=-0.0001
            )

        binary = probability_map >= threshold
        labeled_components, num_components = self._find_connected_components(binary)

        features: List[Dict[str, Any]] = []

        for comp_id in range(1, num_components + 1):
            comp_mask = labeled_components == comp_id
            pixel_count = int(np.sum(comp_mask))

            raw_coords = self._trace_component_boundary(comp_mask, transform)
            if len(raw_coords) < 4:
                continue

            simplified_coords = simplify_polygon(
                raw_coords, epsilon=self.simplify_tolerance
            )
            area_ha = calculate_polygon_area_ha(simplified_coords)

            # Minimum Mapping Unit (MMU) filtering: discard small noise patches
            if area_ha < self.mmu_ha:
                continue

            # Severity score: mean probability within the cluster
            mean_prob = float(np.mean(probability_map[comp_mask]))
            alert_level = (
                "CRITICAL"
                if (area_ha > 1.0 and mean_prob > 0.8)
                else "HIGH"
                if area_ha > 0.5
                else "MEDIUM"
            )

            centroid_lon = float(np.mean([pt[0] for pt in simplified_coords[:-1]]))
            centroid_lat = float(np.mean([pt[1] for pt in simplified_coords[:-1]]))

            feature = {
                "type": "Feature",
                "id": f"alert-cluster-{comp_id:04d}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [simplified_coords],
                },
                "properties": {
                    "cluster_id": comp_id,
                    "area_ha": round(area_ha, 3),
                    "pixel_count": pixel_count,
                    "mean_probability": round(mean_prob, 3),
                    "alert_level": alert_level,
                    "centroid": [round(centroid_lon, 6), round(centroid_lat, 6)],
                },
            }
            features.append(feature)

        fc = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "total_alerts": len(features),
                "total_deforested_ha": round(
                    sum(f["properties"]["area_ha"] for f in features), 3
                ),
                "mmu_filter_ha": self.mmu_ha,
            },
        }

        if as_multipolygon:
            return self.to_multipolygon_feature(fc)

        return fc
