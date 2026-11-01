"""EU Deforestation Regulation (EUDR) Compliance and Validation Engine.

Enforces legal mandates under Regulation (EU) 2023/1115:
1. Temporal Baseline Cutoff: December 31, 2020.
2. Geolocation Requirements: Point coordinates permitted for <= 4 ha;
   closed polygon boundaries mandatory for > 4 ha plots.
3. Coordinate precision validation (>= 6 decimal places).
4. Mandatory commodity scope verification.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from geofm.postprocessing.geometry_utils import (
    calculate_polygon_area_ha,
    is_point_in_polygon,
    polygons_intersect,
    validate_coordinates_precision,
)

# Mandatory EUDR regulatory cutoff date: December 31, 2020
EUDR_CUTOFF_DATE = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Regulated commodities under EUDR Annex I
EUDR_REGULATED_COMMODITIES = {
    "cattle",
    "cocoa",
    "coffee",
    "oil palm",
    "rubber",
    "soya",
    "wood",
}


class EUDRValidator:
    """Validates plot geolocations and compliance status against EUDR rules."""

    def __init__(self, cutoff_date: Optional[datetime] = None):
        self.cutoff_date = cutoff_date if cutoff_date is not None else EUDR_CUTOFF_DATE

    def validate_commodity(self, commodity: str) -> Tuple[bool, str]:
        """Verify whether commodity falls under mandatory EUDR Annex I scope."""
        c = commodity.lower().strip()
        if c in EUDR_REGULATED_COMMODITIES:
            return True, f"Commodity '{commodity}' is covered under EUDR Annex I regulation."
        return (
            False,
            f"Commodity '{commodity}' is outside the scope of EUDR Annex I. "
            f"Mandatory regulated commodities are: {', '.join(sorted(EUDR_REGULATED_COMMODITIES))}.",
        )

    def validate_plot_geometry(
        self,
        geometry_type: str,
        coordinates: Union[List[float], List[List[float]], List[List[List[float]]]],
        declared_area_ha: Optional[float] = None,
    ) -> Tuple[bool, str, float]:
        """Validate plot geometry against EUDR size and format mandates.

        Returns:
            Tuple of (is_valid, message, calculated_area_ha).
        """
        geom_type = geometry_type.lower()

        if geom_type == "point":
            # Coordinates must be [lon, lat]
            if len(coordinates) != 2:
                return False, "Point coordinates must be [longitude, latitude].", 0.0

            lon, lat = float(coordinates[0]), float(coordinates[1])
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                return False, f"Coordinates out of bounds: lon={lon}, lat={lat}", 0.0

            # Under EUDR, point coordinates are strictly disallowed if plot > 4 ha
            area = declared_area_ha if declared_area_ha is not None else 1.0
            if area > 4.0:
                return (
                    False,
                    f"EUDR Mandate Violation: Plot area ({area:.2f} ha) exceeds 4 ha threshold. "
                    f"A closed continuous polygon boundary is legally required, point coordinates rejected.",
                    area,
                )

            # Precision check
            valid_prec, prec_msg = validate_coordinates_precision([[lon, lat]], min_decimals=6)
            if not valid_prec:
                return False, f"Coordinate precision failure: {prec_msg}", area

            return True, "Valid point plot under 4 ha threshold.", area

        elif geom_type == "polygon":
            # Unwrap standard GeoJSON Polygon format [[ [lon, lat], ... ]]
            coords = coordinates
            if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and len(coords[0]) > 0 and isinstance(coords[0][0], (list, tuple)):
                coords = coords[0]

            if len(coords) < 4:
                return False, "Polygon must contain at least 4 coordinates (3 unique + 1 closing).", 0.0

            # Check closure
            if (coords[0][0] != coords[-1][0] or
                    coords[0][1] != coords[-1][1]):
                return False, "Polygon is not closed: first coordinate must equal last coordinate.", 0.0

            # Bounds check
            for pt in coords:
                lon, lat = pt[0], pt[1]
                if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                    return False, f"Polygon vertex out of bounds: [{lon}, {lat}].", 0.0

            # Precision check
            valid_prec, prec_msg = validate_coordinates_precision(coords, min_decimals=6)
            if not valid_prec:
                return False, f"Coordinate precision failure: {prec_msg}", 0.0

            calculated_area = calculate_polygon_area_ha(coords)
            return True, f"Valid polygon with area {calculated_area:.2f} ha.", calculated_area

        else:
            return False, f"Unsupported geometry type: {geometry_type}", 0.0

    def verify_deforestation_timing(
        self, disturbance_datetime: datetime
    ) -> Tuple[bool, str]:
        """Verify whether a detected deforestation event violates the EUDR cutoff.

        Returns:
            Tuple of (is_violation, explanation).
            True if disturbance occurred AFTER Dec 31, 2020 (EUDR VIOLATION).
        """
        if disturbance_datetime.tzinfo is None:
            disturbance_datetime = disturbance_datetime.replace(tzinfo=timezone.utc)

        if disturbance_datetime > self.cutoff_date:
            return (
                True,
                f"EUDR VIOLATION: Disturbance detected on {disturbance_datetime.isoformat()} "
                f"which is AFTER the regulatory cutoff date {self.cutoff_date.isoformat()}.",
            )
        else:
            return (
                False,
                f"Compliant: Disturbance date {disturbance_datetime.isoformat()} "
                f"is on or prior to the cutoff date {self.cutoff_date.isoformat()} (historical change).",
            )
