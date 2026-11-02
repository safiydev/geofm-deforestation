"""STAC API Deforestation Data Loader.

Provides search and query utilities for Microsoft Planetary Computer and Copernicus Data Space
to retrieve multi-spectral Sentinel-2 L2A scenes alongside PRODES Amazon deforestation alerts.
"""

from typing import Any, Dict, List, Optional, Tuple

PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COPERNICUS_STAC = "https://catalogue.dataspace.copernicus.eu/stac"


class STACDeforestationLoader:
    """Queries STAC APIs for Sentinel-2 bitemporal scenes matching forest alert polygons."""

    def __init__(self, endpoint: str = PLANETARY_COMPUTER_STAC):
        self.endpoint = endpoint

    def build_search_query(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str,
        collection: str = "sentinel-2-l2a",
        max_cloud_cover: float = 15.0,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Construct STAC API JSON POST body for bitemporal satellite tile discovery."""
        return {
            "collections": [collection],
            "bbox": list(bbox),
            "datetime": date_range,
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover}
            },
            "limit": limit,
        }
