"""SpatioTemporal Asset Catalog (STAC) generator and query engine.

Constructs compliant STAC Collections, Items, and Assets for Sentinel-2,
Sentinel-1, and Planet NICFI observations with EO, SAR, and Projection extensions.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pystac
from pystac.extensions.eo import EOExtension, Band
from pystac.extensions.sar import SarExtension, Polarization, FrequencyBand
from pystac.extensions.projection import ProjectionExtension


class STACManager:
    """Manages STAC catalogs, collections, and items for deforestation monitoring."""

    def __init__(self, catalog_id: str = "geofm-deforestation-catalog", title: str = "GeoFM Deforestation Detection Catalog"):
        self.catalog = pystac.Catalog(
            id=catalog_id,
            description="Geospatial Foundation Model Deforestation and EUDR Compliance Catalog",
            title=title,
        )

    def create_sentinel2_item(
        self,
        item_id: str,
        bbox: List[float],
        dt: datetime,
        asset_hrefs: Dict[str, str],
        epsg: int = 4326,
        cloud_cover: float = 5.0,
    ) -> pystac.Item:
        """Create a STAC Item for Sentinel-2 optical imagery with EO extension."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        min_x, min_y, max_x, max_y = bbox
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        }

        item = pystac.Item(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=dt,
            properties={
                "eo:cloud_cover": cloud_cover,
                "platform": "sentinel-2",
                "constellation": "sentinel-2",
                "instruments": ["msi"],
                "processing:level": "Level-2A",
            },
        )

        # Apply EO extension
        eo_ext = EOExtension.ext(item, add_if_missing=True)

        # Band configurations for Sentinel-2
        band_configs = {
            "B02": Band.create(name="B02", common_name="blue", center_wavelength=490),
            "B03": Band.create(name="B03", common_name="green", center_wavelength=560),
            "B04": Band.create(name="B04", common_name="red", center_wavelength=665),
            "B08": Band.create(name="B08", common_name="nir", center_wavelength=842),
            "B11": Band.create(name="B11", common_name="swir16", center_wavelength=1610),
            "B12": Band.create(name="B12", common_name="swir22", center_wavelength=2190),
        }

        for band_name, href in asset_hrefs.items():
            asset = pystac.Asset(
                href=href,
                media_type=pystac.MediaType.COG,
                roles=["data"],
                title=f"Sentinel-2 Band {band_name}",
            )
            item.add_asset(band_name, asset)
            if band_name in band_configs:
                eo_asset = EOExtension.ext(asset, add_if_missing=False)
                eo_asset.bands = [band_configs[band_name]]

        # Apply Projection extension
        proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
        proj_ext.epsg = epsg
        proj_ext.bbox = bbox

        return item

    def create_sentinel1_item(
        self,
        item_id: str,
        bbox: List[float],
        dt: datetime,
        asset_hrefs: Dict[str, str],
        epsg: int = 4326,
        orbit_state: str = "descending",
    ) -> pystac.Item:
        """Create a STAC Item for Sentinel-1 C-band SAR imagery with SAR extension."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        min_x, min_y, max_x, max_y = bbox
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        }

        item = pystac.Item(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=dt,
            properties={
                "platform": "sentinel-1",
                "constellation": "sentinel-1",
                "instruments": ["c-sar"],
                "sar:instrument_mode": "IW",
                "sar:frequency_band": FrequencyBand.C,
                "sar:polarizations": [Polarization.VV, Polarization.VH],
                "sat:orbit_state": orbit_state,
            },
        )

        sar_ext = SarExtension.ext(item, add_if_missing=True)
        sar_ext.instrument_mode = "IW"
        sar_ext.frequency_band = FrequencyBand.C
        sar_ext.polarizations = [Polarization.VV, Polarization.VH]

        for pol_name, href in asset_hrefs.items():
            asset = pystac.Asset(
                href=href,
                media_type=pystac.MediaType.COG,
                roles=["data"],
                title=f"Sentinel-1 {pol_name.upper()} Polarization",
            )
            item.add_asset(pol_name, asset)

        proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
        proj_ext.epsg = epsg
        proj_ext.bbox = bbox

        return item

    def query_catalog(
        self,
        items: List[pystac.Item],
        bbox: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        platform: Optional[str] = None,
    ) -> List[pystac.Item]:
        def _to_utc(d: Optional[datetime]) -> Optional[datetime]:
            if d is None:
                return None
            if d.tzinfo is None:
                return d.replace(tzinfo=timezone.utc)
            return d.astimezone(timezone.utc)

        start_utc = _to_utc(start_date)
        end_utc = _to_utc(end_date)
        results = []

        for item in items:
            # Platform check
            if platform and item.properties.get("platform") != platform:
                continue

            # Datetime check
            item_dt = _to_utc(item.datetime)
            if item_dt:
                if start_utc and item_dt < start_utc:
                    continue
                if end_utc and item_dt > end_utc:
                    continue

            # Bounding box spatial intersection check
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                i_min_x, i_min_y, i_max_x, i_max_y = item.bbox
                # Check for overlap
                if (min_x > i_max_x or max_x < i_min_x or
                        min_y > i_max_y or max_y < i_min_y):
                    continue

            results.append(item)
        return results
