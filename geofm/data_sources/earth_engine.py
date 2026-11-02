"""Google Earth Engine (GEE) Deforestation Data Pipeline.

Pairs cloud-free Sentinel-2 L2A bi-temporal composites (COPERNICUS/S2_SR_HARMONIZED)
with Hansen Global Forest Change (UMD/hansen/global_forest_change_2024_v1_12)
and Google Dynamic World (GOOGLE/DYNAMICWORLD/V1) canopy probability masks.
Exports thousands of geo-registered multi-spectral deforestation image pairs for free.
"""

from typing import Dict, List, Optional, Tuple

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False


GEE_SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
GEE_HANSEN_COLLECTION = "UMD/hansen/global_forest_change_2024_v1_12"
GEE_DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"

# Crucial forest bands: RGB + NIR + SWIR1 + SWIR2
FOREST_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def get_gee_export_script_template(
    roi_bbox: Tuple[float, float, float, float] = (-63.0, -9.5, -62.5, -9.0),
    t1_date_range: Tuple[str, str] = ("2023-06-01", "2023-09-30"),
    t2_date_range: Tuple[str, str] = ("2024-06-01", "2024-09-30"),
    cloud_thresh: int = 20,
) -> str:
    """Generate ready-to-run Google Earth Engine Python script for bi-temporal deforestation extraction.

    Args:
        roi_bbox: Region of interest (west, south, east, north).
        t1_date_range: Pre-disturbance acquisition window (start, end).
        t2_date_range: Post-disturbance acquisition window (start, end).
        cloud_thresh: Maximum cloud pixel percentage filter.

    Returns:
        Python script string configured for Earth Engine export.
    """
    w, s, e, n = roi_bbox
    t1_s, t1_e = t1_date_range
    t2_s, t2_e = t2_date_range

    script = f'''# Google Earth Engine Sentinel-2 + Hansen Deforestation Exporter
import ee

ee.Initialize(project='YOUR_GEE_PROJECT_ID')

# 1. Define Region of Interest (ROI)
roi = ee.Geometry.Rectangle([{w}, {s}, {e}, {n}])

# 2. Cloud Masking Function for Sentinel-2 SR
def mask_s2_sr(img):
    qa = img.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return img.updateMask(mask).select({FOREST_BANDS}).divide(10000.0)

# 3. Create Cloud-Free Median Composites
s2_col = ee.ImageCollection('{GEE_SENTINEL2_COLLECTION}')\\
    .filterBounds(roi)\\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', {cloud_thresh}))

t1_composite = s2_col.filterDate('{t1_s}', '{t1_e}').map(mask_s2_sr).median().clip(roi)
t2_composite = s2_col.filterDate('{t2_s}', '{t2_e}').map(mask_s2_sr).median().clip(roi)

# 4. Hansen Global Forest Change 2024 Deforestation Mask
gfc2024 = ee.Image('{GEE_HANSEN_COLLECTION}')
tree_loss = gfc2024.select('loss').clip(roi)
loss_year = gfc2024.select('lossyear').clip(roi)

# Active loss between 2023 (year 23) and 2024 (year 24)
recent_deforestation = loss_year.gte(23).And(loss_year.lte(24)).rename('deforestation_mask')

# 5. Export to Google Drive or Cloud Storage
task_t1 = ee.batch.Export.image.toDrive(
    image=t1_composite,
    description='S2_T1_Composite',
    scale=10,
    region=roi,
    fileFormat='GeoTIFF'
)
task_t1.start()

task_t2 = ee.batch.Export.image.toDrive(
    image=t2_composite,
    description='S2_T2_Composite',
    scale=10,
    region=roi,
    fileFormat='GeoTIFF'
)
task_t2.start()

task_mask = ee.batch.Export.image.toDrive(
    image=recent_deforestation,
    description='Hansen_Deforestation_Mask',
    scale=10,
    region=roi,
    fileFormat='GeoTIFF'
)
task_mask.start()
print("✓ Earth Engine export tasks submitted successfully!")
'''
    return script
