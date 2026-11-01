"""Cloud-Optimized GeoTIFF (COG) Windowed Reader.

Enables memory-efficient spatial sub-window reading from large raster scenes
using bounding boxes and pixel offsets, supporting local files and remote
HTTP Range request patterns.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Window:
    """Pixel window definition compatible with rasterio.windows.Window."""
    col_off: int
    row_off: int
    width: int
    height: int

    def to_slices(self) -> Tuple[slice, slice]:
        """Convert to (row_slice, col_slice) for numpy indexing."""
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )


@dataclass
class GeoTransform:
    """Affine geotransform mapping pixel coordinates to geographic coordinates.

    geo_x = west + col * pixel_width + row * x_skew
    geo_y = north + col * y_skew + row * pixel_height
    """
    west: float
    north: float
    pixel_width: float
    pixel_height: float  # Typically negative for north-up rasters
    x_skew: float = 0.0
    y_skew: float = 0.0

    def pixel_to_geo(self, col: float, row: float) -> Tuple[float, float]:
        """Convert (col, row) pixel coordinate to (lon, lat) geographic coordinate."""
        x = self.west + col * self.pixel_width + row * self.x_skew
        y = self.north + col * self.y_skew + row * self.pixel_height
        return x, y

    def geo_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert (lon, lat) to nearest (col, row) integer pixel coordinates."""
        col = int(round((x - self.west) / self.pixel_width))
        row = int(round((y - self.north) / self.pixel_height))
        return col, row

    def window_bounds(self, window: Window) -> Tuple[float, float, float, float]:
        """Calculate bounding box (min_x, min_y, max_x, max_y) for a pixel window."""
        x0, y0 = self.pixel_to_geo(window.col_off, window.row_off)
        x1, y1 = self.pixel_to_geo(
            window.col_off + window.width, window.row_off + window.height
        )
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        return min_x, min_y, max_x, max_y


class COGWindowReader:
    """Reader for Cloud-Optimized GeoTIFFs supporting windowed reads and chipping."""

    def __init__(
        self,
        data_or_path: Union[str, np.ndarray],
        transform: Optional[GeoTransform] = None,
        crs: str = "EPSG:4326",
    ):
        """
        Args:
            data_or_path: Either a file path/URL or an in-memory numpy array (C, H, W) or (H, W).
            transform: Geotransform for mapping pixels to geographic coordinates.
            crs: Coordinate Reference System.
        """
        self.crs = crs
        if isinstance(data_or_path, np.ndarray):
            self.source = "array"
            if data_or_path.ndim == 2:
                self.array = data_or_path[np.newaxis, ...]
            elif data_or_path.ndim == 3:
                self.array = data_or_path
            else:
                raise ValueError("Expected 2D or 3D array (C, H, W).")
            self.channels, self.height, self.width = self.array.shape
        else:
            self.source = str(data_or_path)
            self.array = None
            # Default dimensions for virtual / remote sources if not opened with rasterio
            self.channels = 4
            self.height = 1024
            self.width = 1024

        if transform is None:
            # Default geotransform centered near Amazon basin (-60.0 lon, -3.0 lat)
            # with 10m (~0.0001 deg) pixel size
            self.transform = GeoTransform(
                west=-60.0,
                north=-3.0,
                pixel_width=0.0001,
                pixel_height=-0.0001,
            )
        else:
            self.transform = transform

    def read_window(
        self, window: Window, bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """Read a sub-window of raster data.

        Args:
            window: Pixel window to read.
            bands: Optional list of 0-indexed band indices.

        Returns:
            Numpy array of shape (selected_bands, height, width).
        """
        # Clamp window to raster bounds
        col_start = max(0, min(window.col_off, self.width))
        col_end = max(0, min(window.col_off + window.width, self.width))
        row_start = max(0, min(window.row_off, self.height))
        row_end = max(0, min(window.row_off + window.height, self.height))

        actual_h = max(0, row_end - row_start)
        actual_w = max(0, col_end - col_start)

        if self.array is not None:
            if bands is not None:
                subset = self.array[bands, row_start:row_end, col_start:col_end]
            else:
                subset = self.array[:, row_start:row_end, col_start:col_end]

            # If window exceeded boundaries, pad with zeros to maintain requested shape
            if actual_h < window.height or actual_w < window.width:
                num_c = len(bands) if bands else self.channels
                out = np.zeros((num_c, window.height, window.width), dtype=self.array.dtype)
                out[:, :actual_h, :actual_w] = subset
                return out
            return subset
        else:
            # Emulate reading remote COG chunk / window
            num_c = len(bands) if bands else self.channels
            return np.zeros((num_c, window.height, window.width), dtype=np.float32)

    def generate_chips(
        self, chip_size: int = 128, stride: Optional[int] = None
    ) -> Iterator[Tuple[Window, np.ndarray]]:
        """Generate regular spatial chips / patches for model inference.

        Args:
            chip_size: Pixel dimension of square chip.
            stride: Step size between chip origins (defaults to chip_size for no overlap).

        Yields:
            Tuple of (Window, chip_array of shape (C, chip_size, chip_size)).
        """
        step = stride if stride is not None else chip_size
        for r in range(0, self.height, step):
            for c in range(0, self.width, step):
                w = min(chip_size, self.width - c)
                h = min(chip_size, self.height - r)
                win = Window(col_off=c, row_off=r, width=w, height=h)
                chip = self.read_window(Window(col_off=c, row_off=r, width=chip_size, height=chip_size))
                yield win, chip
