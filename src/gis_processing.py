"""Area, terrain and normalisation utilities."""
import numpy as np
from .config import AREA_CRS
from .data_validation import validate_crs

def area_hectares(gdf, area_crs: str = AREA_CRS) -> float:
    """Total area in hectares, computed in a projected CRS (never in degrees)."""
    validate_crs(gdf)
    return float(gdf.to_crs(area_crs).geometry.area.sum() / 10_000.0)

def slope_degrees(dem: np.ndarray, cellsize_m: float) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(float), cellsize_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def minmax(a: np.ndarray, invert: bool = False) -> np.ndarray:
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi == lo:
        return np.full_like(a, np.nan)       # no variation: not informative
    n = (a - lo) / (hi - lo)
    return 1 - n if invert else n

def relative_elevation(dem: np.ndarray, river_mask: np.ndarray) -> np.ndarray:
    """Height above the lowest river cell (coarse proxy; not a HAND product)."""
    if not river_mask.any():
        raise ValueError("river_mask is empty")
    return dem - np.nanmin(dem[river_mask])

def cell_area_ha_grid(west: float, north: float, dlon: float, dlat: float, shape: tuple) -> np.ndarray:
    """Per-cell area (ha) for a lat/lon grid (spherical-earth integral; error <<1% at this scale).
    dlon > 0, dlat > 0 (cell size in degrees); rows run north->south from `north`."""
    R = 6371008.8
    ny, nx = shape
    lat_top = np.radians(north - np.arange(ny) * dlat)
    lat_bot = np.radians(north - (np.arange(ny) + 1) * dlat)
    a = R ** 2 * np.radians(dlon) * (np.sin(lat_top) - np.sin(lat_bot))
    return np.repeat(np.abs(a)[:, None], nx, axis=1) / 10_000.0

def read_raster_window(path: str, bbox):
    """Windowed read of a GeoTIFF (avoids loading the whole raster). Returns (array, transform, crs)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        if src.crs is None: raise ValueError("Raster has no CRS")
        w = from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=w, masked=True).astype(float).filled(np.nan)
        return arr, src.window_transform(w), src.crs

def read_raster_window_bytes(raw: bytes, bbox):
    """Like read_raster_window but from uploaded bytes. Requires EPSG:4326 (e.g. WorldPop)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.MemoryFile(raw) as mf, mf.open() as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError("Raster must be in EPSG:4326 (reproject first).")
        w = from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=w, masked=True).astype(float).filled(np.nan)
        return arr, src.window_transform(w)
