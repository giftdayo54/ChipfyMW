"""Exports that preserve CRS/metadata."""
import io
import numpy as np, pandas as pd

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()

def geojson_bytes(gdf) -> bytes:
    if gdf.crs is None: raise ValueError("GeoDataFrame has no CRS")
    return gdf.to_crs("EPSG:4326").to_json().encode()

def geotiff_bytes(arr: np.ndarray, west, north, dlon, dlat, tags: dict = None) -> bytes:
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                     crs="EPSG:4326", transform=from_origin(west, north, dlon, dlat), nodata=np.nan) as dst:
            dst.write(arr.astype("float32"), 1); dst.update_tags(**(tags or {}))
        return mf.read()
