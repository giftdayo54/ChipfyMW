"""Field-boundary mask: restrict satellite results to real field polygons and report per field.
Overlapping polygons: the later feature wins the pixel (documented limitation)."""
import io
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from .config import AREA_CRS

def load_fields(raw, name_col: str = None) -> gpd.GeoDataFrame:
    g = gpd.read_file(io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else raw)
    if g.crs is None: raise ValueError("Field file has no CRS; refusing to guess.")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.buffer(0)                      # repair minor invalid rings
    g = g.to_crs("EPSG:4326").reset_index(drop=True)
    g["field_name"] = g[name_col].astype(str) if name_col and name_col in g else [f"field_{i+1}" for i in range(len(g))]
    return g[["field_name", "geometry"]]

def clip_to_bbox(fields: gpd.GeoDataFrame, bbox) -> gpd.GeoDataFrame:
    return fields[fields.intersects(box(*bbox))].reset_index(drop=True)

def field_id_raster(fields, bbox, shape) -> np.ndarray:
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    w, s, e, n = bbox; ny, nx = shape
    if len(fields) == 0: return np.zeros(shape, np.int32)
    return rasterize([(g, i + 1) for i, g in enumerate(fields.geometry)], out_shape=shape,
                     transform=from_origin(w, n, (e - w) / nx, (n - s) / ny), fill=0, dtype="int32")

def zonal_summary(fields, ids: np.ndarray, area_ha: np.ndarray, masks: dict, valid: np.ndarray = None) -> pd.DataFrame:
    """masks: column -> bool array (ha where True). valid: optional bool array -> 'assessed_ha'."""
    n = len(fields)
    def agg(m): return np.bincount(ids.ravel(), weights=(area_ha * m).ravel(), minlength=n + 1)[1:]
    df = pd.DataFrame({"field_name": fields["field_name"].values,
                       "field_ha": fields.to_crs(AREA_CRS).geometry.area.values / 1e4,
                       "covered_ha": agg(ids > 0)})
    if valid is not None: df["assessed_ha"] = agg(valid)
    for k, m in masks.items(): df[k] = agg(m)
    return df
