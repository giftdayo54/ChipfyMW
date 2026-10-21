"""Sentinel-1 download -> flood mask -> hectares + polygons, for ONE district window.

`fetch` is any callable payload->bytes (normally CDSEClient.process), so the pipeline is testable
offline. Output is satellite-derived flood EXTENT evidence for the stated acquisitions only:
not a forecast, not depth, and unflagged pixels are not confirmed dry."""
import math
from datetime import datetime, timezone
import numpy as np
import geopandas as gpd
from . import data_sources as ds, satellite_processing as sp, gis_processing as gp

def parse_tiff(raw: bytes) -> np.ndarray:
    import rasterio
    with rasterio.MemoryFile(raw) as mf, mf.open() as src:
        return src.read().astype(float)

def _grid(bbox, res_m):
    w, s, e, n = bbox
    lat = math.radians((s + n) / 2)
    width = math.ceil((e - w) * 111_320 * math.cos(lat) / res_m)
    height = math.ceil((n - s) * 110_574 / res_m)
    if max(width, height) > 2500:
        raise ValueError(f"{width}x{height} px exceeds 2500; increase res_m or shrink the window.")
    return width, height

def run_s1_flood(fetch, bbox, pre_range, post_range, res_m: float = 30.0, fields=None, **detect_kw) -> dict:
    bbox = ds.validate_bbox(bbox); w, s, e, n = bbox
    width, height = _grid(bbox, res_m)
    def sar(rng):
        b = parse_tiff(fetch(ds.process_payload("s1", bbox, rng[0], rng[1], width, height)))
        vv = np.where(b[2] > 0, b[0], np.nan)          # dataMask==0 -> no data
        return vv
    pre, post = sar(pre_range), sar(post_range)
    dem = parse_tiff(fetch(ds.process_payload("dem", bbox, "", "", width, height)))[0]
    if not (pre.shape == post.shape == dem.shape):
        raise ValueError("Pre/post/DEM grids differ")
    dlon, dlat = (e - w) / width, (n - s) / height
    cell_m = 0.5 * (dlon * 111_320 * math.cos(math.radians((s + n) / 2)) + dlat * 110_574)
    slope = gp.slope_degrees(dem, cell_m)
    warns = []
    wdb = detect_kw.get("water_db", -18.0)
    perm0 = sp.to_db(sp.lee_filter(pre)) < wdb
    hand = gp.relative_elevation(dem, perm0) if perm0.any() else None
    if hand is None:
        warns.append("No permanent water in window: height-above-water terrain mask skipped.")
    res = sp.detect_flood_s1(pre, post, slope_deg=slope, hand_m=hand, **detect_kw)
    area = gp.cell_area_ha_grid(w, n, dlon, dlat, pre.shape)
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    geoms = [{"geometry": g, "val": v} for g, v in shapes(res.flood.astype("uint8"), mask=res.flood,
             transform=from_origin(w, n, dlon, dlat))]
    gdf = gpd.GeoDataFrame.from_features([{"type": "Feature", "geometry": g["geometry"], "properties": {}} for g in geoms],
                                         crs="EPSG:4326") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    warns += res.warnings + ["Pre/post mosaics may mix orbits/incidence angles; compare same-orbit scenes where possible.",
                             "Height-above-water mask is a coarse proxy, not HAND; thresholds are uncalibrated defaults."]
    table = None
    if fields is not None:
        from . import field_mask as fm
        fl = fm.clip_to_bbox(fields, bbox)
        if len(fl):
            ids = fm.field_id_raster(fl, bbox, pre.shape)
            table = fm.zonal_summary(fl, ids, area, {"flood_ha": res.flood, "unknown_ha": res.excluded})
        else:
            warns.append("No field polygons intersect this window.")
    return {"field_table": table, "flood_ha": float(area[res.flood].sum()), "permanent_water_ha": float(area[res.permanent_water].sum()),
            "excluded_fraction": float(res.excluded.mean()), "polygons": gdf, "warnings": warns,
            "meta": {"bbox": bbox, "pre": pre_range, "post": post_range, "resolution_m": res_m,
                     "source": "Sentinel-1 GRD via Copernicus Data Space; DEM Copernicus GLO-30",
                     "processed_at": datetime.now(timezone.utc).isoformat()},
            "arrays": {"flood": res.flood, "excluded": res.excluded}}
