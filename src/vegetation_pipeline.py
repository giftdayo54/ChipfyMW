"""Sentinel-2 vegetation condition for one district window (Copernicus Process API).
Output = vegetation STRESS evidence, not a drought diagnosis (harvest, fire, pests, land-use
change also lower NDVI). Cloudy scenes are refused rather than interpreted."""
import math
from datetime import datetime, timezone
import numpy as np
from . import data_sources as ds, satellite_processing as sp, gis_processing as gp
from .flood_pipeline import parse_tiff, _grid

def shift_year(ts: str, years: int) -> str:
    y = int(ts[:4]) - years
    return f"{y}{ts[4:]}".replace(f"{y}-02-29", f"{y}-02-28")

def _ndvi(fetch, bbox, rng, width, height):
    b = parse_tiff(fetch(ds.process_payload("s2", bbox, rng[0], rng[1], width, height, mosaicking="leastCC")))
    if b.shape[0] != 7: raise ValueError("Unexpected S2 band count")
    nodata = b[6] <= 0
    scl = np.where(nodata, 0, b[5])
    idx = sp.s2_indices(b[0], b[1], b[2], b[3], b[4], scl)
    return idx, nodata

def run_s2_condition(fetch, bbox, current_range, baseline_years: int = 5, res_m: float = 30.0,
                     min_clear: float = 0.6, fields=None) -> dict:
    bbox = ds.validate_bbox(bbox); w, s, e, n = bbox
    width, height = _grid(bbox, res_m)
    idx, _ = _ndvi(fetch, bbox, current_range, width, height)
    ok, msg = sp.optical_usable(idx["valid_fraction"], min_clear)
    meta = {"bbox": bbox, "window": current_range, "resolution_m": res_m, "baseline_years": baseline_years,
            "source": "Sentinel-2 L2A via Copernicus Data Space", "processed_at": datetime.now(timezone.utc).isoformat()}
    if not ok:
        return {"status": "unusable_cloud", "message": msg, "meta": meta}
    base, skipped = [], []
    for k in range(1, baseline_years + 1):
        rng = (shift_year(current_range[0], k), shift_year(current_range[1], k))
        i2, _ = _ndvi(fetch, bbox, rng, width, height)
        (base if sp.optical_usable(i2["valid_fraction"], min_clear)[0] else skipped).append(i2["NDVI"])
    res = {"status": "ok", "ndvi": idx["NDVI"], "valid_fraction": idx["valid_fraction"], "meta": meta,
           "baseline_used": len(base), "baseline_skipped_cloudy": len(skipped),
           "mean_ndvi": float(np.nanmean(idx["NDVI"])),
           "caveat": ("Vegetation stress only; not a drought diagnosis. Stress = VCI<35 AND z<=-1 vs the baseline years; "
                      "with few baseline years some false alarms occur even in normal seasons. "
                      "Check harvest/fire/land-use before attributing.")}
    if len(base) < 5:
        res.update(vci=None, zscore=None, stress_ha=None,
                   message=f"Only {len(base)} usable baseline years (<5): anomaly/VCI not computed.")
        return res
    an = sp.vegetation_anomaly(idx["NDVI"], np.stack(base))
    area = gp.cell_area_ha_grid(w, n, (e - w) / width, (n - s) / height, idx["NDVI"].shape)
    stressed = (an["VCI"] < 35) & (an["zscore"] <= -1.0)
    res.update(vci=an["VCI"], zscore=an["zscore"], stress_ha=float(area[stressed & ~np.isnan(an["VCI"])].sum()),
               assessed_ha=float(area[~np.isnan(an["VCI"])].sum()))
    res["field_table"] = None
    if fields is not None:
        from . import field_mask as fm
        fl = fm.clip_to_bbox(fields, bbox)
        if len(fl):
            ids = fm.field_id_raster(fl, bbox, idx["NDVI"].shape)
            valid = ~np.isnan(an["VCI"])
            res["field_table"] = fm.zonal_summary(fl, ids, area, {"stressed_ha": stressed & valid}, valid=valid)
            res["stress_ha_in_fields"] = float(res["field_table"]["stressed_ha"].sum())
        else:
            res["message"] = "No field polygons intersect this window."
    return res
