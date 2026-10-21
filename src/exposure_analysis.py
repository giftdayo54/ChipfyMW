"""Exposure = what lies inside a hazard footprint. Needs aligned rasters; never guesses."""
import numpy as np

def _check(*arrs):
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("Rasters must share one grid (resample/align first).")

def exposed_hectares(hazard_mask, area_ha, asset_mask=None):
    """Hectares inside hazard (optionally only where asset_mask, e.g. cropland/sugarcane, is True)."""
    m = hazard_mask.astype(bool)
    if asset_mask is not None:
        _check(m, area_ha, asset_mask); m = m & asset_mask.astype(bool)
    else:
        _check(m, area_ha)
    return float(area_ha[m].sum())

def population_exposed(hazard_mask, pop_grid):
    """Sum of population cells inside hazard. Gridded population is modelled (e.g. WorldPop), so
    treat as an order-of-magnitude estimate. NaN cells are ignored and counted."""
    _check(hazard_mask, pop_grid)
    inside = pop_grid[hazard_mask.astype(bool)]
    return {"people": float(np.nansum(inside)), "nan_cells_in_hazard": int(np.isnan(inside).sum()),
            "caveat": "Modelled population grid; not a census count."}

def population_exposure(pop, pop_transform, flood, excluded, bbox) -> dict:
    """Areal-weighted population exposure. Each population cell is assumed uniformly populated; the
    flood/excluded masks (fine grid) are averaged onto the population grid to get per-cell fractions.
    Returns exposed, unassessed (unknown) and total people for the window."""
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    if flood.shape != excluded.shape: raise ValueError("flood/excluded shapes differ")
    w, s, e, n = bbox; ny, nx = flood.shape
    src_tr = from_origin(w, n, (e - w) / nx, (n - s) / ny)
    def to_pop(a):
        dst = np.zeros(pop.shape, "float32")
        reproject(a.astype("float32"), dst, src_transform=src_tr, src_crs="EPSG:4326",
                  dst_transform=pop_transform, dst_crs="EPSG:4326", resampling=Resampling.average)
        return dst
    f, x, cov = to_pop(flood), to_pop(excluded), to_pop(np.ones(flood.shape))
    p = np.nan_to_num(pop, nan=0.0)
    return {"total_people": float(np.nansum(pop)), "exposed_people": float((p * f).sum()),
            "unassessed_people": float((p * np.clip(x + (1 - cov), 0, 1)).sum()),
            "nodata_cells": int(np.isnan(pop).sum()), "cells": int(pop.size),
            "caveat": ("Modelled gridded population (e.g. WorldPop), assumed uniform within each cell; "
                       "order-of-magnitude estimate, not a census count. 'Unassessed' people live where the "
                       "satellite could not judge flooding; they are not confirmed safe.")}
