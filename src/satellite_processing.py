"""Sentinel-1 flood detection and Sentinel-2 indices (array-level, testable offline).

Caveats built into outputs: SAR cannot see floodwater under dense canopy/urban areas
(and wind roughens water); optical cannot see through cloud. Absence of detection is
NEVER evidence of absence of flooding.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

def to_db(linear: np.ndarray) -> np.ndarray:
    lin = np.where(linear > 0, linear, np.nan)
    return 10.0 * np.log10(lin)

def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Simple Lee speckle filter on linear-power data (NaN-safe via fill with local mean)."""
    x = np.where(np.isnan(img), np.nanmean(img), img).astype(float)
    m = ndi.uniform_filter(x, size); m2 = ndi.uniform_filter(x * x, size)
    var = np.maximum(m2 - m * m, 0)
    noise = np.nanmean(var / np.maximum(m * m, 1e-12))      # global noise variance estimate
    k = var / (var + noise * m * m + 1e-12)
    out = m + k * (x - m)
    out[np.isnan(img)] = np.nan
    return out

@dataclass
class FloodResult:
    flood: np.ndarray            # bool: likely newly inundated open water
    permanent_water: np.ndarray  # bool: water in pre-event image
    excluded: np.ndarray         # bool: masked (terrain / invalid), status UNKNOWN not 'dry'
    warnings: list

def detect_flood_s1(pre_lin, post_lin, slope_deg=None, hand_m=None,
                    water_db=-18.0, diff_db=-3.0, max_slope=5.0, max_hand=15.0,
                    min_pixels=10, filt_size=5) -> FloodResult:
    """Change-detection flood mapping from pre/post VV linear backscatter.
    Flood = post is water AND pre was not AND drop >= |diff_db| AND terrain plausible.
    Thresholds are generic defaults; calibrate per scene/season."""
    if pre_lin.shape != post_lin.shape:
        raise ValueError("pre/post shape mismatch")
    pre, post = to_db(lee_filter(pre_lin, filt_size)), to_db(lee_filter(post_lin, filt_size))
    invalid = np.isnan(pre) | np.isnan(post)
    excluded = invalid.copy()
    if slope_deg is not None:
        excluded |= slope_deg > max_slope          # radar shadow/layover & steep terrain
    if hand_m is not None:
        excluded |= hand_m > max_hand              # too high above drainage to flood plausibly
    perm = (pre < water_db) & ~invalid
    flood = (post < water_db) & ~perm & ((post - pre) <= diff_db) & ~excluded
    lab, n = ndi.label(flood)                      # minimum mapping unit removes speckle noise
    if n:
        sizes = ndi.sum(flood, lab, index=np.arange(1, n + 1))
        flood = np.isin(lab, np.where(sizes >= min_pixels)[0] + 1)
    warns = ["SAR limits: floodwater under dense vegetation/urban areas is mostly undetected; "
             "wind-roughened water may be missed. Unflagged pixels are NOT confirmed dry."]
    if excluded.mean() > 0.5:
        warns.append(f"{excluded.mean():.0%} of the scene excluded (terrain/invalid): low coverage.")
    return FloodResult(flood, perm, excluded, warns)

def flooded_hectares(mask: np.ndarray, pixel_area_m2: float) -> float:
    """pixel_area_m2 must come from a projected grid (e.g. 10 m x 10 m = 100)."""
    return float(mask.sum() * pixel_area_m2 / 10_000.0)

# ---------------- Sentinel-2 ----------------
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)   # nodata, saturated, cloud shadow, cloud med/high, cirrus, snow

def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl.astype(int), SCL_INVALID)

def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)

def s2_indices(b02, b03, b04, b08, b11, scl, scale=1.0, L=0.5) -> dict:
    """Inputs are surface reflectance (0..1; pass scale=1/10000 for DN). Cloud pixels -> NaN."""
    ok = scl_valid_mask(scl)
    b = [np.where(ok, np.asarray(x, float) * scale, np.nan) for x in (b02, b03, b04, b08, b11)]
    blue, green, red, nir, swir = b
    with np.errstate(divide="ignore", invalid="ignore"):
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        savi = (1 + L) * (nir - red) / (nir + red + L)
    return {"NDVI": _nd(nir, red), "NDWI": _nd(green, nir), "MNDWI": _nd(green, swir),
            "EVI": evi, "SAVI": savi, "valid_fraction": float(ok.mean())}

def optical_usable(valid_fraction: float, min_fraction: float = 0.6) -> tuple:
    """If cloud covers too much, optical flood/drought inference is refused."""
    if valid_fraction < min_fraction:
        return False, (f"Only {valid_fraction:.0%} clear pixels: optical result UNRELIABLE; "
                       "this does not indicate absence of flooding.")
    return True, ""

def vegetation_anomaly(current: np.ndarray, baseline_stack: np.ndarray, min_years: int = 5) -> dict:
    """baseline_stack: (years, y, x) of same-season composites. Returns z-score and VCI (%).
    VCI needs >= min_years; NDVI anomalies alone are not a drought diagnosis (harvest, fire,
    land-use change, pests also lower NDVI)."""
    n = np.sum(~np.isnan(baseline_stack), axis=0)
    mu, sd = np.nanmean(baseline_stack, 0), np.nanstd(baseline_stack, 0)
    lo, hi = np.nanmin(baseline_stack, 0), np.nanmax(baseline_stack, 0)
    ok = (n >= min_years) & (sd > 0) & (hi > lo)
    z = np.where(ok, (current - mu) / np.where(sd > 0, sd, np.nan), np.nan)
    vci = np.where(ok, 100 * (current - lo) / np.where(hi > lo, hi - lo, np.nan), np.nan)
    return {"zscore": z, "VCI": vci}
