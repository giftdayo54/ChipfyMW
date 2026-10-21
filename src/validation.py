"""Score a satellite flood mask against a reference flood polygon file (e.g. UNOSAT / CEMS layer,
or hand-digitised). Excluded/unknown pixels are NOT counted as dry or wet: they are left out."""
import numpy as np
import geopandas as gpd

def load_reference(path_or_bytes) -> gpd.GeoDataFrame:
    import io
    g = gpd.read_file(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes)
    if g.crs is None: raise ValueError("Reference file has no CRS; refusing to guess.")
    return g.to_crs("EPSG:4326")

def rasterize_reference(ref: gpd.GeoDataFrame, bbox, shape) -> np.ndarray:
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    w, s, e, n = bbox; ny, nx = shape
    geoms = [g for g in ref.geometry if g is not None and not g.is_empty]
    if not geoms: return np.zeros(shape, bool)
    return rasterize(geoms, out_shape=shape, transform=from_origin(w, n, (e - w) / nx, (n - s) / ny),
                     fill=0, default_value=1, dtype="uint8").astype(bool)

def score_masks(pred: np.ndarray, ref: np.ndarray, excluded: np.ndarray = None) -> dict:
    if pred.shape != ref.shape: raise ValueError("pred/ref shape mismatch")
    ok = np.ones(pred.shape, bool) if excluded is None else ~excluded
    p, r = pred[ok], ref[ok]
    tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())
    out = {"tp": tp, "fp": fp, "fn": fn, "pixels_scored": int(ok.sum()),
           "pixels_excluded": int((~ok).sum())}
    if tp + fn == 0:
        return {**out, "status": "no_reference_flood_in_scored_area",
                "note": "Reference has no flood pixels here; recall/IoU undefined."}
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {**out, "status": "ok", "precision": prec, "recall": rec, "f1": f1, "iou": tp / (tp + fp + fn),
            "note": "Pixel-based, single event: not evidence of general skill. Reference itself has errors."}
