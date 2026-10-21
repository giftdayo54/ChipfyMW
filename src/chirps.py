"""CHIRPS v2.0 rainfall adapter (Climate Hazards Center, UCSB). Africa 0.05 deg GeoTIFFs.

CHIRPS is a satellite + station BLENDED ESTIMATE, not a gauge observation. Recent months are
preliminary and may be revised. Cite: Funk et al. (2015), Scientific Data 2:150066.
URL layout follows the CHC public archive; it was not exercised live during development.
Missing/unpublished/failed months stay NaN and are listed in attrs['failed'] (never filled)."""
import gzip, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
import requests
from . import data_sources as ds

BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
MIN_VALID_FRAC = 0.5

def url_monthly(y: int, m: int) -> str:
    return f"{BASE}/africa_monthly/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz"

def url_daily(d: date) -> str:
    return f"{BASE}/africa_daily/tifs/p05/{d.year}/chirps-v2.0.{d:%Y.%m.%d}.tif.gz"

def window_mean(gz_bytes: bytes, bbox) -> tuple:
    """Mean over cells inside bbox (nodata -> NaN). Returns (mean, n_valid, n_total)."""
    import rasterio
    from rasterio.windows import from_bounds
    raw = gzip.decompress(gz_bytes)
    with rasterio.MemoryFile(raw) as mf, mf.open() as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError("Unexpected CRS in CHIRPS tile")
        arr = src.read(1, window=from_bounds(*bbox, transform=src.transform), masked=True).astype(float).filled(np.nan)
    arr[arr < 0] = np.nan                                   # -9999 nodata guard
    n_valid = int(np.isfinite(arr).sum())
    return (float(np.nanmean(arr)) if n_valid else float("nan")), n_valid, int(arr.size)

def _download(url: str, session, retries: int) -> Optional[bytes]:
    """Bytes, or None if the file does not exist (HTTP 404 = not published). Other failures raise."""
    get = (session or requests).get
    last = None
    for i in range(retries):
        try:
            r = get(url, timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status(); return r.content
        except requests.RequestException as e:
            last = e; time.sleep(min(2 ** i, 8) if retries > 1 else 0)
    raise ds.DataUnavailable(f"{url}: {last}")

def _one(url, bbox, session, retries):
    raw = _download(url, session, retries)
    if raw is None: return float("nan"), "not published (404)"
    mean, nv, nt = window_mean(raw, bbox)
    if nt == 0 or nv / nt < MIN_VALID_FRAC: return float("nan"), f"only {nv}/{nt} valid cells in window"
    return mean, ""

def _cache_file(bbox, kind):
    return ds.CACHE_DIR / f"chirps_{kind}_{hashlib.sha256(json.dumps(bbox).encode()).hexdigest()[:16]}.json"

def _load(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def _run(keys_urls: dict, bbox, session, workers, retries, progress):
    out, failed = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, u, bbox, session, retries): k for k, u in keys_urls.items()}
        for n, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            try:
                v, why = f.result()
                if why: failed.append((k, why))
                else: out[k] = v
            except Exception as e:
                failed.append((k, str(e)))
            if progress: progress(n, len(futs))
    return out, sorted(failed)

def monthly_series(bbox, start_year: int, end_year: int, session=None, workers: int = 4, retries: int = 3,
                   now: Optional[datetime] = None, progress=None) -> pd.Series:
    """District-window mean monthly rainfall (mm/month). Compatible with rainfall_analysis.spi()."""
    bbox = ds.validate_bbox(bbox)
    if start_year < 1981 or end_year < start_year: raise ValueError("CHIRPS starts in 1981; check year range")
    now = now or datetime.now(timezone.utc)
    months = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13) if (y, m) < (now.year, now.month)]
    cf = _cache_file(bbox, "monthly"); cache = _load(cf)
    todo = {f"{y}-{m:02d}": url_monthly(y, m) for y, m in months if f"{y}-{m:02d}" not in cache}
    got, failed = _run(todo, bbox, session, workers, retries, progress) if todo else ({}, [])
    if got:
        cache.update(got); ds.CACHE_DIR.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(cache))
    idx = pd.date_range(f"{start_year}-01-01", periods=len(months), freq="MS")
    s = pd.Series([cache.get(f"{y}-{m:02d}", np.nan) for y, m in months], index=idx, name="precip_mm")
    s.attrs.update(source="CHIRPS v2.0 Africa monthly (0.05 deg)", failed=failed, bbox=bbox, kind="satellite-gauge blended estimate",
                   retrieved_at=datetime.now(timezone.utc).isoformat(),
                   note="Blended estimate, not gauge data; recent months preliminary and may be revised.")
    return s

def daily_series(bbox, start: date, end: date, session=None, workers: int = 4, retries: int = 3, progress=None) -> pd.Series:
    """District-window mean daily rainfall (mm/day), max 400 days per call. Latest days may be unpublished."""
    bbox = ds.validate_bbox(bbox)
    days = pd.date_range(start, end, freq="D")
    if len(days) == 0 or len(days) > 400: raise ValueError("Request 1..400 days per call")
    cf = _cache_file(bbox, "daily"); cache = _load(cf)
    todo = {f"{d:%Y-%m-%d}": url_daily(d.date()) for d in days if f"{d:%Y-%m-%d}" not in cache}
    got, failed = _run(todo, bbox, session, workers, retries, progress) if todo else ({}, [])
    if got:
        cache.update(got); ds.CACHE_DIR.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(cache))
    s = pd.Series([cache.get(f"{d:%Y-%m-%d}", np.nan) for d in days], index=days, name="precip_mm")
    s.attrs.update(source="CHIRPS v2.0 Africa daily (0.05 deg)", failed=failed, bbox=bbox, kind="satellite-gauge blended estimate",
                   retrieved_at=datetime.now(timezone.utc).isoformat(), note="Blended estimate, not gauge data; preliminary.")
    return s

def completeness(s: pd.Series) -> dict:
    n = len(s); miss = int(s.isna().sum())
    return {"months_or_days": n, "missing": miss, "fraction_missing": (miss / n) if n else 1.0}
