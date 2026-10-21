"""Open-Meteo adapters (no key needed). Forecast = model output; archive = ERA5-based reanalysis.
Both are gridded model products for a POINT, not rain-gauge observations."""
import time
from datetime import datetime, timezone
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def _get(url, params, session=None, retries=3):
    s = session or requests
    last = None
    for i in range(retries):
        try:
            r = s.get(url, params=params, timeout=30); r.raise_for_status(); return r.json()
        except requests.RequestException as e:
            last = e; time.sleep(min(2 ** i, 8) if retries > 1 else 0)
    raise ConnectionError(f"Weather service unavailable: {last}")

def parse_daily(js: dict, kind: str, source: str) -> pd.DataFrame:
    d = js.get("daily") or {}
    if "time" not in d or "precipitation_sum" not in d:
        raise ValueError("Unexpected response: no daily precipitation")
    df = pd.DataFrame({"precip_mm": d["precipitation_sum"]}, index=pd.to_datetime(d["time"]))
    df.attrs.update(kind=kind, source=source, retrieved_at=datetime.now(timezone.utc).isoformat(),
                    note="Model grid-cell value, not a gauge. Model run/issue time not reported by this endpoint.")
    return df

def fetch_forecast(lat, lon, days=7, session=None) -> pd.DataFrame:
    if not 1 <= days <= 16: raise ValueError("days must be 1..16")
    js = _get(FORECAST_URL, {"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                             "forecast_days": days, "timezone": "Africa/Blantyre"}, session)
    return parse_daily(js, "forecast", "Open-Meteo forecast API")

def fetch_archive(lat, lon, start: str, end: str, session=None) -> pd.DataFrame:
    js = _get(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                            "daily": "precipitation_sum", "timezone": "Africa/Blantyre"}, session)
    return parse_daily(js, "reanalysis", "Open-Meteo archive (ERA5-based)")
