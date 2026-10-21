"""Trend statistics and rainy-season metrics."""
import numpy as np
import pandas as pd
from scipy import stats

def mann_kendall(x) -> dict:
    """Mann-Kendall trend test + Sen's slope per time step (no tie correction; n>=8 required)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 8:
        return {"status": "insufficient_data", "n": n}
    i, j = np.triu_indices(n, 1)
    d = x[j] - x[i]
    s = np.sign(d).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope = float(np.median(d / (j - i)))
    return {"status": "ok", "n": n, "z": float(z), "p": float(p), "sen_slope": slope,
            "trend": "none significant (p>=0.05)" if p >= 0.05 else ("increasing" if z > 0 else "decreasing")}

def rainy_season_totals(daily: pd.Series, start_month: int = 11, end_month: int = 4,
                        min_coverage: float = 0.9) -> pd.Series:
    """Nov-Apr totals labelled by the year the season starts. Seasons with poor coverage -> NaN."""
    d = daily[(daily.index.month >= start_month) | (daily.index.month <= end_month)]
    yr = np.where(d.index.month >= start_month, d.index.year, d.index.year - 1)
    g = d.groupby(yr)
    tot, cnt = g.sum(min_count=1), g.count()
    expected = 181
    return tot.where(cnt / expected >= min_coverage)

def dry_spells(daily: pd.Series, threshold_mm: float = 1.0, min_len: int = 7) -> list:
    out, start, n = [], None, 0
    for t, v in daily.items():
        if not np.isnan(v) and v < threshold_mm:
            start = start or t; n += 1
        else:
            if n >= min_len: out.append((start, n))
            start, n = None, 0
    if n >= min_len: out.append((start, n))
    return out

def oni_phase(v: float) -> str:
    """Oceanic Nino Index phase label. ENSO shifts odds regionally; it is not a local predictor."""
    if np.isnan(v): return "Unavailable"
    return "El Nino" if v >= 0.5 else "La Nina" if v <= -0.5 else "Neutral"
