"""Rainfall accumulation, anomalies, dry spells and SPI (gamma, zero-inflated)."""
import numpy as np
import pandas as pd
from scipy import stats
from .config import MIN_SPI_YEARS

def accumulate(daily: pd.Series, days: int) -> pd.Series:
    return daily.rolling(days, min_periods=days).sum()

def anomaly_pct(value: float, climatology_values: np.ndarray) -> float:
    m = np.nanmean(climatology_values)
    return np.nan if m <= 0 else 100.0 * (value - m) / m

def percentile_rank(value: float, climatology_values: np.ndarray) -> float:
    c = np.asarray(climatology_values, float); c = c[~np.isnan(c)]
    return np.nan if c.size == 0 else 100.0 * (c < value).mean()

def max_consecutive_dry_days(daily: pd.Series, wet_threshold_mm: float = 1.0) -> int:
    best = run = 0
    for v in daily.fillna(np.inf).values:   # NaN breaks a run rather than extending it
        run = run + 1 if v < wet_threshold_mm else 0
        best = max(best, run)
    return best

def spi(monthly: pd.Series, scale: int, min_years: int = MIN_SPI_YEARS) -> pd.Series:
    """SPI for a monthly precipitation series. Returns all-NaN if baseline is inadequate."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    acc = monthly.rolling(scale, min_periods=scale).sum()
    out = pd.Series(np.nan, index=monthly.index, name=f"SPI{scale}")
    for m in range(1, 13):
        x = acc[acc.index.month == m].dropna()
        if len(x) < min_years:
            continue                         # insufficient data: report NaN, not a fake value
        p0 = (x == 0).mean()
        pos = x[x > 0]
        if len(pos) < 10:
            continue
        a, _, scale_g = stats.gamma.fit(pos, floc=0)
        cdf = p0 + (1 - p0) * stats.gamma.cdf(x, a, loc=0, scale=scale_g)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out.loc[x.index] = stats.norm.ppf(cdf)
    return out

def spi_category(v: float) -> str:
    if np.isnan(v): return "Unavailable"
    if v <= -2: return "Extremely dry"
    if v <= -1.5: return "Severely dry"
    if v <= -1: return "Moderately dry"
    if v < 1: return "Near normal"
    if v < 1.5: return "Moderately wet"
    if v < 2: return "Very wet"
    return "Extremely wet"
