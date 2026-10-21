"""Flood SUSCEPTIBILITY baseline (weighted overlay). Not a forecast, not calibrated."""
import numpy as np
from .gis_processing import minmax

# Provisional expert-judgement weights: MUST be validated against flood observations.
DEFAULT_WEIGHTS = {"low_elevation": 0.25, "low_slope": 0.20, "flow_accumulation": 0.20,
                   "near_river": 0.20, "soil_poor_drainage": 0.15}
LABEL = "Flood susceptibility index (relative, uncalibrated; NOT a probability or forecast)"

def susceptibility(layers: dict, weights: dict = None) -> np.ndarray:
    """layers: factor name -> array already oriented so higher = more susceptible
    is NOT assumed; supply raw arrays keyed by factor and orientation is applied here."""
    w = weights or DEFAULT_WEIGHTS
    missing = [k for k in w if k not in layers]
    if missing:
        raise KeyError(f"Missing layers: {missing}. Refusing to score with partial inputs.")
    total = sum(w.values())
    invert = {"low_elevation", "low_slope", "near_river"}   # raw value high -> less susceptible
    acc = 0
    for k, wt in w.items():
        acc = acc + (wt / total) * minmax(layers[k], invert=k in invert)
    return acc

def sensitivity(layers: dict, weights: dict = None, delta: float = 0.1) -> dict:
    """Mean absolute change in index when each weight is scaled by (1 +/- delta)."""
    base = susceptibility(layers, weights)
    w0 = weights or DEFAULT_WEIGHTS
    res = {}
    for k in w0:
        diffs = []
        for s in (1 - delta, 1 + delta):
            w = dict(w0); w[k] = w0[k] * s
            diffs.append(np.nanmean(np.abs(susceptibility(layers, w) - base)))
        res[k] = float(np.mean(diffs))
    return res

def classify(index: np.ndarray, breaks=(0.2, 0.4, 0.6, 0.8)) -> np.ndarray:
    """Relative quintile-style classes 0..4 (Very Low..Very High). Breaks are arbitrary
    until calibrated; NaN stays NaN so missing data is never shown as 'Very Low'."""
    out = np.digitize(index, breaks).astype(float)
    out[np.isnan(index)] = np.nan
    return out

# Provisional rainfall-trigger thresholds (mm). NOT calibrated: tune with local flood history.
DEFAULT_TRIGGERS = {1: 50.0, 3: 100.0}

def rainfall_trigger(daily_mm, triggers: dict = None) -> dict:
    """Checks daily rainfall against accumulation thresholds. Windows with any missing day are
    NOT scored as zero: they are skipped and counted. Rainfall only: says nothing about river level
    or catchment wetness, and 'not exceeded' is not 'safe'."""
    import numpy as np, pandas as pd
    t = triggers or DEFAULT_TRIGGERS
    x = pd.Series(np.asarray(daily_mm, float))
    if x.dropna().empty:
        return {"status": "Unavailable", "detail": "No rainfall data."}
    hits, maxima, incomplete = [], {}, 0
    for w, thr in t.items():
        if len(x) < w: continue
        acc = x.rolling(w, min_periods=w).sum().iloc[w - 1:]
        incomplete += int(acc.isna().sum())
        if acc.notna().any():
            maxima[w] = float(acc.max())
            if acc.max() >= thr: hits.append(f"{w}-day {acc.max():.0f} mm >= {thr:.0f} mm")
    gap = f" {incomplete} window(s) skipped for missing days (a hit is a lower bound)." if incomplete else ""
    if hits:
        return {"status": "Rainfall trigger exceeded", "detail": "; ".join(hits) + "." + gap, "max_accumulation_mm": maxima}
    if not maxima:
        return {"status": "Unavailable", "detail": "No complete accumulation window." + gap}
    return {"status": "Rainfall trigger not exceeded" + (" (data gaps)" if incomplete else ""), "max_accumulation_mm": maxima,
            "detail": "Rainfall only; river levels and catchment wetness not assessed." + gap}

def antecedent_totals(observed, windows=(1, 3, 7, 14, 30), today=None) -> dict:
    """Trailing observed rainfall ending at the last valid day. Incomplete windows -> None."""
    import pandas as pd
    v = observed.dropna()
    if v.empty: return {"status": "Unavailable"}
    last = v.index.max(); today = pd.Timestamp(today or pd.Timestamp.now().normalize())
    out = {"last_observed": last, "days_stale": int((today - last).days)}
    for w in windows:
        seg = observed.loc[last - pd.Timedelta(days=w - 1): last]
        out[f"{w}d_mm"] = float(seg.sum()) if len(seg) == w and seg.notna().all() else None
    return out

def combine_observed_forecast(observed, forecast, lookback: int = 7):
    """Daily table (precip_mm, source). Observed up to its last valid day, forecast after it.
    Days between them stay NaN ('missing'); they are never filled. Returns (table, info)."""
    import numpy as np, pandas as pd
    ov = observed.dropna()
    if ov.empty: raise ValueError("No observed rainfall")
    last = ov.index.max(); start = last - pd.Timedelta(days=lookback - 1)
    idx = pd.date_range(start, max(forecast.index.max(), last), freq="D")
    df = pd.DataFrame({"precip_mm": np.nan, "source": "missing"}, index=idx)
    o = observed.reindex(idx[idx <= last]); df.loc[o.index, "precip_mm"] = o.values
    df.loc[o.index[o.notna()], "source"] = "observed (CHIRPS)"
    f = forecast["precip_mm"].reindex(idx[idx > last]); f = f[f.notna()]
    df.loc[f.index, "precip_mm"] = f.values; df.loc[f.index, "source"] = "forecast"
    gap = int(((df.index > last) & (df["source"] == "missing")).sum())
    return df, {"last_observed": last, "gap_days": gap, "lookback_days": lookback}
