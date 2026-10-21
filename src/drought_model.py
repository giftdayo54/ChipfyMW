"""Drought classification (multi-indicator) and a chronologically validated forecast baseline."""
from typing import Optional
import numpy as np
import pandas as pd

SEVERITY = {0: "None detected", 1: "Moderate", 2: "Severe", 3: "Extreme"}

def _sev_spi(v): return 3 if v <= -2 else 2 if v <= -1.5 else 1 if v <= -1 else 0
def _sev_vci(v): return 3 if v < 10 else 2 if v < 20 else 1 if v < 35 else 0
def _sev_z(v): return 3 if v <= -2 else 2 if v <= -1.5 else 1 if v <= -1 else 0

def classify_drought(spi: Optional[float] = None, vci: Optional[float] = None,
                     soil_moisture_z: Optional[float] = None) -> dict:
    """Worst severity among AVAILABLE indicators. No indicators -> 'Unavailable' (never 'None')."""
    parts = {}
    for name, v, f in (("SPI", spi, _sev_spi), ("VCI", vci, _sev_vci), ("SM_z", soil_moisture_z, _sev_z)):
        if v is not None and not np.isnan(v):
            parts[name] = f(v)
    if not parts:
        return {"category": "Unavailable", "n_indicators": 0, "components": {},
                "note": "No indicator available; this is NOT a low-risk result."}
    sev = max(parts.values())
    note = "Single indicator only: lower confidence." if len(parts) == 1 else ""
    if "VCI" in parts and len(parts) == 1:
        note += " NDVI-based stress can also come from harvest, fire, pests or land-use change."
    return {"category": SEVERITY[sev], "n_indicators": len(parts), "components": parts, "note": note.strip()}

def climatology_forecast(train: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    clim = train.groupby(train.index.month).mean()
    return pd.Series(clim.reindex(target_index.month).values, index=target_index)

def build_features(s: pd.Series, lags=(1, 2, 3)) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for l in lags:
        df[f"lag{l}"] = s.shift(l)
    m = s.index.month
    df["sin_m"], df["cos_m"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return df.dropna()

def evaluate_chronological(s: pd.Series, test_frac: float = 0.25, seed: int = 0) -> dict:
    """Train on the earliest part, test on the latest (no shuffling). Compares a random
    forest with seasonal climatology; reports skill = 1 - MAE_model/MAE_climatology."""
    from sklearn.ensemble import RandomForestRegressor
    df = build_features(s)
    if len(df) < 120:
        return {"status": "insufficient_data", "note": "Need >= 10 years of monthly data."}
    cut = int(len(df) * (1 - test_frac))
    tr, te = df.iloc[:cut], df.iloc[cut:]
    assert tr.index.max() < te.index.min()
    X = [c for c in df.columns if c != "y"]
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed).fit(tr[X], tr["y"])
    mae_rf = float(np.mean(np.abs(rf.predict(te[X]) - te["y"])))
    mae_cl = float(np.mean(np.abs(climatology_forecast(tr["y"], te.index) - te["y"])))
    skill = 1 - mae_rf / mae_cl
    return {"status": "ok", "train_end": str(tr.index.max().date()), "test_start": str(te.index.min().date()),
            "mae_model": mae_rf, "mae_climatology": mae_cl, "skill_vs_climatology": float(skill),
            "verdict": ("Model beats climatology on held-out period." if skill > 0.05 else
                        "No demonstrated skill over climatology: do NOT use as a forecast.")}
