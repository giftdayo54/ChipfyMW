"""Validation helpers. Missing data must never look like 'low risk'."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Metric:
    """A value that may be unavailable. Always carries source and timestamp."""
    name: str
    value: Optional[float]
    unit: str = ""
    source: str = ""
    valid_time: Optional[pd.Timestamp] = None
    synthetic: bool = False
    note: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None and not np.isnan(self.value)

    def display(self) -> str:
        if not self.available:
            return f"UNAVAILABLE ({self.note or 'no data'}) - not an indication of low risk"
        tag = " [SYNTHETIC TEST DATA]" if self.synthetic else ""
        return f"{self.value:.2f} {self.unit}{tag}".strip()

def validate_crs(gdf, expected: Optional[str] = None):
    if gdf.crs is None:
        raise ValueError("Layer has no CRS; refusing to guess.")
    if expected and gdf.crs.to_string() != expected:
        raise ValueError(f"CRS {gdf.crs.to_string()} != expected {expected}")
    return gdf

def validate_time_series(s: pd.Series, max_missing_frac: float = 0.1) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex.")
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise ValueError("Timestamps must be sorted and unique.")
    if (s.dropna() < 0).any():
        raise ValueError("Negative precipitation values found.")
    if s.isna().mean() > max_missing_frac:
        raise ValueError(f"{s.isna().mean():.0%} missing exceeds {max_missing_frac:.0%} limit.")
    return s

def staleness_warning(valid_time: pd.Timestamp, now: pd.Timestamp, max_age_hours: float) -> Optional[str]:
    age = (now - valid_time) / pd.Timedelta(hours=1)
    return f"Data are {age:.0f} h old (limit {max_age_hours:.0f} h)." if age > max_age_hours else None
