"""All fixtures here are SYNTHETIC and used only for unit tests."""
import numpy as np, pandas as pd, pytest
from shapely.geometry import box
import geopandas as gpd
from src import rainfall_analysis as ra, gis_processing as gp, flood_model as fm, data_validation as dv

def synthetic_monthly(years=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", periods=years*12, freq="MS")
    return pd.Series(rng.gamma(2.0, 40.0, len(idx)), index=idx)

def test_spi_is_standard_normal_like():
    s = ra.spi(synthetic_monthly(), 3).dropna()
    assert abs(s.mean()) < 0.15 and 0.8 < s.std() < 1.2

def test_spi_refuses_short_baseline():
    assert ra.spi(synthetic_monthly(10), 3).isna().all()

def test_dry_days():
    s = pd.Series([0, 0, 5, 0, 0, 0, 2.0])
    assert ra.max_consecutive_dry_days(s) == 3

def test_area_hectares_1km_square():
    g = gpd.GeoDataFrame(geometry=[box(34.9, -16.0, 34.909, -15.991)], crs="EPSG:4326")
    assert 90 < gp.area_hectares(g) < 110   # ~1 km x 1 km ~ 100 ha

def test_area_requires_crs():
    with pytest.raises(ValueError):
        gp.area_hectares(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))

def test_susceptibility_and_missing_layer():
    r = np.random.default_rng(1)
    L = {k: r.random((20, 20)) for k in fm.DEFAULT_WEIGHTS}
    idx = fm.susceptibility(L)
    assert np.nanmin(idx) >= 0 and np.nanmax(idx) <= 1
    del L["near_river"]
    with pytest.raises(KeyError):
        fm.susceptibility(L)

def test_nan_not_low_risk():
    assert np.isnan(fm.classify(np.array([np.nan]))[0])

def test_metric_unavailable():
    assert "UNAVAILABLE" in dv.Metric("flood_ha", None).display()

def test_validate_series_rejects_negative():
    s = pd.Series([1, -1], index=pd.date_range("2020", periods=2))
    with pytest.raises(ValueError):
        dv.validate_time_series(s)
