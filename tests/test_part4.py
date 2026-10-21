"""Synthetic fixtures only (unit tests)."""
import numpy as np, pytest, requests
from src import (flood_model as fm, gis_processing as gp, exposure_analysis as ea, risk_assessment as ra,
                 adaptation as ad, export_utils as ex, reporting as rp, weather_sources as ws)

def test_trigger_states():
    assert fm.rainfall_trigger([5, 60, 3])["status"] == "Rainfall trigger exceeded"
    r = fm.rainfall_trigger([5, 10, 3]); assert "not exceeded" in r["status"] and "not assessed" in r["detail"]
    assert fm.rainfall_trigger([np.nan])["status"] == "Unavailable"

def test_cell_area_ha():
    a = gp.cell_area_ha_grid(34.9, -15.9, 0.0001, 0.0001, (10, 10))    # ~11 m cells
    assert abs(a[0, 0] - (0.0001 * 111_195 * 0.0001 * 111_195 * np.cos(np.radians(15.9))) / 1e4) < 2e-4

def test_exposure():
    area = np.full((4, 4), 1.0); flood = np.zeros((4, 4), bool); flood[:2] = True
    crop = np.zeros((4, 4), bool); crop[:, :2] = True
    assert ea.exposed_hectares(flood, area) == 8 and ea.exposed_hectares(flood, area, crop) == 4
    with pytest.raises(ValueError): ea.exposed_hectares(flood, np.ones((3, 3)))
    assert ea.population_exposed(flood, np.full((4, 4), 2.0))["people"] == 16

def test_risk_incomplete_has_no_composite():
    r = ra.risk_components(0.8, None, 0.5)
    assert r["priority_index"] is None and "Not low risk" in r["note"]
    assert ra.risk_components(1, 1, 1)["priority_index"] == 1
    with pytest.raises(ValueError): ra.risk_components(2, 0, 0)

def test_recommend_and_notice():
    r = ad.recommend(True, True, sugarcane=True)
    assert {k for k, _ in r["items"]} >= {"Flood", "Drought", "Sugarcane/irrigation", "Compound"}
    assert ad.recommend(False, False)["items"] == []

def test_geotiff_roundtrip(tmp_path):
    import rasterio
    b = ex.geotiff_bytes(np.arange(6, dtype=float).reshape(2, 3), 34.0, -15.0, 0.1, 0.1, {"src": "synthetic"})
    p = tmp_path / "t.tif"; p.write_bytes(b)
    with rasterio.open(p) as s:
        assert s.crs.to_string() == "EPSG:4326" and s.tags()["src"] == "synthetic"
    arr, tr, crs = gp.read_raster_window(str(p), (34.0, -15.2, 34.2, -15.0))
    assert arr.shape == (2, 2)

def test_pdf():
    pdf = rp.build_pdf("Chikwawa", "2026-10-19", [("SPI-3", "UNAVAILABLE", "-")], ["Copernicus"], ["Test"],
                       ad.recommend(True, False))
    assert pdf[:4] == b"%PDF"

def test_weather_parse_and_failure():
    js = {"daily": {"time": ["2026-10-20", "2026-10-21"], "precipitation_sum": [1.0, 40.0]}}
    df = ws.parse_daily(js, "forecast", "x"); assert df.attrs["kind"] == "forecast" and df.precip_mm.sum() == 41
    with pytest.raises(ValueError): ws.parse_daily({}, "forecast", "x")
    class S:
        def get(self, *a, **k): raise requests.ConnectionError("down")
    with pytest.raises(ConnectionError): ws._get("u", {}, S(), retries=1)
