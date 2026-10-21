"""SYNTHETIC fixtures only."""
import numpy as np, pandas as pd, pytest, requests, geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin
from src import flood_model as fm, exposure_analysis as ea, infrastructure as inf, data_sources as ds

def test_trigger_does_not_treat_gaps_as_zero():
    r = fm.rainfall_trigger([np.nan, 40, 40, 40])               # 3-day window 120 complete
    assert r["status"] == "Rainfall trigger exceeded"
    r = fm.rainfall_trigger([30, np.nan, 5, 5, 5])
    assert "data gaps" in r["status"] and "skipped" in r["detail"]
    assert fm.rainfall_trigger([np.nan, 5, np.nan, 5])["status"] == "Rainfall trigger not exceeded (data gaps)"
    assert fm.rainfall_trigger([np.nan, 5, np.nan])["status"] == "Rainfall trigger not exceeded (data gaps)"

def test_antecedent_and_combine_keep_gap():
    idx = pd.date_range("2026-10-01", "2026-10-15"); obs = pd.Series(10.0, index=idx); obs["2026-10-10"] = np.nan
    a = fm.antecedent_totals(obs, today="2026-10-20")
    assert a["last_observed"] == pd.Timestamp("2026-10-15") and a["days_stale"] == 5
    assert a["1d_mm"] == 10 and a["3d_mm"] == 30 and a["7d_mm"] is None   # 7-day window contains the missing day
    fcidx = pd.date_range("2026-10-18", periods=3); fc = pd.DataFrame({"precip_mm": [60., 1., 1.]}, index=fcidx)
    df, info = fm.combine_observed_forecast(obs, fc, lookback=5)
    assert info["gap_days"] == 2 and df.loc["2026-10-16", "source"] == "missing" and np.isnan(df.loc["2026-10-16", "precip_mm"])
    assert df.loc["2026-10-18", "source"] == "forecast" and df.loc["2026-10-15", "source"].startswith("observed")

BBOX = (34.80, -16.10, 34.84, -16.06)
def test_population_exposure_areal():
    pop = np.full((4, 4), 100.0); tr = from_origin(34.80, -16.06, 0.01, 0.01)
    flood = np.zeros((40, 40), bool); flood[:20, :20] = True
    exc = np.zeros((40, 40), bool); exc[30:, 30:] = True
    r = ea.population_exposure(pop, tr, flood, exc, BBOX)
    assert r["total_people"] == 1600 and r["exposed_people"] == pytest.approx(400, abs=1) and r["unassessed_people"] == pytest.approx(100, abs=1)
    pop[0, 0] = np.nan; assert ea.population_exposure(pop, tr, flood, exc, BBOX)["nodata_cells"] == 1

OV = {"elements": [
    {"type": "node", "id": 1, "lat": -16.07, "lon": 34.81, "tags": {"amenity": "clinic", "name": "In Clinic"}},
    {"type": "node", "id": 2, "lat": -16.0705, "lon": 34.8160, "tags": {"amenity": "school", "name": "Near School"}},
    {"type": "node", "id": 3, "lat": -16.095, "lon": 34.835, "tags": {"amenity": "hospital", "name": "Unknown Hosp"}},
    {"type": "node", "id": 4, "lat": -16.06, "lon": 34.83, "tags": {"amenity": "school", "name": "Far School"}},
    {"type": "way", "id": 5, "center": {"lat": -16.0701, "lon": 34.805}, "tags": {"amenity": "hospital", "name": "Way Hosp"}},
    {"type": "way", "id": 6, "tags": {"highway": "primary", "name": "M1"}, "geometry": [{"lat": -16.07, "lon": 34.80}, {"lat": -16.07, "lon": 34.84}]},
    {"type": "way", "id": 7, "tags": {"highway": "secondary", "bridge": "yes"}, "geometry": [{"lat": -16.08, "lon": 34.81}, {"lat": -16.08, "lon": 34.812}]}]}

def test_parse_and_exposure():
    g = inf.parse_overpass(OV); assert set(g.kind) == {"health", "school", "road", "bridge"}
    flood = gpd.GeoDataFrame(geometry=[box(34.80, -16.075, 34.815, -16.065)], crs="EPSG:4326")
    exc = np.zeros((40, 40), bool); exc[30:, 30:] = True
    r = inf.exposure_summary(g, flood, exc, BBOX, near_m=150)
    st = dict(zip(r["table"]["name"].replace("", "road"), r["table"]["status"]))
    assert st["In Clinic"] == "inside detected flood extent" and st["Way Hosp"] == "inside detected flood extent"
    assert st["Near School"].startswith("within") and st["Unknown Hosp"].startswith("in area satellite")
    assert "NOT confirmed safe" in st["Far School"]
    assert r["road_km_total"] > 3.5 and 1.4 < r["road_km_in_flood"] < 1.8   # ~1.65 km of M1 inside 0.015 deg

def test_no_flood_polygons():
    g = inf.parse_overpass(OV)
    r = inf.exposure_summary(g, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), np.zeros((10, 10), bool), BBOX)
    assert r["road_km_in_flood"] == 0 and not r["table"]["status"].str.startswith("inside").any()

def test_overpass_failure_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    class Bad:
        def post(self, *a, **k): raise requests.ConnectionError("down")
    with pytest.raises(ds.DataUnavailable): inf.fetch_osm(BBOX, Bad(), retries=1)
    n = {"c": 0}
    class R:
        def raise_for_status(self): pass
        def json(self): return OV
    class Good:
        def post(self, *a, **k): n["c"] += 1; return R()
    inf.fetch_osm(BBOX, Good()); inf.fetch_osm(BBOX, Good()); assert n["c"] == 1
    with pytest.raises(ValueError): inf.build_query((30, -20, 40, -9))
