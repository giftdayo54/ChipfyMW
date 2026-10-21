"""SYNTHETIC fixtures only."""
import numpy as np, pytest, rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box
from src import validation as v, vegetation_pipeline as vp

def test_scores_and_excluded_not_counted():
    ref = np.zeros((10, 10), bool); ref[:5, :5] = True
    pred = np.zeros((10, 10), bool); pred[:5, :4] = True; pred[8, 8] = True
    r = v.score_masks(pred, ref)
    assert r["tp"] == 20 and r["fn"] == 5 and r["fp"] == 1
    assert r["recall"] == pytest.approx(0.8) and r["iou"] == pytest.approx(20 / 26)
    ex = np.zeros((10, 10), bool); ex[:5, 4] = True        # missed column excluded -> unscored
    assert v.score_masks(pred, ref, ex)["fn"] == 0

def test_no_reference_flood():
    assert v.score_masks(np.ones((3, 3), bool), np.zeros((3, 3), bool))["status"].startswith("no_reference")

def test_rasterize_and_crs_required():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 0.5)], crs="EPSG:4326")
    m = v.rasterize_reference(g, (0, 0, 1, 1), (10, 10))
    assert m.sum() == 25

BBOX = (34.80, -16.10, 34.84, -16.06)
def tiff7(red, nir, scl, mask):
    h, w = red.shape
    b = np.stack([np.full((h, w), .05), np.full((h, w), .08), red, nir, np.full((h, w), .2), scl, mask]).astype("float32")
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=w, count=7, dtype="float32", crs="EPSG:4326",
                     transform=from_origin(0, 0, 1, 1)) as d: d.write(b)
        return mf.read()

def fetch_factory(cloudy=False):
    def fetch(payload):
        o = payload["output"]; h, w = o["height"], o["width"]
        year = int(payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:4])
        r = np.random.default_rng(year)
        nir = 0.5 + r.normal(0, 0.03, (h, w)); red = np.full((h, w), 0.08)
        if year == 2026: nir[:20, :20] = 0.25          # stressed block in current year
        scl = np.full((h, w), 9.0 if cloudy else 4.0)
        return tiff7(red, nir, scl, np.ones((h, w)))
    return fetch

def test_stress_detected():
    r = vp.run_s2_condition(fetch_factory(), BBOX, ("2026-09-01T00:00:00Z", "2026-09-30T00:00:00Z"), 6)
    assert r["status"] == "ok" and r["baseline_used"] == 6
    assert r["vci"][5, 5] < 20 and r["vci"][-5, -5] > 20 and r["stress_ha"] > 0 and "not a drought" in r["caveat"]

def test_cloudy_refused():
    r = vp.run_s2_condition(fetch_factory(cloudy=True), BBOX, ("2026-09-01T00:00:00Z", "2026-09-30T00:00:00Z"))
    assert r["status"] == "unusable_cloud" and "absence" in r["message"]

def test_too_few_baseline_years():
    r = vp.run_s2_condition(fetch_factory(), BBOX, ("2026-09-01T00:00:00Z", "2026-09-30T00:00:00Z"), 3)
    assert r["vci"] is None and "<5" in r["message"]

def test_shift_year():
    assert vp.shift_year("2026-09-01T00:00:00Z", 2) == "2024-09-01T00:00:00Z"
