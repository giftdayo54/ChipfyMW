"""Synthetic fixtures only (unit tests)."""
import numpy as np, pytest
from src import satellite_processing as sp, data_sources as ds

def scene(seed=0, n=60):
    r = np.random.default_rng(seed)
    pre = r.gamma(20, 0.01, (n, n)); post = pre.copy()   # speckled land ~ -7 dB
    pre[:, :10] = 0.005; post[:, :10] = 0.005             # permanent river (-23 dB)
    post[20:40, 20:40] = 0.004                             # new flood block
    return pre, post

def test_flood_detected_permanent_water_separated():
    pre, post = scene()
    res = sp.detect_flood_s1(pre, post)
    assert res.flood[25:35, 25:35].all() and not res.flood[:, :10].any()
    assert res.permanent_water[:, :5].all() and res.warnings

def test_terrain_mask_marks_unknown_not_dry():
    pre, post = scene()
    slope = np.zeros(pre.shape); slope[20:40, 20:40] = 20
    res = sp.detect_flood_s1(pre, post, slope_deg=slope)
    assert not res.flood[25:35, 25:35].any() and res.excluded[25:35, 25:35].all()

def test_shape_mismatch():
    with pytest.raises(ValueError): sp.detect_flood_s1(np.ones((3, 3)), np.ones((4, 4)))

def test_hectares():
    assert sp.flooded_hectares(np.ones((100, 100), bool), 100.0) == pytest.approx(100.0)

def test_s2_cloud_mask_and_indices():
    z = np.full((4, 4), 0.1); nir = np.full((4, 4), 0.5)
    scl = np.full((4, 4), 4); scl[0, 0] = 9
    r = sp.s2_indices(z, z, z, nir, z, scl)
    assert np.isnan(r["NDVI"][0, 0]) and r["NDVI"][1, 1] == pytest.approx(0.4 / 0.6)
    assert r["valid_fraction"] == pytest.approx(15 / 16)

def test_optical_refused_when_cloudy():
    ok, msg = sp.optical_usable(0.3)
    assert not ok and "NOT" not in msg and "absence" in msg

def test_vci_needs_baseline():
    base = np.random.default_rng(0).random((3, 5, 5))
    assert np.isnan(sp.vegetation_anomaly(np.full((5, 5), .5), base)["VCI"]).all()

def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    with pytest.raises(ds.CredentialsMissing): ds.CDSEClient().token()

def test_bbox_guard_and_payload():
    with pytest.raises(ValueError): ds.validate_bbox((30, -20, 40, -9))     # whole-country: refused
    p = ds.process_payload("s1", (34.7, -16.3, 35.0, -16.0), "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 512, 512)
    assert p["input"]["data"][0]["type"] == "sentinel-1-grd"

def test_stac_cache_and_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    calls = {"n": 0}
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"features": [{"id": "a", "properties": {"datetime": "2026-01-02", "eo:cloud_cover": 5}}]}
    class S:
        def request(self, *a, **k): calls["n"] += 1; return R()
    c = ds.CDSEClient(session=S())
    a = c.stac_search("sentinel-2-l2a", (34.7, -16.3, 35.0, -16.0), "2026-01-01", "2026-01-31")
    b = c.stac_search("sentinel-2-l2a", (34.7, -16.3, 35.0, -16.0), "2026-01-01", "2026-01-31")
    assert a == b and calls["n"] == 1 and ds.summarise_items(a)[0]["cloud_cover"] == 5

def test_api_failure_is_explicit(monkeypatch):
    class S:
        def request(self, *a, **k): raise ds.requests.ConnectionError("down")
    with pytest.raises(ds.DataUnavailable):
        ds.CDSEClient(session=S(), retries=1).stac_search("sentinel-1-grd", (34.7, -16.3, 35.0, -16.0), "2026-01-01", "2026-01-31")
