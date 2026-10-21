"""SYNTHETIC end-to-end test: fake Process API returns generated GeoTIFFs."""
import numpy as np, pytest, rasterio
from rasterio.transform import from_origin
from src import flood_pipeline as fp, data_sources as ds

BBOX = (34.80, -16.10, 34.84, -16.06)

def tiff(bands):
    b = np.stack(bands).astype("float32")
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=b.shape[1], width=b.shape[2], count=b.shape[0], dtype="float32",
                     crs="EPSG:4326", transform=from_origin(0, 0, 1, 1)) as d: d.write(b)
        return mf.read()

def make_fetch(width_height_holder):
    def fetch(payload):
        o = payload["output"]; h, w = o["height"], o["width"]
        r = np.random.default_rng(0)
        if payload["input"]["data"][0]["type"] == "dem":
            yy = np.linspace(0, 3, h)[:, None] * np.ones((1, w))   # gentle ramp, low terrain
            return tiff([yy])
        pre = r.gamma(20, 0.01, (h, w)); pre[:, :8] = 0.005
        post = pre.copy()
        if payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"].startswith("2026-03"):
            post[h // 2:h // 2 + 30, w // 2:w // 2 + 30] = 0.004
        return tiff([post if "2026-03" in payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"] else pre,
                     pre * 0.5, np.ones((h, w))])
    return fetch

def test_pipeline_end_to_end():
    out = fp.run_s1_flood(make_fetch(None), BBOX, ("2026-02-01T00:00:00Z", "2026-02-10T00:00:00Z"),
                          ("2026-03-14T00:00:00Z", "2026-03-20T00:00:00Z"), res_m=30)
    # 30x30 px block at ~30 m => ~81 ha
    assert 60 < out["flood_ha"] < 100 and out["permanent_water_ha"] > 0
    assert len(out["polygons"]) >= 1 and out["polygons"].crs.to_string() == "EPSG:4326"
    assert any("not confirmed dry" in w.lower() for w in out["warnings"])
    poly_ha = out["polygons"].to_crs("EPSG:32736").area.sum() / 1e4
    assert poly_ha == pytest.approx(out["flood_ha"], rel=0.05)

def test_oversize_window_refused():
    with pytest.raises(ValueError): fp.run_s1_flood(lambda p: b"", (34.5, -16.5, 35.5, -15.5), ("a", "b"), ("c", "d"), res_m=10)

def test_dem_payload():
    p = ds.process_payload("dem", BBOX, "", "", 100, 100)
    assert p["input"]["data"][0]["dataFilter"]["demInstance"] == "COPERNICUS_30"
