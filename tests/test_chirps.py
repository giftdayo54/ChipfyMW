"""SYNTHETIC fixtures only: fake CHC server returns generated gz GeoTIFFs."""
import gzip, numpy as np, pandas as pd, pytest, requests, rasterio
from datetime import date, datetime, timezone
from rasterio.transform import from_origin
from src import chirps as ch, data_sources as ds, rainfall_analysis as ra

BBOX = (34.70, -16.20, 34.90, -16.00)          # 0.2 deg -> 4x4 cells at 0.05

def gz_tile(value=100.0, nodata_cells=0):
    a = np.full((40, 40), value, "float32")     # covers 34..36 E, -15..-17 S region
    if nodata_cells: a[:, :] = np.where(np.arange(40 * 40).reshape(40, 40) % 2 == 0, -9999, value) if nodata_cells == 99 else a
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=40, width=40, count=1, dtype="float32", crs="EPSG:4326",
                     transform=from_origin(34.0, -15.0, 0.05, 0.05), nodata=-9999) as d: d.write(a, 1)
        return gzip.compress(mf.read())

class Resp:
    def __init__(s, code, content=b""): s.status_code, s.content = code, content
    def raise_for_status(s):
        if s.status_code >= 400: raise requests.HTTPError(str(s.status_code))

class Server:
    def __init__(s, missing=(), fail=(), value=100.0): s.calls, s.missing, s.fail, s.value = [], missing, fail, value
    def get(s, url, timeout=0):
        s.calls.append(url)
        if any(k in url for k in s.fail): raise requests.ConnectionError("down")
        if any(k in url for k in s.missing): return Resp(404)
        return Resp(200, gz_tile(s.value))

def test_urls():
    assert ch.url_monthly(2023, 3).endswith("africa_monthly/tifs/chirps-v2.0.2023.03.tif.gz")
    assert ch.url_daily(date(2023, 3, 14)).endswith("africa_daily/tifs/p05/2023/chirps-v2.0.2023.03.14.tif.gz")

def test_window_mean_and_nodata():
    assert ch.window_mean(gz_tile(80.0), BBOX)[0] == pytest.approx(80.0)
    m, nv, nt = ch.window_mean(gz_tile(80.0, 99), BBOX); assert 0 < nv < nt and m == pytest.approx(80.0)

def test_monthly_series_cache_and_gaps(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    now = datetime(2021, 1, 15, tzinfo=timezone.utc)
    srv = Server(missing=("2020.12",), fail=("2020.06",))
    s = ch.monthly_series(BBOX, 2020, 2020, srv, retries=1, now=now)
    assert len(s) == 12 and s.isna().sum() == 2 and s.dropna().eq(100.0).all()
    assert {k for k, _ in s.attrs["failed"]} == {"2020-06", "2020-12"} and "blended" in s.attrs["kind"]
    n = len(srv.calls); ch.monthly_series(BBOX, 2020, 2020, Server(fail=("2020",)), retries=1, now=now)
    # cached months are not refetched; only the two missing ones are retried
    assert len(srv.calls) == n

def test_future_months_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    s = ch.monthly_series(BBOX, 2020, 2020, Server(), retries=1, now=datetime(2020, 4, 10, tzinfo=timezone.utc))
    assert len(s) == 3                                   # Jan-Mar only; April not yet complete

def test_low_valid_cells_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    class S2(Server):
        def get(s, url, timeout=0):
            a = np.full((40, 40), -9999, "float32")
            with rasterio.MemoryFile() as mf:
                with mf.open(driver="GTiff", height=40, width=40, count=1, dtype="float32", crs="EPSG:4326",
                             transform=from_origin(34.0, -15.0, 0.05, 0.05), nodata=-9999) as d: d.write(a, 1)
                return Resp(200, gzip.compress(mf.read()))
    s = ch.monthly_series(BBOX, 2020, 2020, S2(), retries=1, now=datetime(2020, 3, 1, tzinfo=timezone.utc))
    assert s.isna().all() and "valid cells" in s.attrs["failed"][0][1]

def test_daily_and_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    s = ch.daily_series(BBOX, date(2023, 3, 10), date(2023, 3, 14), Server(missing=("03.14",)), retries=1)
    assert len(s) == 5 and s.isna().sum() == 1
    with pytest.raises(ValueError): ch.daily_series(BBOX, date(2020, 1, 1), date(2023, 1, 1))
    with pytest.raises(ValueError): ch.monthly_series(BBOX, 1975, 1980)

def test_feeds_spi_without_filling(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    s = ch.monthly_series(BBOX, 2020, 2020, Server(), retries=1, now=datetime(2021, 1, 1, tzinfo=timezone.utc))
    assert ra.spi(s, 3).isna().all()                      # 1 year: refused, not faked
    assert ch.completeness(s)["fraction_missing"] == 0
