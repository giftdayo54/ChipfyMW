"""SYNTHETIC fixtures only."""
import numpy as np, pytest, geopandas as gpd
from shapely.geometry import box
from src import field_mask as fm, vegetation_pipeline as vp, flood_pipeline as fp
from tests.test_part6 import fetch_factory, BBOX as VBBOX
from tests.test_flood_pipeline import make_fetch, BBOX as FBBOX

def fields_gdf(bbox):
    w, s, e, n = bbox; mx, my = (w + e) / 2, (s + n) / 2
    return gpd.GeoDataFrame({"field_name": ["A", "B"]}, geometry=[box(w, my, mx, n), box(mx, s, e, my)], crs="EPSG:4326")

def test_load_requires_crs_and_names(tmp_path):
    g = fields_gdf(VBBOX); p = tmp_path / "f.geojson"; g.to_file(p, driver="GeoJSON")
    g = g.rename(columns={"field_name": "name"}); g.to_file(p, driver="GeoJSON")
    out = fm.load_fields(str(p), name_col="name"); assert list(out.field_name) == ["A", "B"]
    q = tmp_path / "nocrs.gpkg"; gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]).to_file(q, driver="GPKG")
    with pytest.raises(ValueError): fm.load_fields(str(q))

def test_zonal_summary_ha():
    f = fields_gdf((0, 0, 0.01, 0.01)); ids = fm.field_id_raster(f, (0, 0, 0.01, 0.01), (10, 10))
    area = np.ones((10, 10)); m = np.ones((10, 10), bool)
    t = fm.zonal_summary(f, ids, area, {"x_ha": m})
    assert t.covered_ha.sum() == 50 and (t.field_ha > 0).all()

def test_vegetation_stress_only_inside_fields():
    # field A = NW quadrant contains the synthetic stressed block (top-left 20x20 px)
    f = fields_gdf(VBBOX)
    r = vp.run_s2_condition(fetch_factory(), VBBOX, ("2026-09-01T00:00:00Z", "2026-09-30T00:00:00Z"), 6, fields=f)
    t = r["field_table"]; assert t is not None and t.stressed_ha[0] > t.stressed_ha[1]
    assert r["stress_ha_in_fields"] <= r["stress_ha"] + 1e-6

def test_flood_field_table_reports_unknown():
    f = fields_gdf(FBBOX)
    r = fp.run_s1_flood(make_fetch(None), FBBOX, ("2026-02-01T00:00:00Z", "2026-02-10T00:00:00Z"),
                        ("2026-03-14T00:00:00Z", "2026-03-20T00:00:00Z"), fields=f)
    assert {"flood_ha", "unknown_ha"} <= set(r["field_table"].columns)
