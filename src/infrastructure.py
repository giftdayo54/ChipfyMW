"""OpenStreetMap infrastructure (hospitals/clinics, schools, roads, bridges) via Overpass, and
exposure to a flood footprint. Data (c) OpenStreetMap contributors, ODbL: attribution required.
OSM completeness in Malawi varies: an asset missing from OSM is not an asset that does not exist."""
import json, time
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from . import data_sources as ds
from .config import AREA_CRS

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "M-CHIP/0.1 (decision-support research tool)"}

def build_query(bbox) -> str:
    w, s, e, n = ds.validate_bbox(bbox); b = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:60];(node["amenity"~"^(hospital|clinic|doctors|school|college)$"]{b};'
            f'way["amenity"~"^(hospital|clinic|doctors|school|college)$"]{b};node["healthcare"]{b};);out center tags;'
            f'(way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]{b};);out geom tags;')

def _kind(tags: dict) -> str:
    a = tags.get("amenity", "")
    if a in ("hospital", "clinic", "doctors") or "healthcare" in tags: return "health"
    if a in ("school", "college"): return "school"
    if "highway" in tags: return "bridge" if tags.get("bridge") not in (None, "no") else "road"
    return "other"

def parse_overpass(js: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in js.get("elements", []):
        tags = el.get("tags", {}); geom = None
        if el["type"] == "node" and "lat" in el: geom = Point(el["lon"], el["lat"])
        elif "geometry" in el and len(el["geometry"]) >= 2: geom = LineString([(p["lon"], p["lat"]) for p in el["geometry"]])
        elif "center" in el: geom = Point(el["center"]["lon"], el["center"]["lat"])
        if geom is None: continue
        rows.append({"kind": _kind(tags), "name": tags.get("name", ""), "osm_id": f'{el["type"]}/{el["id"]}', "geometry": geom})
    return gpd.GeoDataFrame(rows, columns=["kind", "name", "osm_id", "geometry"], geometry="geometry", crs="EPSG:4326")

def fetch_osm(bbox, session=None, retries: int = 3, ttl_s: float = 7 * 86400) -> gpd.GeoDataFrame:
    q = build_query(bbox); hit = ds.cached_json("overpass:" + q, ttl_s)
    if hit is None:
        last = None
        for i in range(retries):
            try:
                r = (session or requests).post(OVERPASS_URL, data={"data": q}, headers=HEADERS, timeout=90)
                r.raise_for_status(); hit = r.json(); break
            except (requests.RequestException, ValueError) as e:
                last = e; time.sleep(min(2 ** i, 8) if retries > 1 else 0)
        if hit is None: raise ds.DataUnavailable(f"Overpass failed: {last}")
        ds.store_json("overpass:" + q, hit)
    return parse_overpass(hit)

def sample_mask(gdf, mask, bbox):
    """Value of a bool raster (north-up, covering bbox) at each feature's representative point."""
    w, s, e, n = bbox; ny, nx = mask.shape
    pts = gdf.geometry.representative_point()
    cols = ((pts.x - w) / (e - w) * nx).astype(int).clip(0, nx - 1)
    rows = ((n - pts.y) / (n - s) * ny).astype(int).clip(0, ny - 1)
    return mask[rows.values, cols.values]

def exposure_summary(assets: gpd.GeoDataFrame, flood_polys: gpd.GeoDataFrame, excluded_mask, bbox, near_m: float = 100.0) -> dict:
    a = assets.to_crs(AREA_CRS)
    union = flood_polys.to_crs(AREA_CRS).geometry.union_all() if len(flood_polys) else None
    if union is None:
        inside = pd.Series(False, index=a.index); near = inside.copy()
    else:
        inside = a.geometry.intersects(union)
        near = ~inside & (a.geometry.distance(union) <= near_m)
    unk = pd.Series(sample_mask(assets, excluded_mask, bbox), index=a.index) & ~inside
    status = pd.Series("outside detected extent (NOT confirmed safe)", index=a.index)
    status[unk] = "in area satellite could not assess"
    status[near] = f"within {near_m:.0f} m of detected flood"
    status[inside] = "inside detected flood extent"
    table = pd.DataFrame({"kind": a["kind"].values, "name": a["name"].values, "osm_id": a["osm_id"].values, "status": status.values})
    roads = a[a["kind"].isin(["road", "bridge"]) & a.geometry.geom_type.eq("LineString")]
    km_in = (roads.geometry.intersection(union).length.sum() / 1000.0) if union is not None and len(roads) else 0.0
    return {"table": table, "road_km_total": float(roads.length.sum() / 1000.0), "road_km_in_flood": float(km_in),
            "counts": table.groupby(["kind", "status"]).size().rename("n").reset_index(),
            "caveat": f"Flood extent is 20-100 m pixel based; 'within {near_m:.0f} m' flags positional uncertainty. "
                      "OSM is incomplete in places; missing assets are not reported."}
