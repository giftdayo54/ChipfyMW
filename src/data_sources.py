"""Copernicus Data Space Ecosystem (CDSE) client: auth, STAC search, Process API.

Credentials come ONLY from env vars CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (OAuth client
created in the CDSE dashboard). Without them the client raises CredentialsMissing and
the UI must show an 'unavailable' state; nothing is ever substituted.
NOTE: endpoints follow CDSE docs but were not exercised live in development.
"""
import hashlib, json, logging, os, time
from pathlib import Path
from typing import Optional
import requests

log = logging.getLogger("mchip.data")
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CACHE_DIR = Path(os.environ.get("MCHIP_CACHE", ".cache"))

class CredentialsMissing(RuntimeError): ...
class DataUnavailable(RuntimeError): ...

def _cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")

def cached_json(key: str, ttl_s: float):
    p = _cache_path(key)
    if p.exists() and time.time() - p.stat().st_mtime < ttl_s:
        return json.loads(p.read_text())
    return None

def store_json(key: str, obj) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(obj))

def validate_bbox(bbox) -> tuple:
    w, s, e, n = map(float, bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"Invalid bbox {bbox}")
    if (e - w) * (n - s) > 4.0:      # guard: force district-level selection before heavy queries
        raise ValueError("bbox too large; select a district or catchment first")
    return w, s, e, n

class CDSEClient:
    def __init__(self, session: Optional[requests.Session] = None, retries: int = 3):
        self.s = session or requests.Session()
        self.retries = retries
        self._token, self._exp = None, 0.0

    def _request(self, method, url, **kw):
        last = None
        for i in range(self.retries):
            try:
                r = self.s.request(method, url, timeout=60, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                last = e
                log.warning("request failed (%s/%s) %s: %s", i + 1, self.retries, url, e)
                time.sleep(min(2 ** i, 8) if self.retries > 1 else 0)
        raise DataUnavailable(f"{url} failed after {self.retries} tries: {last}")

    def token(self) -> str:
        if self._token and time.time() < self._exp - 30:
            return self._token
        cid, sec = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
        if not (cid and sec):
            raise CredentialsMissing("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (see .env.example).")
        r = self._request("POST", TOKEN_URL, data={"grant_type": "client_credentials",
                          "client_id": cid, "client_secret": sec}).json()
        self._token, self._exp = r["access_token"], time.time() + r.get("expires_in", 600)
        return self._token

    def stac_search(self, collection: str, bbox, start: str, end: str,
                    max_cloud: Optional[float] = None, limit: int = 50, ttl_s: float = 3600) -> list:
        """STAC search (public; no token needed). Returns list of item dicts."""
        body = {"collections": [collection], "bbox": list(validate_bbox(bbox)),
                "datetime": f"{start}/{end}", "limit": limit}
        if max_cloud is not None:
            body["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
        key = json.dumps(body, sort_keys=True)
        hit = cached_json(key, ttl_s)
        if hit is not None:
            return hit
        items = self._request("POST", STAC_URL, json=body).json().get("features", [])
        store_json(key, items)
        return items

    def process(self, payload: dict) -> bytes:
        """Sentinel Hub Process API (needs token). Returns raw bytes (e.g. GeoTIFF)."""
        hdr = {"Authorization": f"Bearer {self.token()}"}
        return self._request("POST", PROCESS_URL, json=payload, headers=hdr).content

def summarise_items(items: list) -> list:
    """Item id, acquisition time, cloud cover: used for the freshness/diagnostics page."""
    out = []
    for it in items:
        p = it.get("properties", {})
        out.append({"id": it.get("id"), "datetime": p.get("datetime"),
                    "cloud_cover": p.get("eo:cloud_cover"),
                    "orbit": p.get("sat:orbit_state")})
    return sorted(out, key=lambda d: d["datetime"] or "")

S1_EVALSCRIPT = """//VERSION=3
function setup(){return{input:["VV","VH","dataMask"],output:{bands:3,sampleType:"FLOAT32"}}}
function evaluatePixel(s){return[s.VV,s.VH,s.dataMask]}"""
DEM_EVALSCRIPT = """//VERSION=3
function setup(){return{input:["DEM"],output:{bands:1,sampleType:"FLOAT32"}}}
function evaluatePixel(s){return[s.DEM]}"""
S2_EVALSCRIPT = """//VERSION=3
function setup(){return{input:["B02","B03","B04","B08","B11","SCL","dataMask"],output:{bands:7,sampleType:"FLOAT32"}}}
function evaluatePixel(s){return[s.B02,s.B03,s.B04,s.B08,s.B11,s.SCL,s.dataMask]}"""

def process_payload(kind: str, bbox, start: str, end: str, width: int, height: int,
                    mosaicking: str = "mostRecent") -> dict:
    """Build a Process API request. kind: 's1' (GRD, linear power) or 's2' (L2A)."""
    if kind not in ("s1", "s2", "dem"):
        raise ValueError("kind must be 's1', 's2' or 'dem'")
    if not (0 < width <= 2500 and 0 < height <= 2500):
        raise ValueError("width/height must be in 1..2500")
    dtype = {"s1": "sentinel-1-grd", "s2": "sentinel-2-l2a", "dem": "dem"}[kind]
    data = {"type": dtype, "dataFilter": {"timeRange": {"from": start, "to": end},
            "mosaickingOrder": mosaicking}}
    if kind == "dem":
        data = {"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_30"}}
    if kind == "s1":
        data["dataFilter"].update({"acquisitionMode": "IW", "polarization": "DV"})
        data["processing"] = {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True,
                              "demInstance": "COPERNICUS_30"}
    return {"input": {"bounds": {"bbox": list(validate_bbox(bbox)),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                      "data": [data]},
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": {"s1": S1_EVALSCRIPT, "s2": S2_EVALSCRIPT, "dem": DEM_EVALSCRIPT}[kind]}
