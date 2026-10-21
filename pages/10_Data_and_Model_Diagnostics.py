import os, pandas as pd, requests, streamlit as st
from pathlib import Path
from src import ui, registry, data_sources as ds, data_validation as dv
district, _ = ui.page("Data and Model Diagnostics")
S = st.session_state
st.subheader("Credentials")
ok = bool(os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET"))
st.write("Copernicus client credentials:", "configured" if ok else "MISSING: satellite pages will show errors")
if st.button("Check Copernicus STAC reachability"):
    try: st.write("HTTP", requests.get("https://stac.dataspace.copernicus.eu/v1", timeout=15).status_code)
    except requests.RequestException as e: st.error(f"Unreachable: {e}")
st.subheader("Freshness (this session)")
rows = []
if S.get("fc"): rows.append(("Rainfall forecast", S["fc"][0].attrs["retrieved_at"]))
for k, n in (("s1", "Sentinel-1 flood"), ("veg", "Sentinel-2 vegetation")):
    if S.get(k): rows.append((n, S[k]["meta"]["processed_at"]))
st.dataframe(pd.DataFrame(rows, columns=["product", "retrieved / processed (UTC)"])) if rows else st.info("Nothing run yet.")
st.subheader("Model performance")
sk = S.get("skill"); st.json(sk) if sk else st.info("No forecast-skill check yet (upload rainfall on the Drought page).")
log = S.get("val_log"); st.write("Flood validations this session:", len(log) if log else 0)
st.subheader("Cache"); files = list(ds.CACHE_DIR.glob("*.json")) if Path(ds.CACHE_DIR).exists() else []
st.write(f"{len(files)} cached responses, {sum(f.stat().st_size for f in files)/1e3:.0f} kB")
st.subheader("Source and methodology registry"); st.dataframe(pd.DataFrame(registry.SOURCES))
