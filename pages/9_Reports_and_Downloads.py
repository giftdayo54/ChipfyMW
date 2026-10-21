import pandas as pd, streamlit as st
from src import ui, reporting as rp, adaptation as ad, export_utils as ex
district, _ = ui.page("Reports and Downloads")
S = st.session_state; rec = S.get("rec") or ad.recommend(False, False)
fc, s1, veg = S.get("fc"), S.get("s1"), S.get("veg")
metrics = [("Rainfall trigger", fc[1]["status"] if fc else "UNAVAILABLE", "Open-Meteo forecast" if fc else "not fetched"),
           ("Satellite flood extent (ha)", f"{s1['flood_ha']:.0f}" if s1 else "UNAVAILABLE", f"S1 {s1['meta']['post']}" if s1 else "not run"),
           ("Vegetation stress (ha)", "UNAVAILABLE" if not veg or veg.get("stress_ha") is None else f"{veg['stress_ha']:.0f}", "S2" if veg else "not run")]
pe, inf_ = S.get("pop_exp"), S.get("infra")
metrics += [("People in detected flood extent (est.)", f"{pe['exposed_people']:.0f}" if pe else "UNAVAILABLE", "WorldPop-type grid, areal weighting" if pe else "no population raster"),
            ("Road km in detected flood", f"{inf_['road_km_in_flood']:.1f}" if inf_ else "UNAVAILABLE", "OSM" if inf_ else "not fetched")]
if S.get("trig_combined"): metrics.append(("Observed+forecast rainfall trigger", S["trig_combined"]["status"], "CHIRPS + Open-Meteo"))
pdf = rp.build_pdf(district, str(pd.Timestamp.today().date()), metrics, ["Copernicus Sentinel-1/2/DEM", "Open-Meteo", "CHIRPS v2.0 (CHC)", "OpenStreetMap contributors (ODbL)", "Population grid (user supplied)"],
                   ["Decision support only", "Thresholds uncalibrated", "No flood-depth estimate"], rec)
st.download_button("PDF report", pdf, f"mchip_{district}.pdf", "application/pdf")
st.download_button("Summary CSV", ex.csv_bytes(pd.DataFrame(metrics, columns=["indicator", "value", "source"])), "summary.csv")
if s1 and len(s1["polygons"]): st.download_button("Flood polygons (GeoJSON)", ex.geojson_bytes(s1["polygons"]), "flood_extent.geojson")
for k, r in (("flood", s1), ("stress", veg)):
    if r and r.get("field_table") is not None: st.download_button(f"Per-field {k} table (CSV)", ex.csv_bytes(r["field_table"]), f"fields_{k}.csv")
