import folium, pandas as pd, streamlit as st
from streamlit_folium import st_folium
from src import ui, data_sources as ds, weather_sources as ws, flood_model as fm, flood_pipeline as fp
from src import visualisation as vis, validation as val, export_utils as ex, chirps
import plotly.express as px
district, scale = ui.page("Flood Intelligence")
S = st.session_state
t1, t2, t3 = st.tabs(["Rainfall trigger (forecast)", "Sentinel-1 flood extent", "Susceptibility"])
with t1:
    st.write("7-day model rainfall (Open-Meteo, centroid grid cell). Provisional thresholds, uncalibrated.")
    if st.button("Fetch forecast"):
        try:
            lat, lon = vis.DISTRICT_CENTROIDS[district]
            df = ws.fetch_forecast(lat, lon, 7); S["fc"] = (df, fm.rainfall_trigger(df["precip_mm"].values))
        except (ConnectionError, ValueError) as e: st.error(f"Forecast unavailable: {e}")
    if "fc" in S:
        df, trig = S["fc"]; st.metric("Rainfall trigger", trig["status"]); st.caption(trig["detail"])
        st.bar_chart(df["precip_mm"]); st.caption(f"{df.attrs['source']} | retrieved {df.attrs['retrieved_at']} | {df.attrs['note']}")
        st.caption("Not a flood forecast or warning: no river level, soil wetness or upstream rain.")
with t1:
    st.markdown("---"); st.subheader("Observed rainfall (CHIRPS daily) with forecast")
    st.caption("CHIRPS is a satellite-gauge blended estimate with a publication lag of a few days. Missing days stay missing; they are never filled or treated as zero.")
    days = st.slider("Observed days to fetch", 7, 30, 14)
    if st.button("Fetch CHIRPS observed"):
        try:
            end = pd.Timestamp.now().normalize(); bar = st.progress(0.0)
            S["obs"] = chirps.daily_series(vis.district_bbox(district), (end - pd.Timedelta(days=days)).date(), end.date(),
                                           progress=lambda n, tot: bar.progress(n / tot))
        except (ds.DataUnavailable, ValueError) as e: st.error(f"CHIRPS unavailable: {e}")
    obs = S.get("obs")
    if obs is not None:
        if obs.dropna().empty:
            st.warning(f"No CHIRPS days available ({len(obs.attrs['failed'])} failed or unpublished). Observed antecedent rainfall is UNAVAILABLE.")
        else:
            a = fm.antecedent_totals(obs); c = st.columns(5)
            for col, w in zip(c, (1, 3, 7, 14, 30)):
                v = a.get(f"{w}d_mm"); col.metric(f"{w}-day observed (mm)", "UNAVAILABLE" if v is None else f"{v:.0f}")
            st.caption(f"Last observed day {a['last_observed'].date()} ({a['days_stale']} days before today). "
                       "No daily baseline is loaded, so there is no anomaly/percentile here.")
            if "fc" in S:
                tbl, info = fm.combine_observed_forecast(obs, S["fc"][0])
                trig = fm.rainfall_trigger(tbl["precip_mm"].values); S["trig_combined"] = trig
                st.metric("Combined trigger (last 7 observed days + forecast)", trig["status"]); st.caption(trig["detail"])
                if info["gap_days"]: st.warning(f"{info['gap_days']} day(s) between the last observed day and the forecast have no data.")
                st.plotly_chart(px.bar(tbl.reset_index(names="date"), x="date", y="precip_mm", color="source",
                                       color_discrete_map={"observed (CHIRPS)": "#2c7fb8", "forecast": "#f39c12", "missing": "#bbbbbb"}),
                                use_container_width=True)
            else:
                st.info("Fetch the forecast above to see the combined observed + forecast trigger.")
        if obs.attrs["failed"]: st.caption(f"Failed/unpublished days: {[k for k, _ in obs.attrs['failed']][:6]}")

with t2:
    st.write("Satellite-derived flood EXTENT for chosen acquisition windows. Needs Copernicus credentials.")
    fields = ui.fields_uploader()
    a, b = st.columns(2)
    pre = a.date_input("Pre-event window", (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10")), key="pre")
    post = b.date_input("Post-event window", (pd.Timestamp("2026-03-14"), pd.Timestamp("2026-03-20")), key="post")
    res_m = st.select_slider("Resolution (m)", [20, 30, 50, 100], value=30)
    if st.button("Run flood mapping") and len(pre) == 2 and len(post) == 2:
        try:
            with st.spinner("Downloading and processing..."):
                S["s1"] = fp.run_s1_flood(ds.CDSEClient().process, vis.district_bbox(district),
                    (f"{pre[0]}T00:00:00Z", f"{pre[1]}T23:59:59Z"), (f"{post[0]}T00:00:00Z", f"{post[1]}T23:59:59Z"), res_m, fields=fields)
        except ds.CredentialsMissing as e: st.error(f"Credentials missing: {e}")
        except (ds.DataUnavailable, ValueError) as e: st.error(f"Could not complete: {e}")
    r = S.get("s1")
    if r:
        c = st.columns(3); c[0].metric("Detected flood extent (ha)", f"{r['flood_ha']:.0f}")
        c[1].metric("Permanent water (ha)", f"{r['permanent_water_ha']:.0f}"); c[2].metric("Excluded / unknown", f"{r['excluded_fraction']:.0%}")
        for w_ in r["warnings"]: st.caption(w_)
        if r["field_table"] is not None: st.dataframe(r["field_table"].round(1))
        ref_up = st.file_uploader("Optional reference flood polygons to score this result", type=["geojson", "json", "gpkg"], key="refup")
        if ref_up is not None:
            try:
                ref = val.load_reference(ref_up.getvalue())
                sc = val.score_masks(r["arrays"]["flood"], val.rasterize_reference(ref, r["meta"]["bbox"], r["arrays"]["flood"].shape), r["arrays"]["excluded"])
                st.json(sc); S.setdefault("val_log", []).append({"district": district, "meta": r["meta"]["post"], **sc})
            except Exception as e: st.error(f"Could not score: {e}")
with t3:
    st.write("Terrain susceptibility needs DEM-derived layers; the baseline is in `src/flood_model.py` "
             "(weights provisional). The Sentinel-1 workflow already applies slope and height-above-water masks.")
    st.json(fm.DEFAULT_WEIGHTS); st.caption(fm.LABEL)
