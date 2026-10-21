import pandas as pd, streamlit as st
from src import ui, config, data_sources as ds, rainfall_analysis as ra, drought_model as dm, visualisation as vis
from src import vegetation_pipeline as vp, chirps
district, scale = ui.page("Drought Intelligence")
S = st.session_state
t1, t2 = st.tabs(["Meteorological (SPI)", "Vegetation stress (Sentinel-2)"])
with t1:
    with st.expander("Fetch CHIRPS rainfall for this district (satellite-gauge blended estimate)"):
        y0, y1 = st.slider("Years", 1981, pd.Timestamp.today().year, (1991, pd.Timestamp.today().year))
        st.caption("About 12 downloads per year, cached on disk after the first run. SPI needs >= 30 years.")
        if st.button("Fetch CHIRPS"):
            bar = st.progress(0.0)
            try:
                s = chirps.monthly_series(vis.district_bbox(district), y0, y1, progress=lambda n, tot: bar.progress(n / tot))
                S["rain"] = s; S["rain_info"] = {**chirps.completeness(s), "failed": s.attrs["failed"], "note": s.attrs["note"]}
            except (ds.DataUnavailable, ValueError) as e: st.error(f"CHIRPS unavailable: {e}")
        info = S.get("rain_info")
        if info:
            st.write(f"{info['months_or_days']} months, {info['missing']} missing ({info['fraction_missing']:.0%}). {info['note']}")
            if info["failed"]: st.warning(f"Gaps are left empty, not filled: {info['failed'][:5]}{' ...' if len(info['failed']) > 5 else ''}")
    up = st.file_uploader("CSV: date, precip_mm (daily or monthly)", type="csv")
    if up is None and "rain" not in S:
        st.info("No rainfall data: drought metrics UNAVAILABLE (not a low-risk result).")
    else:
        try:
            if up is not None: S["rain"] = ui.parse_rain(up.getvalue())
            monthly = S["rain"]; spi = ra.spi(monthly, scale); last = spi.dropna()
            S["spi_latest"] = float(last.iloc[-1]) if len(last) else None
            res = dm.classify_drought(spi=S["spi_latest"])
            c = st.columns(3); c[0].metric(f"SPI-{scale}", "UNAVAILABLE" if S["spi_latest"] is None else f"{S['spi_latest']:.2f}")
            c[1].metric("Category", res["category"]); c[2].metric("Latest valid month", str(last.index[-1].date()) if len(last) else "n/a")
            if S["spi_latest"] is None: st.warning(f"SPI unavailable: needs >= {config.MIN_SPI_YEARS} years of data.")
            st.plotly_chart(vis.spi_chart(spi, scale), use_container_width=True)
            S["skill"] = dm.evaluate_chronological(monthly); st.write("Forecast-skill check (chronological split):", S["skill"])
        except Exception as e: st.error(f"Could not process file: {e}")
with t2:
    st.write("Stress vs the same season in earlier years. Not a drought diagnosis.")
    fields = ui.fields_uploader()
    cur = st.date_input("Current window", (pd.Timestamp.today() - pd.Timedelta(days=30), pd.Timestamp.today()), key="vw")
    nb = st.slider("Baseline years", 3, 8, 5)
    if st.button("Run vegetation analysis") and len(cur) == 2:
        try:
            with st.spinner("Downloading current and baseline composites..."):
                S["veg"] = vp.run_s2_condition(ds.CDSEClient().process, vis.district_bbox(district),
                    (f"{cur[0]}T00:00:00Z", f"{cur[1]}T23:59:59Z"), nb, fields=fields)
        except ds.CredentialsMissing as e: st.error(f"Credentials missing: {e}")
        except (ds.DataUnavailable, ValueError) as e: st.error(f"Could not complete: {e}")
    v = S.get("veg")
    if v:
        if v["status"] != "ok": st.warning(v["message"])
        else:
            c = st.columns(3); c[0].metric("Clear pixels", f"{v['valid_fraction']:.0%}"); c[1].metric("Mean NDVI", f"{v['mean_ndvi']:.2f}")
            c[2].metric("Stressed area, ha", "UNAVAILABLE" if v["stress_ha"] is None else f"{v['stress_ha']:.0f}")
            if v.get("message"): st.warning(v["message"])
            if v.get("field_table") is not None: st.dataframe(v["field_table"].round(1))
            st.caption(v["caveat"])
