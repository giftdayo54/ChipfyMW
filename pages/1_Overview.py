import pandas as pd, streamlit as st
from src import ui, data_validation as dv
district, scale = ui.page("Overview")
S = st.session_state
st.subheader(f"Status: {district}")
c = st.columns(4)
fc = S.get("fc"); s1 = S.get("s1"); veg = S.get("veg"); spi = S.get("spi_latest")
c[0].metric("Rainfall trigger (forecast)", fc[1]["status"] if fc else "UNAVAILABLE")
c[1].metric("Satellite flood extent (ha)", f"{s1['flood_ha']:.0f}" if s1 else "UNAVAILABLE")
c[2].metric("Veg. stress (ha)", "UNAVAILABLE" if not veg or veg.get("stress_ha") is None else f"{veg['stress_ha']:.0f}")
c[3].metric(f"SPI-{scale}", "UNAVAILABLE" if spi is None else f"{spi:.2f}")
if fc:
    w = dv.staleness_warning(pd.Timestamp(fc[0].attrs["retrieved_at"]), pd.Timestamp.now(tz="UTC"), 12)
    if w: st.warning("Forecast is stale: " + w)
st.caption("Values appear after you run the analyses on the Flood and Drought pages. Population exposure needs a population raster (not loaded).")
