import numpy as np, plotly.express as px, streamlit as st
from src import ui, climate_analysis as ca
district, _ = ui.page("Climate Variability")
S = st.session_state
if "rain" not in S: st.info("Upload rainfall on the Drought Intelligence page first. Trends UNAVAILABLE.")
else:
    monthly = S["rain"]; seas = ui.seasonal_totals(monthly)
    st.plotly_chart(px.bar(x=seas.index, y=seas.values, labels={"x": "Season start year", "y": "Nov-Apr rainfall (mm)"}), use_container_width=True)
    st.write("Season-total trend (Mann-Kendall / Sen):", ca.mann_kendall(seas.values))
    clim = monthly.groupby(monthly.index.month).mean()
    st.plotly_chart(px.line(x=clim.index, y=clim.values, labels={"x": "Month", "y": "Mean monthly rainfall (mm)"}), use_container_width=True)
oni = st.number_input("ONI value (enter from NOAA CPC)", value=0.0, step=0.1)
st.metric("ENSO phase", ca.oni_phase(oni)); st.caption("ENSO shifts regional odds; it is not a deterministic local flood/drought predictor.")
