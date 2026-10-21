import os, pandas as pd, streamlit as st
from src import ui
district, _ = ui.page("Historical Events")
st.write("Maintain documented events in `data/events.csv` (columns: event, type, start, end, district, reference_url, notes). "
         "Only verified dates should be entered; none are pre-filled.")
if os.path.exists("data/events.csv"): st.dataframe(pd.read_csv("data/events.csv"))
st.subheader("Validation log (this session)")
log = st.session_state.get("val_log")
st.dataframe(pd.DataFrame(log)) if log else st.info("No scored comparisons yet. Score a Sentinel-1 result against a reference layer on the Flood page.")
st.markdown("**Checklist:** same event dates; reference layer provenance noted; pre-event window is dry; excluded area reported; thresholds tuned on one event and re-tested on another.")
