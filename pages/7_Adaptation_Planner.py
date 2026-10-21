import streamlit as st
from src import ui, adaptation as ad
district, _ = ui.page("Adaptation Planner")
S = st.session_state
fc = S.get("fc"); default_f = bool(fc and fc[1]["status"] == "Rainfall trigger exceeded") or bool(S.get("s1") and S["s1"]["flood_ha"] > 0)
fl = st.checkbox("Flood signal present", value=default_f); dr = st.checkbox("Drought signal present")
cane = st.checkbox("Sugarcane / irrigated agriculture", value=True)
rec = ad.recommend(fl, dr, cane); S["rec"] = rec; st.caption(rec["notice"])
if not rec["items"]: st.info("No signal selected: nothing to recommend. This is not a statement that risk is low.")
for k, v in rec["items"]: st.write(f"**{k}:** {v}")
