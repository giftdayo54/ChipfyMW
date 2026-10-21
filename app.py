"""M-CHIP home. Use the sidebar page list to navigate."""
import streamlit as st
from src import ui
ui.page("Malawi Climate Hazard Intelligence Platform (M-CHIP)")
st.write("Flood and drought monitoring for Malawi using Copernicus satellite data, with transparent, "
         "uncalibrated-where-unvalidated methods. Start at **Overview**, or open **Data and Model Diagnostics** "
         "to check credentials and source status.")
st.info("Nothing here is a validated operational forecast. Missing data is shown as UNAVAILABLE, never as low risk.")
