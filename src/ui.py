"""Shared Streamlit helpers: page frame, sidebar, session-state accessors."""
import io
import pandas as pd
import streamlit as st
from . import config, visualisation as vis, field_mask as fm

def page(title: str):
    st.set_page_config(page_title=f"M-CHIP - {title}", layout="wide")
    st.title(title); st.warning(config.DISCLAIMER)
    with st.sidebar:
        st.selectbox("District", sorted(vis.DISTRICT_CENTROIDS), key="district")
        st.selectbox("SPI scale (months)", config.SPI_SCALES_MONTHS, index=1, key="scale")
        st.caption("Selections persist across pages.")
    return st.session_state["district"], st.session_state["scale"]

def unavailable(label: str):
    st.metric(label, "UNAVAILABLE")

@st.cache_data(show_spinner=False)
def parse_rain(raw: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"]).set_index("date").sort_index()
    s = df["precip_mm"]
    if s.index.to_series().diff().median() < pd.Timedelta(days=20):
        s = s.resample("MS").sum(min_count=20)
    return s

def fields_uploader():
    up = st.file_uploader("Field boundaries (GeoJSON or GPKG, must declare a CRS)", type=["geojson", "json", "gpkg"], key="field_up")
    if up is not None:
        try:
            st.session_state["fields"] = fm.load_fields(up.getvalue())
            st.success(f"{len(st.session_state['fields'])} field polygons loaded.")
        except Exception as e:
            st.error(f"Could not load fields: {e}")
    return st.session_state.get("fields")

def seasonal_totals(monthly: pd.Series) -> pd.Series:
    import numpy as np
    ms = monthly[(monthly.index.month >= 11) | (monthly.index.month <= 4)]
    yr = np.where(ms.index.month >= 11, ms.index.year, ms.index.year - 1)
    g = ms.groupby(yr)
    return g.sum(min_count=6)[g.count() == 6]
