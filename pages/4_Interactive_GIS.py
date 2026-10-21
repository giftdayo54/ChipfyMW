import folium, streamlit as st
from streamlit_folium import st_folium
from src import ui, visualisation as vis
district, _ = ui.page("Interactive GIS")
S = st.session_state
fields = ui.fields_uploader()
m = vis.build_map(district)
if fields is not None:
    folium.GeoJson(fields.__geo_interface__, name="Fields", style_function=lambda f: {"color": "#27ae60", "fillOpacity": .2}).add_to(m)
s1 = S.get("s1")
if s1 and len(s1["polygons"]):
    folium.GeoJson(s1["polygons"].__geo_interface__, name="Detected flood (Sentinel-1)", style_function=lambda f: {"color": "#0b5394", "fillOpacity": .5}).add_to(m)
folium.LayerControl().add_to(m)
st_folium(m, height=600, use_container_width=True)
st.caption("Click the map for coordinates. District markers are approximate centroids unless data/districts.geojson exists.")
