"""Map and chart builders."""
import json
from pathlib import Path
import folium
import plotly.graph_objects as go
import numpy as np

# APPROXIMATE district centroids for navigation only. Replace with official boundaries by
# placing data/districts.geojson (properties.name) in the project.
DISTRICT_CENTROIDS = {"Chikwawa": (-16.03, 34.80), "Nsanje": (-16.92, 35.26), "Blantyre": (-15.79, 35.00),
    "Thyolo": (-16.07, 35.14), "Mulanje": (-16.03, 35.50), "Phalombe": (-15.80, 35.65),
    "Zomba": (-15.38, 35.33), "Chiradzulu": (-15.70, 35.18), "Mwanza": (-15.60, 34.52),
    "Neno": (-15.40, 34.65), "Balaka": (-14.98, 34.96), "Mangochi": (-14.48, 35.26),
    "Machinga": (-14.97, 35.52)}
BOUNDARY_FILE = Path("data/districts.geojson")

def district_bbox(name: str, half_deg: float = 0.15) -> tuple:
    lat, lon = DISTRICT_CENTROIDS[name]
    return (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)

def build_map(selected: str) -> folium.Map:
    lat, lon = DISTRICT_CENTROIDS[selected]
    m = folium.Map(location=[lat, lon], zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    if BOUNDARY_FILE.exists():
        folium.GeoJson(json.loads(BOUNDARY_FILE.read_text()), name="Districts (official file)",
                       style_function=lambda f: {"fillOpacity": 0.05, "weight": 1}).add_to(m)
    else:
        for n, (a, b) in DISTRICT_CENTROIDS.items():
            folium.CircleMarker([a, b], radius=9 if n == selected else 5, tooltip=f"{n} (approx. centroid)",
                                color="#c0392b" if n == selected else "#34495e", fill=True).add_to(m)
    s, w, n_, e = district_bbox(selected)[1], district_bbox(selected)[0], district_bbox(selected)[3], district_bbox(selected)[2]
    folium.Rectangle([[s, w], [n_, e]], color="#e67e22", fill=False, tooltip="Analysis window").add_to(m)
    folium.LatLngPopup().add_to(m)
    folium.LayerControl().add_to(m)
    return m

def spi_chart(spi_series, scale: int) -> go.Figure:
    v = spi_series.dropna()
    fig = go.Figure(go.Bar(x=v.index, y=v.values, marker_color=np.where(v.values < 0, "#c0392b", "#2980b9"),
                           name=f"SPI-{scale} (historical, observed)"))
    fig.add_hline(y=-1, line_dash="dot", annotation_text="Moderately dry")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="SPI")
    return fig
