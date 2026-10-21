# M-CHIP: Malawi Climate Hazard Intelligence Platform (in progress)

Decision-support tool. It does NOT replace official warnings from Malawi's responsible authorities.
Missing data is shown as UNAVAILABLE, never as low risk.

## Install and run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env      # add CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (free Copernicus Data Space OAuth client)
    export $(grep -v '^#' .env | xargs)
    streamlit run app.py
    python -m pytest          # unit tests (synthetic fixtures only)

## Pages
Overview, Flood Intelligence (rainfall trigger, Sentinel-1 extent, scoring vs reference), Drought Intelligence
(SPI from your CSV, Sentinel-2 vegetation stress), Interactive GIS, Climate Variability, Exposure and Vulnerability,
Adaptation Planner, Historical Events, Reports and Downloads, Data and Model Diagnostics.

## Field boundaries
Upload GeoJSON/GPKG with a declared CRS (Flood or Drought page). Results are then reported per field in hectares.

## Known limitations
- Copernicus endpoints and weather APIs were not exercised live during development; tests use synthetic data.
- Flood thresholds, VCI stress rule and susceptibility weights are uncalibrated defaults. Calibrate on a documented event
  (e.g. Cyclone Freddy, March 2023) and re-test on another before relying on outputs.
- Sentinel-1 under-detects flooding below dense canopy/urban areas; unflagged pixels are not confirmed dry.
- Sentinel-2 mosaics use least-cloud tiles, not per-pixel cloud-free composites.
- CHIRPS adapter written from the CHC archive layout but not tested live; gaps are left empty, never filled.
- Population needs a user-supplied GeoTIFF; OSM infrastructure is incomplete in places; Overpass/CHIRPS daily not tested live.
- No SMAP or climate-projection modules yet.
- District markers are approximate centroids; add data/districts.geojson for official boundaries.
