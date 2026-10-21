"""Synthetic fixtures only (unit tests)."""
import numpy as np, pandas as pd
from src import drought_model as dm, climate_analysis as ca

def test_no_indicators_is_unavailable():
    r = dm.classify_drought()
    assert r["category"] == "Unavailable" and r["n_indicators"] == 0

def test_worst_indicator_wins():
    r = dm.classify_drought(spi=-0.2, vci=15)
    assert r["category"] == "Severe" and r["n_indicators"] == 2

def test_forecast_split_is_chronological_and_honest():
    idx = pd.date_range("1990-01-01", periods=30 * 12, freq="MS")
    s = pd.Series(np.random.default_rng(0).gamma(2, 40, len(idx)), index=idx)   # pure noise
    r = dm.evaluate_chronological(s)
    assert r["train_end"] < r["test_start"] and r["skill_vs_climatology"] < 0.1
    assert dm.evaluate_chronological(s.iloc[:60])["status"] == "insufficient_data"

def test_mann_kendall():
    up = ca.mann_kendall(np.arange(20) + np.random.default_rng(0).normal(0, .5, 20))
    assert up["trend"] == "increasing" and up["sen_slope"] > 0.8
    assert ca.mann_kendall([1, 2, 3])["status"] == "insufficient_data"

def test_season_totals_and_dry_spell():
    idx = pd.date_range("2000-11-01", "2001-04-30", freq="D")
    d = pd.Series(2.0, index=idx)
    assert ca.rainy_season_totals(d).iloc[0] == 2.0 * len(idx)
    d.iloc[10:20] = 0
    assert ca.dry_spells(d)[0][1] == 10
    assert ca.oni_phase(0.8) == "El Nino" and ca.oni_phase(float("nan")) == "Unavailable"
