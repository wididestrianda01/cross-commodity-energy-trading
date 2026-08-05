"""Streamlit dashboard — Cross-Commodity Energy Trading Analytics."""

import streamlit as st

st.set_page_config(
    page_title="Cross-Commodity Analytics",
    page_icon=":chart:",
    layout="wide",
)

from energy_cross_commodity.utils.config import load_config
from energy_cross_commodity.db import get_connection

cfg = load_config()
conn = get_connection(cfg.data.db_path)

st.title("Cross-Commodity Energy Trading Analytics")
st.caption("Brent Crude  ·  TTF Natural Gas  ·  EUA Carbon  ·  European Power")

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Monitor",
    "Correlation Lab",
    "Risk Command",
    "Fuel Switch",
])

with tab1:
    from energy_cross_commodity.dashboard.tab_market import render
    render(conn, cfg)

with tab2:
    from energy_cross_commodity.dashboard.tab_correlation import render
    render(conn, cfg)

with tab3:
    from energy_cross_commodity.dashboard.tab_risk import render
    render(conn, cfg)

with tab4:
    from energy_cross_commodity.dashboard.tab_fuel_switch import render
    render(conn, cfg)
