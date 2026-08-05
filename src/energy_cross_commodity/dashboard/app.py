"""Streamlit dashboard — Cross-Commodity Energy Trading Analytics."""

import streamlit as st
from energy_cross_commodity.utils.config import load_config
from energy_cross_commodity.db import get_connection

st.set_page_config(
    page_title="Cross-Commodity Analytics",
    page_icon=":chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,300..700;1,300..700&family=Georgia&family=Geist+Mono&display=swap');

    :root {
        --kth-navy: #00003C;
        --kth-canvas: #FAFAFA;
        --kth-accent: #00003C;
    }

    html, body, [class*="css"] {
        font-family: 'Figtree', 'Georgia', sans-serif;
        background-color: var(--kth-canvas);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Figtree', sans-serif;
        color: var(--kth-navy);
        font-weight: 600;
    }

    code, pre, .stCode {
        font-family: 'Geist Mono', monospace;
    }

    .stApp {
        background-color: var(--kth-canvas);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
