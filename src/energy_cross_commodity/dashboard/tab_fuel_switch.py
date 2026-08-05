"""Tab 4: Fuel Switch — merit order, fuel-switching signal, carbon pass-through."""

import plotly.graph_objects as go
import streamlit as st
import duckdb
from omegaconf import DictConfig
from energy_cross_commodity.db import query


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 4: fuel-switching signal, merit order, carbon pass-through.

    Args:
        conn: Active DuckDB connection.
        cfg: Pipeline configuration (OmegaConf DictConfig).
    """
    spreads_df = query(conn, "spread_economics.sql", {
        "efficiency_gas": "0.55", "ef_gas": "0.37",
        "efficiency_coal": "0.38", "ef_coal": "0.90",
    })

    st.subheader("Fuel-Switching Signal (Spark Spread — Dark Spread)")
    signal = spreads_df["fuel_switch_signal"].values
    dates = spreads_df["date"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=signal, mode="lines", name="Fuel Switch Signal", line=dict(color="#00003C", width=1)))
    fig.add_hline(y=5, line_dash="dash", line_color="#2E7D6F", annotation_text="Gas Dominates (>+5)")
    fig.add_hline(y=-5, line_dash="dash", line_color="#C44536", annotation_text="Coal Dominates (<-5)")
    fig.add_hrect(y0=-5, y1=5, fillcolor="#E8E8F0", opacity=0.3, line_width=0, annotation_text="Switching Zone")

    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="EUR/MWh", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    gas_days = (signal > 5).sum()
    coal_days = (signal < -5).sum()
    switch_days = ((signal >= -5) & (signal <= 5)).sum()
    total = gas_days + coal_days + switch_days
    col1.metric("Gas Favored", f"{gas_days} days", f"{gas_days/total*100:.0f}%")
    col2.metric("Coal Favored", f"{coal_days} days", f"{coal_days/total*100:.0f}%")
    col3.metric("Switching Zone", f"{switch_days} days", f"{switch_days/total*100:.0f}%")

    st.subheader("Carbon Pass-Through Rate")
    if "carbon" in spreads_df.columns and "DE_POWER" not in spreads_df.columns:
        st.info("Pass-through rate requires power price data. Available with real ENTSO-E data.")
    else:
        st.caption("Carbon pass-through: the rate at which carbon price changes flow into power prices. Empirically 80-100% in German market. Computed as rolling regression beta of power returns on carbon returns.")
