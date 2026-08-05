"""Tab 1: Market Monitor — price heatmap, normalized chart, spread dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st
import duckdb
from omegaconf import DictConfig


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 1: price heatmap, normalized chart, and spread dashboard.

    Args:
        conn: Active DuckDB connection.
        cfg: Pipeline configuration (OmegaConf DictConfig).
    """
    prices = conn.execute("""
        SELECT date, commodity_key, price_eur_mwh
        FROM fact_prices
        WHERE date >= '2022-01-01'
        ORDER BY date, commodity_key
    """).df()

    st.subheader("Price Heatmap — Daily Returns")
    pivot = prices.pivot(index="commodity_key", columns="date", values="price_eur_mwh")
    log_rets = np.log(pivot / pivot.shift(1, axis=1)).iloc[:, -20:]
    returns = log_rets.clip(lower=-1.0, upper=1.0)

    fig = px.imshow(
        returns,
        color_continuous_scale=["#C44536", "#FAFAFA", "#2E7D6F"],
        color_continuous_midpoint=0,
        aspect="auto",
    )
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Normalized Price Chart (Index = 100 at 2022-01-01)")
    first_valid = pivot.bfill(axis=1).iloc[:, 0]
    norm = pivot.div(first_valid, axis=0) * 100
    fig2 = px.line(norm.T)
    fig2.update_layout(
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15),
        yaxis_title="Index (100 = Jan 2022)",
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Spark / Dark / Crack Spread Dashboard")
    from energy_cross_commodity.db import query
    spreads_df = query(conn, "spread_economics.sql", {
        "efficiency_gas": "0.55", "ef_gas": "0.37",
        "efficiency_coal": "0.38", "ef_coal": "0.90",
    })

    fig3 = make_subplots(rows=1, cols=3, subplot_titles=["Clean Spark Spread", "Clean Dark Spread", "Fuel Switch Signal"])
    fig3.add_trace(go.Scatter(x=spreads_df["date"], y=spreads_df["clean_spark_spread"], mode="lines", name="Spark", line=dict(color="#00003C")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=spreads_df["date"], y=spreads_df["clean_dark_spread"], mode="lines", name="Dark", line=dict(color="#C44536")), row=1, col=2)
    positive = spreads_df["fuel_switch_signal"].clip(lower=0)
    negative = spreads_df["fuel_switch_signal"].clip(upper=0)
    fig3.add_trace(go.Bar(x=spreads_df["date"], y=positive, name="Gas Favored", marker_color="#2E7D6F"), row=1, col=3)
    fig3.add_trace(go.Bar(x=spreads_df["date"], y=negative, name="Coal Favored", marker_color="#C44536"), row=1, col=3)
    fig3.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, width="stretch")

    with st.expander("Methodology & Interpretation"):
        st.markdown("""
**Price heatmap** shows the last 20 trading days of daily returns for each commodity.
Red cells indicate upward moves (>2%), green cells indicate downward moves (<-2%),
white cells show flat trading. This is a desk-level morning view — a trader sees at a
glance which commodities are moving and in which direction.

**Normalized price chart** rebases all commodities to 100 at January 2022. The visual
separation between lines reveals relative performance: the 2022 gas crisis shows TTF
diverging sharply from Brent and power, while the post-2023 convergence reflects market
normalization. Commodities that trade in different native units (USD/bbl, EUR/MWh,
EUR/tCO2) become directly comparable after normalization.

**Spread dashboard** shows the three core trading signals:
- *Clean Spark Spread* (gas-to-power margin): power revenue minus gas fuel cost minus
  carbon cost. Positive → gas plants are profitable.
- *Clean Dark Spread* (coal-to-power margin): same calculation for coal. The carbon
  cost component is roughly 2.5× larger for coal due to higher emissions intensity.
- *Fuel-switching signal* (CSS − CDS): green bars = gas favored, red bars = coal favored.
  The August 2022 gas crisis produced a sustained coal-favored signal as gas became
  uneconomical despite high carbon costs.
""")
