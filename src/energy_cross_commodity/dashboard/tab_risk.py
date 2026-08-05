"""Tab 3: Risk Command — VaR waterfall, breaches, scenario P&L."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import duckdb
from omegaconf import DictConfig
from energy_cross_commodity.risk.scenarios import SCENARIOS, run_scenario


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 3: VaR waterfall, scenario P&L breakdown.

    Args:
        conn: Active DuckDB connection.
        cfg: Pipeline configuration (OmegaConf DictConfig).
    """
    prices = conn.execute("""
        SELECT date, commodity_key, price_native
        FROM fact_prices WHERE date >= '2019-01-01'
        ORDER BY date, commodity_key
    """).df()
    pivot = prices.pivot(index="date", columns="commodity_key", values="price_native")
    returns = np.log(pivot / pivot.shift(1)).dropna()

    st.subheader("Portfolio VaR Decomposition")
    positions = {k: v.notional_eur for k, v in cfg.portfolio.positions.items()}
    vol = returns.std() * np.sqrt(252)
    individual_var = {c: abs(positions.get(c, 0)) * vol.get(c, 0) * 1.645 for c in returns.columns if c in positions}

    fig = go.Figure(go.Waterfall(
        name="VaR", orientation="v",
        measure=["relative"] * len(individual_var) + ["total"],
        x=list(individual_var.keys()) + ["Total"],
        y=list(individual_var.values()) + [sum(individual_var.values())],
        connector={"line": {"color": "#6B6B6B"}},
        decreasing={"marker": {"color": "#C44536"}},
        increasing={"marker": {"color": "#C44536"}},
        totals={"marker": {"color": "#00003C"}},
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    total_var = sum(individual_var.values())
    col1.metric("VaR 95% (1-day)", f"€{total_var:,.0f}")
    col2.metric("VaR 99% (1-day)", f"€{total_var * 1.41:,.0f}")
    col3.metric("ES 97.5%", f"€{total_var * 1.25:,.0f}")

    st.subheader("Stress Scenario P&L")
    scenario_choice = st.selectbox("Select scenario", list(SCENARIOS.keys()), format_func=lambda x: SCENARIOS[x].name)
    scenario = SCENARIOS[scenario_choice]
    current_prices = {c: float(pivot[c].iloc[-1]) for c in pivot.columns if c in positions}
    result = run_scenario(positions, scenario, current_prices)

    items = list(result.pnl_by_position.keys())
    values = list(result.pnl_by_position.values())

    fig2 = go.Figure(go.Waterfall(
        name="Scenario P&L", orientation="v",
        measure=["relative"] * len(items) + ["total"],
        x=items + ["Total"],
        y=values + [result.total_pnl],
        connector={"line": {"color": "#6B6B6B"}},
        increasing={"marker": {"color": "#2E7D6F"}},
        decreasing={"marker": {"color": "#C44536"}},
        totals={"marker": {"color": "#00003C"}},
    ))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        title=f"{scenario.name}: Net P&L = €{result.total_pnl:,.0f}")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(scenario.description)
