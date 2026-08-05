"""Tab 4: Fuel Switch — merit order, fuel-switching signal, carbon pass-through."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    spreads_df_clean = spreads_df.dropna(subset=["power", "carbon"])
    if len(spreads_df_clean) < 60:
        st.info("Insufficient data for pass-through estimation.")
    else:
        dr_power = np.diff(np.log(spreads_df_clean["power"].values))
        dr_carbon = np.diff(np.log(spreads_df_clean["carbon"].values))
        window = 60
        betas = np.full(len(spreads_df_clean) - window, np.nan)
        beta_dates = spreads_df_clean["date"].values[window:]
        for i in range(len(betas)):
            end = i + window
            pw = dr_power[i:end]
            ca = dr_carbon[i:end]
            mask = ~(np.isnan(pw) | np.isnan(ca))
            if mask.sum() > 10:
                slope, _ = np.polyfit(ca[mask], pw[mask], 1)
                betas[i] = slope

        valid = ~np.isnan(betas)
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(
            x=beta_dates[valid], y=betas[valid], mode="lines",
            line=dict(color="#00003C", width=1.5), name="Rolling β (60-day)",
        ))
        fig_pt.add_hrect(y0=0.80, y1=1.00, fillcolor="#2E7D6F", opacity=0.1,
            line_width=0, annotation_text="Empirical Range (0.80–1.00)")
        fig_pt.add_hline(y=0, line_dash="dash", line_color="#6B6B6B", line_width=0.5)
        fig_pt.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title="β (power sensitivity to carbon)")
        st.plotly_chart(fig_pt, use_container_width=True)

        current_beta = betas[valid][-1] if valid.any() else np.nan
        st.caption(
            f"Current pass-through: β = {current_beta:.2f}. "
            f"β = 0.85 means a €10/t carbon increase flows through as €8.50/MWh in German power. "
            "This is the carbon-price-to-power-price transmission mechanism."
        )
    st.subheader("3-2-1 Crack Spread — Seasonal Decomposition")
    crack_prices = conn.execute("""
        SELECT date,
            MAX(CASE WHEN commodity_key='RBOB' THEN price_native END) AS rbob,
            MAX(CASE WHEN commodity_key='GASOIL' THEN price_native END) AS gasoil,
            MAX(CASE WHEN commodity_key='BRENT' THEN price_native END) AS brent
        FROM fact_prices
        WHERE commodity_key IN ('RBOB','GASOIL','BRENT')
        GROUP BY date
        HAVING rbob IS NOT NULL AND gasoil IS NOT NULL AND brent IS NOT NULL
        ORDER BY date
    """).df()

    if len(crack_prices) < 504:
        st.info("Insufficient crack spread history for seasonal decomposition (need ~2 years).")
    else:
        from energy_cross_commodity.spreads.crack_spread import compute_321_crack, decompose_crack_spread
        import pandas as pd

        crack = compute_321_crack(
            crack_prices["rbob"].values,
            crack_prices["gasoil"].values,
            crack_prices["brent"].values,
        )
        dates = pd.DatetimeIndex(crack_prices["date"].values)
        decomp = decompose_crack_spread(dates, crack, period=252)

        fig_seas = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Trend", "Seasonal (Annual Pattern)", "Residual"],
            vertical_spacing=0.08)

        fig_seas.add_trace(go.Scatter(x=dates, y=decomp["trend"], mode="lines",
            line=dict(color="#00003C", width=1.5), name="Trend"), row=1, col=1)
        fig_seas.add_trace(go.Scatter(x=dates, y=decomp["seasonal"], mode="lines",
            line=dict(color="#2E7D6F", width=1), name="Seasonal"), row=2, col=1)
        fig_seas.add_hline(y=0, line_dash="dash", line_color="#6B6B6B", line_width=0.5, row=2, col=1)
        fig_seas.add_trace(go.Scatter(x=dates, y=decomp["resid"], mode="lines",
            line=dict(color="#6B6B6B", width=0.8), name="Residual"), row=3, col=1)
        fig_seas.add_hline(y=0, line_dash="dash", line_color="#6B6B6B", line_width=0.5, row=3, col=1)
        fig_seas.update_layout(height=500, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
        st.plotly_chart(fig_seas, use_container_width=True)
        st.caption(
            "STL decomposition of the 3-2-1 crack spread. "
            "The seasonal component captures predictable annual patterns — summer gasoline demand, winter heating oil. "
            "The residual highlights structural breaks like COVID 2020 and the 2022 energy crisis."
        )
