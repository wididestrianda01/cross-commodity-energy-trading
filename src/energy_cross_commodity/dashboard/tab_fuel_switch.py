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
    st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig_pt, width="stretch")

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
        st.plotly_chart(fig_seas, width="stretch")
        st.caption(
            "STL decomposition of the 3-2-1 crack spread. "
            "The seasonal component captures predictable annual patterns — summer gasoline demand, winter heating oil. "
            "The residual highlights structural breaks like COVID 2020 and the 2022 energy crisis."
        )

    st.subheader("Break-Even Carbon Price")
    be_df = spreads_df.dropna(subset=["gas", "coal", "carbon"])
    if len(be_df) < 60:
        st.info("Insufficient data for break-even carbon.")
    else:
        from energy_cross_commodity.spreads.spark_spread import compute_break_even_carbon
        be_carbon = compute_break_even_carbon(be_df["gas"].values, be_df["coal"].values)
        be_dates = be_df["date"].values

        fig_be = go.Figure()
        fig_be.add_trace(go.Scatter(x=be_dates, y=be_carbon, mode="lines",
            line=dict(color="#00003C", width=1.5), name="Break-Even Carbon"))
        fig_be.add_trace(go.Scatter(x=be_dates, y=be_df["carbon"].values, mode="lines",
            line=dict(color="#C44536", width=1.5), name="Actual EUA Price"))

        # Shade region where actual > break-even (gas favored)
        gas_favored = be_df["carbon"].values > be_carbon
        fig_be.add_trace(go.Scatter(
            x=be_dates, y=np.where(gas_favored, be_df["carbon"].values, np.nan),
            mode="none", fill="tozeroy", fillcolor="#2E7D6F", opacity=0.08,
            name="Gas Favored (EUA > Break-Even)", showlegend=True,
        ))
        fig_be.add_trace(go.Scatter(
            x=be_dates, y=np.where(~gas_favored, be_df["carbon"].values, np.nan),
            mode="none", fill="tozeroy", fillcolor="#C44536", opacity=0.08,
            name="Coal Favored (EUA < Break-Even)", showlegend=True,
        ))

        fig_be.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title="EUR/tCO2", legend=dict(orientation="h", yanchor="top", y=-0.15))
        st.plotly_chart(fig_be, width="stretch")

        current_be = float(be_carbon[-1]) if len(be_carbon) > 0 else np.nan
        current_eua = float(be_df["carbon"].values[-1]) if len(be_df) > 0 else np.nan
        be_col1, be_col2 = st.columns(2)
        be_col1.metric("Break-Even Carbon", f"€{current_be:.1f}/t",
            delta=f"vs Actual €{current_eua:.1f}/t",
            delta_color="normal" if current_eua > current_be else "inverse")
        be_col2.metric("Actual EUA", f"€{current_eua:.1f}/t")
        st.caption(
            "When actual carbon price > break-even, gas generation is cheaper than coal. "
            "The widening gap since 2021 reflects structural coal-to-gas switching driven by carbon policy."
        )

    with st.expander("Methodology & Interpretation"):
        st.markdown("""
**Fuel-switching signal** = Clean Spark Spread − Clean Dark Spread. When the signal is
above +5 EUR/MWh, gas generation is the cheaper marginal fuel and tends to set the
power price. Below −5 EUR/MWh, coal is cheaper. The ±5 EUR/MWh band is the switching
zone where small price changes can flip the marginal fuel. The August 2022 gas crisis
produced roughly 60 consecutive trading days of coal-favored signal as TTF spiked to
over 300 EUR/MWh.

**Carbon pass-through rate** measures how carbon price changes flow through to German
power prices. It is estimated as the rolling 60-day regression coefficient β in:
Δlog(P_power) = α + β × Δlog(P_carbon) + ε. The empirical range is 0.80–1.00, meaning
a €10/t carbon increase raises power prices by €8–10/MWh. The pass-through is near
complete because carbon is a marginal cost passed directly to consumers under the EU
ETS — generators do not absorb carbon costs; they pass them through.

**Seasonal decomposition** uses STL (Seasonal-Trend decomposition with LOESS) on the
3-2-1 crack spread with a period of 252 trading days. The trend component captures
structural shifts (the 2022 energy crisis spike, post-2023 normalization). The seasonal
component isolates the predictable annual pattern — stronger crack spreads in summer
(gasoline demand) and winter (heating oil demand). The residual captures noise and
one-off events (COVID 2020, the March 2022 spike). If the seasonal component has
consistent amplitude, the pattern is stable and tradable; if it varies, structural
change is occurring.

**Break-even carbon price** solves CSS = CDS for P_carbon: the carbon price at which
gas and coal generation are equally profitable on a clean basis. The formula:
P_BE = (P_gas/η_gas − P_coal/η_coal) / (ε_coal − ε_gas)
where η is thermal efficiency (0.55 gas, 0.38 coal) and ε is the emission factor
(0.37 gas, 0.90 coal tCO2/MWh). When the actual EUA price exceeds the break-even,
gas is structurally cheaper despite its higher fuel cost — carbon policy has made gas
the preferred marginal fuel in Germany since approximately 2021.
""")
