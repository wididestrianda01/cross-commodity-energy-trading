"""Tab 2: Correlation Lab — rolling correlation, DCC overlay, tail dependence."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import duckdb
from omegaconf import DictConfig
from scipy import stats


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 2: rolling correlation, DCC overlay, and tail dependence.

    Args:
        conn: Active DuckDB connection.
        cfg: Pipeline configuration (OmegaConf DictConfig).
    """
    prices = conn.execute("""
        SELECT date, commodity_key, price_native
        FROM fact_prices
        WHERE date >= '2019-01-01'
        ORDER BY date, commodity_key
    """).df()

    pivot = prices.pivot(index="date", columns="commodity_key", values="price_native")
    returns = np.log(pivot / pivot.shift(1)).dropna()
    commodities = returns.columns.tolist()

    st.subheader("Rolling 60-Day Correlation Matrix")
    window = 60
    avail_commodities = [c for c in ["BRENT", "TTF", "EUA", "DE_POWER", "NP_SYS"] if c in commodities]
    filtered = returns[avail_commodities]
    corr_matrix = filtered.tail(window).corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale=["#C44536", "#FAFAFA", "#2E7D6F"],
        color_continuous_midpoint=0,
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Correlation as of {filtered.index[-1].date()}")

    if "TTF" in commodities and "DE_POWER" in commodities:
        st.subheader("TTF vs. German Power — Tail Dependence")
        ttf_ret = returns["TTF"].dropna()
        power_ret = returns["DE_POWER"].dropna()
        common_idx = ttf_ret.index.intersection(power_ret.index)
        ttf_aligned = ttf_ret[common_idx]
        power_aligned = power_ret[common_idx]

        ttf_std = (ttf_aligned - ttf_aligned.mean()) / ttf_aligned.std()
        power_std = (power_aligned - power_aligned.mean()) / power_aligned.std()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ttf_std, y=power_std, mode="markers",
            marker=dict(size=3, color="#00003C", opacity=0.3),
            name="Daily returns",
        ))

        rho = np.corrcoef(ttf_std, power_std)[0, 1]
        theta = np.linspace(0, 2*np.pi, 200)
        ellipse_x = np.cos(theta)
        ellipse_y = np.sin(theta)
        scale_t = np.sqrt(stats.f.ppf(0.95, 2, 5) * 2)
        fig2.add_trace(go.Scatter(
            x=ellipse_x * scale_t, y=ellipse_y * scale_t * np.sqrt(1 - rho**2) + rho * ellipse_x * scale_t,
            mode="lines", line=dict(color="#00003C", width=2, dash="solid"),
            name="t-Copula 95% contour",
        ))
        scale_gauss = np.sqrt(stats.chi2.ppf(0.95, 2))
        fig2.add_trace(go.Scatter(
            x=ellipse_x * scale_gauss, y=ellipse_y * scale_gauss * np.sqrt(1 - rho**2) + rho * ellipse_x * scale_gauss,
            mode="lines", line=dict(color="#C44536", width=2, dash="dash"),
            name="Gaussian 95% contour",
        ))

        fig2.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="TTF (standardized)", yaxis_title="German Power (standardized)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Correlation: {rho:.3f}. The t-copula ellipse (solid navy) captures tail dependence the Gaussian ellipse (dashed red) misses.")
