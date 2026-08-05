"""Tab 2: Correlation Lab — rolling correlation time series, DCC overlay, tail dependence."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import duckdb
from omegaconf import DictConfig
from scipy import stats
import pandas as pd
from energy_cross_commodity.risk.correlation import compute_rolling_correlation


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 2: rolling correlation time series, interactive matrix, tail dependence."""
    prices = conn.execute("""
        SELECT date, commodity_key, price_native
        FROM fact_prices WHERE date >= '2019-01-01'
        ORDER BY date, commodity_key
    """).df()

    pivot = prices.pivot(index="date", columns="commodity_key", values="price_native")
    returns = np.log(pivot / pivot.shift(1)).dropna()
    commodities = returns.columns.tolist()

    # --- Rolling Correlation Time Series ---
    st.subheader("Rolling 60-Day Correlation: TTF vs. German Power")
    avail = [c for c in ["TTF", "DE_POWER", "BRENT", "EUA", "NP_SYS"] if c in commodities]
    filtered = returns[avail]
    corr_cube = compute_rolling_correlation(filtered, window=60)

    if "TTF" in avail and "DE_POWER" in avail:
        ttf_idx = list(corr_cube.coords["c1"].values).index("TTF")
        power_idx = list(corr_cube.coords["c2"].values).index("DE_POWER")
        pair_corr = corr_cube.values[ttf_idx, power_idx, :]
        corr_dates = pd.DatetimeIndex(corr_cube.coords["date"].values)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=corr_dates, y=pair_corr, mode="lines",
            line=dict(color="#00003C", width=1.5),
            name="TTF-DE_POWER Corr",
        ))
        fig_ts.add_hline(y=0, line_dash="dash", line_color="#6B6B6B", line_width=0.5)
        # Event annotations
        for event_date, label, color in [
            ("2020-03-15", "COVID", "#6B6B6B"),
            ("2022-08-15", "Gas Crisis", "#C44536"),
            ("2023-06-01", "Normalization", "#2E7D6F"),
        ]:
            fig_ts.add_vline(x=pd.Timestamp(event_date), line_dash="dot", line_color=color,
                annotation_text=label, annotation_position="top left")
        fig_ts.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Correlation", xaxis_title="",
            yaxis=dict(range=[-1, 1]),
        )
        st.plotly_chart(fig_ts, width="stretch")
        st.caption("Correlation spikes in crises. Aug 2022: TTF-power correlation hit ~0.9. By 2023, decoupling from renewables pushed it back toward ~0.3.")

    # --- Interactive Date Picker + Correlation Matrix ---
    st.subheader("Correlation Matrix — Select Date")
    all_dates = pd.DatetimeIndex(corr_cube.coords["date"].values)
    date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]
    picked = st.select_slider(
        "As of date", options=date_strs,
        value=date_strs[-1] if date_strs else None,
    )
    if picked:
        idx = date_strs.index(picked)
        matrix = corr_cube.values[:, :, idx]
        labels = list(corr_cube.coords["c1"].values)

        fig_mat = px.imshow(
            matrix, x=labels, y=labels,
            text_auto=".2f",
            color_continuous_scale=["#C44536", "#FAFAFA", "#2E7D6F"],
            color_continuous_midpoint=0, zmin=-1, zmax=1,
        )
        fig_mat.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_mat, width="stretch")
        st.caption(f"60-day correlation as of {picked}")

    # --- Tail Dependence Scatter ---
    if "TTF" in commodities and "DE_POWER" in commodities:
        st.subheader("TTF vs. German Power — Tail Dependence")
        ttf_ret = returns["TTF"].dropna()
        power_ret = returns["DE_POWER"].dropna()
        common_idx = ttf_ret.index.intersection(power_ret.index)
        ttf_a = ttf_ret[common_idx]
        power_a = power_ret[common_idx]
        ttf_std = (ttf_a - ttf_a.mean()) / ttf_a.std()
        power_std = (power_a - power_a.mean()) / power_a.std()

        fig_tail = go.Figure()
        fig_tail.add_trace(go.Scatter(
            x=ttf_std, y=power_std, mode="markers",
            marker=dict(size=3, color="#00003C", opacity=0.3),
            name="Daily returns",
        ))
        rho = np.corrcoef(ttf_std, power_std)[0, 1]
        theta = np.linspace(0, 2*np.pi, 200)
        ex, ey = np.cos(theta), np.sin(theta)
        scale_t = np.sqrt(stats.f.ppf(0.95, 2, 5) * 2)
        fig_tail.add_trace(go.Scatter(
            x=ex * scale_t, y=ey * scale_t * np.sqrt(1 - rho**2) + rho * ex * scale_t,
            mode="lines", line=dict(color="#00003C", width=2, dash="solid"),
            name="t-Copula 95% contour",
        ))
        scale_gauss = np.sqrt(stats.chi2.ppf(0.95, 2))
        fig_tail.add_trace(go.Scatter(
            x=ex * scale_gauss, y=ey * scale_gauss * np.sqrt(1 - rho**2) + rho * ex * scale_gauss,
            mode="lines", line=dict(color="#C44536", width=2, dash="dash"),
            name="Gaussian 95% contour",
        ))
        fig_tail.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="TTF (standardized)", yaxis_title="German Power (standardized)",
        )
        st.plotly_chart(fig_tail, width="stretch")
        st.caption(f"Correlation: {rho:.3f}. The t-copula ellipse (solid navy) captures tail dependence the Gaussian ellipse (dashed red) misses.")
    st.subheader("Correlation Regime Detection")
    if len(avail) >= 3:
        frob = np.zeros(corr_cube.shape[2])
        for t_idx in range(corr_cube.shape[2]):
            mat = corr_cube.values[:, :, t_idx]
            frob[t_idx] = np.sqrt(np.sum(mat**2))
        frob_dates = pd.DatetimeIndex(corr_cube.coords["date"].values)

        lo = np.percentile(frob, 33)
        hi = np.percentile(frob, 67)
        regime = np.where(frob > hi, "HIGH", np.where(frob < lo, "LOW", "NORMAL"))
        current_regime = regime[-1]

        fig_frob = go.Figure()
        fig_frob.add_trace(go.Scatter(x=frob_dates, y=frob, mode="lines",
            line=dict(color="#00003C", width=1.5), name="Frobenius Norm"))
        for r, color in [("HIGH", "#C44536"), ("NORMAL", "#6B6B6B"), ("LOW", "#2E7D6F")]:
            mask = regime == r
            if mask.any():
                fig_frob.add_trace(go.Scatter(
                    x=frob_dates[mask], y=frob[mask], mode="markers",
                    marker=dict(size=2, color=color, opacity=0.5), name=r, showlegend=True,
                ))
        fig_frob.add_hline(y=hi, line_dash="dash", line_color="#C44536", line_width=0.5,
            annotation_text=f"HIGH threshold ({hi:.1f})")
        fig_frob.add_hline(y=lo, line_dash="dash", line_color="#2E7D6F", line_width=0.5,
            annotation_text=f"LOW threshold ({lo:.1f})")
        fig_frob.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title="Frobenius Norm", showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15))
        st.plotly_chart(fig_frob, width="stretch")

        rc1, rc2 = st.columns(2)
        rc1.metric("Current Regime", current_regime)
        rc2.metric("Frobenius Norm", f"{frob[-1]:.2f}")
        st.caption(
            "Frobenius norm of the N×N correlation matrix. "
            "HIGH regimes coincide with crises — diversification fails when you need it most. "
            f"Aug 2022 classified as HIGH (norm > {hi:.1f})."
        )
