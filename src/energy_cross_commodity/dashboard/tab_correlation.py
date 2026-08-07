"""Tab 2: Correlation Lab — rolling correlation time series, DCC overlay, tail dependence."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import duckdb
from omegaconf import DictConfig, OmegaConf
from scipy import stats
import pandas as pd
from energy_cross_commodity.risk.correlation import compute_rolling_correlation
from energy_cross_commodity.risk.copula import fit_t_copula
from energy_cross_commodity.risk.returns import compute_log_returns


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 2: rolling correlation time series, interactive matrix, tail dependence."""
    prices = conn.execute("""
        SELECT date, commodity_key, price_native
        FROM fact_prices WHERE date >= '2019-01-01'
        ORDER BY date, commodity_key
    """).df()

    pivot = prices.pivot(index="date", columns="commodity_key", values="price_native")
    # Displaced log returns: German power clears negative, where a plain log is undefined
    # and would silently drop the whole row across every commodity.
    displacements = OmegaConf.to_container(cfg.risk.price_displacement_eur, resolve=True)
    returns = compute_log_returns(pivot, displacements).dropna()
    commodities = returns.columns.tolist()

    # --- Rolling Correlation Time Series ---
    st.subheader("Rolling 60-Day Correlation: TTF vs. German Power")
    avail = [c for c in ["TTF", "DE_POWER", "BRENT", "EUA", "NO1_POWER"] if c in commodities]
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
        for event_date, label, color in [
            ("2020-03-15", "COVID", "#6B6B6B"),
            ("2022-08-15", "Gas Crisis", "#C44536"),
            ("2023-06-01", "Normalization", "#2E7D6F"),
        ]:
            fig_ts.add_shape(type="line", x0=event_date, x1=event_date, y0=0, y1=1,
                xref="x", yref="paper", line=dict(dash="dot", color=color, width=1))
            fig_ts.add_annotation(x=event_date, y=1.02, xref="x", yref="paper",
                text=label, showarrow=False, font=dict(size=10, color=color), xanchor="left")
        fig_ts.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Correlation", xaxis_title="",
            yaxis=dict(range=[-1, 1]),
        )
        st.plotly_chart(fig_ts, width="stretch")
        st.caption("Daily-return correlation peaks near 0.34 in September 2022, then averages roughly zero from 2023 onward. The gas-power link is far weaker at daily frequency than at monthly or quarterly frequency (Epps effect).")

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
        # Degrees of freedom are estimated, not assumed — a hardcoded nu would draw an
        # ellipse for a model the panel was never fitted to.
        pair_fit = fit_t_copula(returns[["TTF", "DE_POWER"]])
        nu = pair_fit.df
        theta = np.linspace(0, 2*np.pi, 200)
        ex, ey = np.cos(theta), np.sin(theta)
        scale_t = np.sqrt(stats.f.ppf(0.95, 2, nu) * 2)
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
        st.caption(
            f"Correlation {rho:.3f}, fitted degrees of freedom ν = {nu:.1f}. At this ν the "
            "t-copula ellipse (solid navy) sits almost on top of the Gaussian one (dashed red) — "
            "this pair shows little tail dependence at daily frequency. The gap widens as ν falls."
        )
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
            f"Frobenius norm of the {len(avail)}×{len(avail)} correlation matrix. The floor is "
            f"√{len(avail)} ≈ {np.sqrt(len(avail)):.2f} (zero correlation everywhere) and the ceiling is "
            f"{len(avail)} (everything moves as one). The series stays near the floor, so the "
            f"LOW/NORMAL/HIGH split at {lo:.2f}/{hi:.2f} separates weak states from slightly less "
            "weak ones, not calm markets from crises."
        )

    with st.expander("Methodology & Interpretation"):
        st.markdown("""
**Rolling correlation time series** tracks the 60-day Pearson correlation between TTF
natural gas and German day-ahead power, computed on displaced log returns (German prices
clear negative, where a plain log is undefined). The series peaks near 0.34 in September
2022, when gas dominated the merit order, and averages close to zero from 2023 onward.

Read the level with care: this is a *daily-return* correlation, and it is much lower than
the gas-power relationship measured on prices or at lower frequency. Correlation estimated
from short-horizon returns is biased toward zero when the two series do not trade or settle
in step — the Epps effect. Day-ahead power is fixed once daily against weather and renewable
output, while gas trades continuously, so a large share of daily power variance is simply
not a gas story. Aggregating returns to monthly or quarterly recovers a far stronger link.
The near-zero daily figure is therefore a statement about measurement frequency, not
evidence that gas has stopped setting the marginal price.

**Correlation matrix** shows the N×N pairwise correlations for the selected 60-day window.
The interactive date picker lets you explore how the correlation structure evolved. The
persistently strong cell is DE_POWER–NO1_POWER, two coupled power markets; the commodity
pairs stay weak throughout. Even at the height of the August 2022 gas crisis the off-diagonal
entries only span roughly −0.1 to 0.3 — daily returns did not collapse onto a single factor,
which is the same frequency effect described above rather than evidence of diversification.

**Tail dependence plot** compares the bivariate t-copula 95% contour (solid navy) against
the Gaussian 95% contour (dashed red). Both use the same linear correlation ρ; the t-copula
adds heavy tails through its degrees of freedom ν, which is estimated from the data rather
than assumed. Coefficient of tail dependence λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) tends to
zero as ν grows, so the two contours converge when the fitted ν is large.

For TTF against German power the fit lands near ν ≈ 19 with ρ close to zero, which implies
λ ≈ 0 — the ellipses nearly coincide and there is no joint-tail clustering to speak of at
daily frequency. Counting empirical joint exceedances agrees: at the 10% level the pair
breaches jointly on the downside slightly more often than independence predicts and on the
upside slightly less. The t-copula still earns its place in the portfolio VaR, where the
full nine-series panel fits a heavier ν ≈ 13 and Brent against RBOB reaches λ ≈ 0.23, the
strongest pair in the panel. That dependence lives in other pairs, not this one.

**Correlation regime detection** computes the Frobenius norm (√Σᵢⱼ ρᵢⱼ²) of the rolling
correlation matrix and classifies each time point into LOW (<33rd percentile), NORMAL,
or HIGH (>67th percentile) regimes. The textbook motivation is that correlations converge
toward 1.0 in a crisis, so diversification fails precisely when it is most needed.

This sample does not show that pattern, and the diagnostic is kept here because the negative
result is the finding. The norm is bounded below by √N (zero correlation everywhere) and above
by N; it stays close to the lower bound throughout, so the percentile thresholds end up only
a few hundredths apart and the labels partition estimation noise rather than market states.
August 2022 classifies as LOW, not HIGH. A crisis-correlation study on this panel needs
lower-frequency returns, where the cross-commodity link is actually measurable.
""")
