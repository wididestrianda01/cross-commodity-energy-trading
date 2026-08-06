"""Tab 3: Risk Command — VaR waterfall, breaches, scenario P&L."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import duckdb
from omegaconf import DictConfig, OmegaConf
from energy_cross_commodity.risk.portfolio import expand_spread_positions
from energy_cross_commodity.risk.returns import compute_log_returns
from energy_cross_commodity.risk.scenarios import SCENARIOS, run_scenario


@st.cache_data(ttl=3600)
def _fit_copula_var(prices_json: str, positions_json: str, displacements_json: str) -> dict:
    """Fit t-copula portfolio VaR. Cached to avoid refitting on tab switches.

    Args:
        prices_json: JSON-serialized pivot table of prices.
        positions_json: JSON-serialized exposures keyed by price factor.
        displacements_json: JSON-serialized displacements for series that
            can print negative prices.

    Returns:
        Dict with var_95, var_99, es_975, df, component_var, commodities.
    """
    import json
    import pandas as pd
    from energy_cross_commodity.risk.returns import compute_log_returns
    from energy_cross_commodity.risk.var_engine import compute_portfolio_var, fit_fhs_copula

    pivot = pd.read_json(prices_json, orient="split")
    positions = json.loads(positions_json)
    displacements = json.loads(displacements_json)

    factors = [c for c in pivot.columns if c in positions]
    returns = compute_log_returns(pivot[factors], displacements)

    copula, garch_fits = fit_fhs_copula(returns)
    result = compute_portfolio_var(returns, positions, copula, garch_fits=garch_fits)

    return {
        "var_95": float(result.var_95),
        "var_99": float(result.var_99),
        "es_975": float(result.es_975),
        "df": float(copula.df),
        "component_var": {k: float(v) for k, v in result.component_var.items()},
        "commodities": list(returns.columns),
    }


def render(conn: duckdb.DuckDBPyConnection, cfg: DictConfig) -> None:
    """Render Tab 3: VaR waterfall, backtest chart, scenario P&L breakdown."""
    import json

    prices = conn.execute("""
        SELECT date, commodity_key, price_native
        FROM fact_prices WHERE date >= '2019-01-01'
        ORDER BY date, commodity_key
    """).df()
    pivot = prices.pivot(index="date", columns="commodity_key", values="price_native")

    # Spreads are traded as one line but carry risk through their legs, so the
    # book is restated as signed exposures to the underlying price factors.
    positions = expand_spread_positions(
        {k: v.notional_eur for k, v in cfg.portfolio.positions.items()},
        OmegaConf.to_container(cfg.portfolio.spread_legs, resolve=True),
    )
    displacements = OmegaConf.to_container(cfg.risk.price_displacement_eur, resolve=True)

    # Copula VaR (cached)
    var_result = _fit_copula_var(
        pivot.to_json(orient="split", date_format="iso"),
        json.dumps(positions),
        json.dumps(displacements),
    )

    st.subheader("Portfolio VaR Decomposition")
    comp_var = var_result["component_var"]
    fig = go.Figure(go.Waterfall(
        name="VaR", orientation="v",
        measure=["relative"] * len(comp_var) + ["total"],
        x=list(comp_var.keys()) + ["Total"],
        y=list(comp_var.values()) + [var_result["var_95"]],
        connector={"line": {"color": "#6B6B6B"}},
        decreasing={"marker": {"color": "#C44536"}},
        increasing={"marker": {"color": "#C44536"}},
        totals={"marker": {"color": "#00003C"}},
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        title="Euler Component VaR (t-Copula Simulation, 10K draws)")
    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("VaR 95% (1-day)", f"€{var_result['var_95']:,.0f}",
        help="10,000 t-copula draws, Euler-allocated to positions")
    col2.metric("VaR 99% (1-day)", f"€{var_result['var_99']:,.0f}",
        help="99th percentile of simulated P&L distribution")
    col3.metric("ES 97.5%", f"€{var_result['es_975']:,.0f}",
        help="Expected Shortfall: mean loss beyond 97.5th percentile")

    st.caption(
        f"t-copula ν = {var_result['df']:.2f} on {len(var_result['commodities'])} price factors. "
        "Component VaRs are Euler allocations and sum to total VaR up to simulation noise."
    )

    # --- VaR Backtesting Chart ---
    st.subheader("VaR Backtesting")
    bt_df = conn.execute("SELECT date, pnl, var_estimate FROM var_backtest ORDER BY date").df()

    if bt_df.empty:
        st.info("No backtest data. Run `python -m energy_cross_commodity.pipeline` to populate.")
    else:
        bt_df["date"] = pd.to_datetime(bt_df["date"])
        bt_df["breach"] = bt_df["pnl"] < -bt_df["var_estimate"]

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_df["pnl"], mode="lines",
            line=dict(color="#6B6B6B", width=0.8), name="Daily P&L",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=-bt_df["var_estimate"], mode="lines",
            line=dict(color="#00003C", width=1.5), name="VaR 95% (lower bound)",
            fill=None,
        ))
        breaches_df = bt_df[bt_df["breach"]]
        if not breaches_df.empty:
            fig_bt.add_trace(go.Scatter(
                x=breaches_df["date"], y=breaches_df["pnl"],
                mode="markers", marker=dict(color="#C44536", size=6, symbol="x"),
                name=f"Breaches ({len(breaches_df)})",
            ))

        fig_bt.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="P&L (EUR)", xaxis_title="",
            showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_bt, width="stretch")

        from energy_cross_commodity.risk.var_engine import kupiec_test
        breaches = int(bt_df["breach"].sum())
        total = len(bt_df)
        kt = kupiec_test(breaches, total, 0.95)

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("VaR 95% Breaches", f"{breaches} / {total}",
            delta=f"Expected ~{total*0.05:.0f}", delta_color="off")
        bc2.metric("Kupiec POF p-value", f"{kt['p_value']:.3f}",
            delta="Well-calibrated" if kt["p_value"] > 0.05 else "Check model",
            delta_color="normal" if kt["p_value"] > 0.05 else "inverse")
        coverage = breaches / total if total > 0 else 0
        bc3.metric("Actual Coverage", f"{coverage:.1%}",
            delta="vs 5.0% target", delta_color="off")
    st.subheader("Stress Scenario P&L")
    scenario_choice = st.selectbox("Select scenario", list(SCENARIOS.keys()), format_func=lambda x: SCENARIOS[x].name)
    scenario = SCENARIOS[scenario_choice]
    result = run_scenario(positions, scenario)

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
    st.plotly_chart(fig2, width="stretch")
    st.caption(scenario.description)

    st.subheader("Portfolio P&L Trajectory")
    port_returns = compute_log_returns(
        pivot[[c for c in pivot.columns if c in positions]], displacements
    )
    daily_pnl = pd.Series(0.0, index=port_returns.index)
    for c in port_returns.columns:
        daily_pnl += port_returns[c] * positions[c]
    cum_pnl = daily_pnl.cumsum()

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl.values, mode="lines",
        line=dict(color="#00003C", width=1.5), name="Cumulative P&L",
        fill="tozeroy", fillcolor="#2E7D6F" if cum_pnl.values[-1] > 0 else "#C44536",
    ))
    for evt_date, label, color in [
        ("2020-03-15", "COVID", "#6B6B6B"),
        ("2022-08-15", "Gas Crisis", "#C44536"),
        ("2024-01-01", "Recovery", "#2E7D6F"),
    ]:
        fig_pnl.add_shape(type="line", x0=evt_date, x1=evt_date, y0=0, y1=1,
            xref="x", yref="paper", line=dict(dash="dot", color=color, width=1))
        fig_pnl.add_annotation(x=evt_date, y=1.02, xref="x", yref="paper",
            text=label, showarrow=False, font=dict(size=10, color=color), xanchor="left")
    fig_pnl.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Cumulative P&L (EUR)")
    st.plotly_chart(fig_pnl, width="stretch")

    total_ret = float(cum_pnl.values[-1])
    daily_vals = daily_pnl.dropna()
    sharpe = float(daily_vals.mean() / daily_vals.std() * np.sqrt(252)) if daily_vals.std() > 0 else 0.0
    peak = cum_pnl.expanding().max()
    drawdown = (cum_pnl - peak) / peak.abs().replace(0, 1)
    max_dd = float(drawdown.min())
    pct_pos = float((daily_vals > 0).mean() * 100)

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Total Return", f"€{total_ret:,.0f}")
    pc2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    pc3.metric("Max Drawdown", f"{max_dd:.1%}")
    pc4.metric("% Positive Days", f"{pct_pos:.0f}%")

    with st.expander("Methodology & Interpretation"):
        st.markdown(f"""
**Portfolio VaR decomposition** uses Euler-allocated component VaR from 10,000 filtered
historical simulation draws. Each position's bar represents its marginal contribution to
total portfolio risk, estimated as a kernel-smoothed conditional expectation of the
position's return given that portfolio P&L sits at −VaR. The component VaRs sum to the
total diversified VaR. This answers the desk-level question: "which position should I cut
to reduce risk by €X?"

The model pipeline: (1) univariate GARCH(1,1) with Student-t errors fitted per commodity
→ standardized residuals and a one-step-ahead volatility forecast; (2) multivariate
t-copula fitted by canonical maximum pseudo-likelihood on the *ranks* of those residuals
→ correlation matrix R and degrees of freedom ν; (3) 10,000 dependent uniform draws from
the fitted copula, mapped back through each commodity's *empirical* residual quantile
function and rescaled by its forecast volatility; (4) portfolio P&L = Σ(positions ×
simulated returns); (5) VaR₉₅ = −Q₀.₀₅(P&L), ES₉₇.₅ = −E[P&L | P&L ≤ −VaR₉₇.₅].

**VaR backtesting** evaluates the model's predictive accuracy. The model is re-estimated on
a rolling {int(cfg.risk.rolling_window)}-day window and its 95% VaR compared against the
realized next-day P&L. The Kupiec (1995) proportion-of-failures test asks whether the
observed breach rate is consistent with the model's 5% expected rate; a p-value > 0.05 means
correct unconditional coverage cannot be rejected. That is weak evidence rather than
validation — the test has limited power at this sample size, and it says nothing about
whether breaches cluster, which is what a conditional-coverage test addresses.

**Stress scenarios** are deterministic full revaluations: the scenario fixes where every
price goes, so each leg's P&L is its signed exposure times its shock and the total is their
sum. No correlation is involved, because specifying the joint move is exactly what a
scenario does — correlation matters for VaR, where the moves are unknown. Three scenarios
are modeled:
- *Gas crisis*: TTF +300%, power +200%, carbon +50%, Brent +30%. A Nord Stream-style
  supply shock.
- *Global recession*: Brent −40%, TTF −30%, power −25%, carbon −20%. Demand destruction
  where all risk assets sell off together.
- *Energy transition*: carbon +200% (€150/t), coal −40%, Brent −30%, power −10%. A
  structural shift where carbon policy drives fuel switching and renewables cannibalize
  power prices.

**Position mapping.** The book is quoted as outright legs plus a short 3-2-1 crack and a
short spark spread, but a spread level is a price difference: it goes negative and has no
log return, so it cannot be a risk factor. Each spread is therefore decomposed into its
underlyings — the delta of a spread is the sum of the deltas of its legs — and the risk
engine runs on the netted factor exposures. Day-ahead power prints negative on oversupply
days, so power returns use a displaced log return ln((P+k)/(P₋₁+k)); without it a plain
log return is undefined and a panel-wide dropna would delete exactly those days.

**Portfolio P&L trajectory** shows the cumulative profit/loss of the hypothetical integrated
energy book. Performance metrics (Sharpe ratio, maximum drawdown, percent positive days)
are computed on daily P&L. The annotations highlight how the portfolio performed through the
key market events of 2020-2024.
""")
