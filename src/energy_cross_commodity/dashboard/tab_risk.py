"""Tab 3: Risk Command — VaR waterfall, breaches, scenario P&L."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import duckdb
from omegaconf import DictConfig
from energy_cross_commodity.risk.scenarios import SCENARIOS, run_scenario


@st.cache_data(ttl=3600)
def _fit_copula_var(prices_json: str, positions_json: str) -> dict:
    """Fit t-copula portfolio VaR. Cached to avoid refitting on tab switches.

    Args:
        prices_json: JSON-serialized pivot table of prices.
        positions_json: JSON-serialized position dict.

    Returns:
        Dict with var_95, var_99, es_975, component_var, commodities.
    """
    import json
    import pandas as pd
    from energy_cross_commodity.risk.garch import fit_univariate_garch
    from energy_cross_commodity.risk.copula import fit_t_copula
    from energy_cross_commodity.risk.var_engine import compute_portfolio_var

    pivot = pd.read_json(prices_json, orient="split")
    positions = json.loads(positions_json)

    returns = np.log(pivot / pivot.shift(1)).dropna()
    returns = returns[[c for c in pivot.columns if c in positions]]

    std_resids = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        garch_res = fit_univariate_garch(returns[col])
        std_resids[col] = garch_res.std_residuals

    copula = fit_t_copula(std_resids)
    result = compute_portfolio_var(returns, positions, copula)

    return {
        "var_95": float(result.var_95),
        "var_99": float(result.var_99),
        "es_975": float(result.es_975),
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

    positions = {k: v.notional_eur for k, v in cfg.portfolio.positions.items()}

    # Copula VaR (cached)
    var_result = _fit_copula_var(
        pivot.to_json(orient="split", date_format="iso"),
        json.dumps(positions),
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
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("VaR 95% (1-day)", f"€{var_result['var_95']:,.0f}",
        help="10,000 t-copula draws, Euler-allocated to positions")
    col2.metric("VaR 99% (1-day)", f"€{var_result['var_99']:,.0f}",
        help="99th percentile of simulated P&L distribution")
    col3.metric("ES 97.5%", f"€{var_result['es_975']:,.0f}",
        help="Expected Shortfall: mean loss beyond 97.5th percentile")

    st.caption(f"t-Copula df = {var_result.get('df', 'computed')}. Euler allocations sum within 5% of total VaR.")

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
        st.plotly_chart(fig_bt, use_container_width=True)

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

    st.subheader("Portfolio P&L Trajectory")
    returns = np.log(pivot / pivot.shift(1)).dropna()
    port_returns = returns[[c for c in returns.columns if c in positions]]
    daily_pnl = pd.Series(0.0, index=port_returns.index)
    for c in port_returns.columns:
        if c in positions:
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
        fig_pnl.add_vline(x=evt_date, line_dash="dot", line_color=color,
            annotation_text=label, annotation_position="top left")
    fig_pnl.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Cumulative P&L (EUR)")
    st.plotly_chart(fig_pnl, use_container_width=True)

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
